import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.providers.common.httpx import HttpxClientConfig, HttpxProviderError
from src.providers.volcengine_ark_provider.prefix_cache import (
    PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS,
    PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES,
    PREFIX_CACHE_INELIGIBLE_TTL_SECONDS,
    PREFIX_CACHE_NAMESPACE,
    PrefixCacheManager,
)
from src.providers.volcengine_ark_provider.provider import (
    VolcengineArkResponsesProvider,
)


def _client_config(*, api_key: str = "ark-key", base_url: str = "https://ark.example/api/v3") -> HttpxClientConfig:
    return HttpxClientConfig(
        base_url=base_url,
        default_headers={"Authorization": f"Bearer {api_key}"},
    )


def _body(*, system_text: str = "system prompt") -> dict:
    return {
        "model": "doubao-test",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "你好"}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "look up data",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "thinking": {"type": "disabled"},
        "stream": False,
        "max_output_tokens": 128,
        "temperature": 0.3,
    }


def _request(*, stream: bool = False, system_text: str = "system prompt") -> dict:
    return {
        "model_info": {
            "model_identifier": "doubao-test",
            "force_stream_mode": stream,
            "extra_params": {"thinking": {"type": "disabled"}},
        },
        "api_provider": {
            "api_key": "ark-key",
            "auth_type": "bearer",
            "base_url": "https://ark.example/api/v3",
            "default_headers": {},
            "default_query": {},
        },
        "message_list": [
            {"role": "system", "parts": [{"type": "text", "text": system_text}]},
            {"role": "user", "parts": [{"type": "text", "text": "你好"}]},
        ],
        "tool_options": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "look up data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "temperature": 0.3,
        "max_tokens": 128,
    }


def _completed_response() -> dict:
    return {
        "id": "resp_answer",
        "model": "doubao-test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "回答"}],
            }
        ],
        "usage": {"input_tokens": 300, "output_tokens": 2, "total_tokens": 302},
    }


def _json_body(request: httpx.Request) -> dict:
    body = json.loads(request.content.decode("utf-8"))
    assert isinstance(body, dict)
    return cast(dict, body)


def test_prefix_cache_key_covers_scope_model_prefix_thinking_and_tools(
    tmp_path: Path,
) -> None:
    manager = PrefixCacheManager(PluginStateStore(tmp_path / "state.sqlite3"), ttl_seconds=259200)
    base = _body()
    base_plan = manager.build_plan(base, client_config=_client_config())
    assert base_plan is not None

    variants: list[tuple[dict, HttpxClientConfig]] = []
    changed_model = deepcopy(base)
    changed_model["model"] = "doubao-other"
    variants.append((changed_model, _client_config()))
    changed_prefix = _body(system_text="other system prompt")
    variants.append((changed_prefix, _client_config()))
    changed_thinking = deepcopy(base)
    changed_thinking["thinking"] = {"type": "enabled"}
    variants.append((changed_thinking, _client_config()))
    changed_tools = deepcopy(base)
    changed_tools["tools"][0]["name"] = "other_tool"
    variants.append((changed_tools, _client_config()))
    variants.append((deepcopy(base), _client_config(api_key="other-key")))
    variants.append((deepcopy(base), _client_config(base_url="https://other.example/api/v3")))
    header_override_plan = manager.build_plan(
        deepcopy(base),
        client_config=_client_config(),
        headers={"authorization": "Bearer other-key"},
    )
    query_override_plan = manager.build_plan(
        deepcopy(base),
        client_config=_client_config(),
        query={"api_key": "other-key"},
    )
    assert header_override_plan is not None
    assert query_override_plan is not None
    assert header_override_plan.cache_key != base_plan.cache_key
    assert query_override_plan.cache_key != base_plan.cache_key

    variant_keys = {
        plan.cache_key
        for body, config in variants
        if (plan := manager.build_plan(body, client_config=config)) is not None
    }
    assert len(variant_keys) == len(variants)
    assert base_plan.cache_key not in variant_keys


@pytest.mark.parametrize(
    "field,value",
    [
        ("instructions", "system instruction"),
        ("store", False),
        ("caching", {"type": "enabled"}),
        ("previous_response_id", "resp_manual"),
        ("text", {"format": {"type": "json_schema"}}),
        ("tools", [{"type": "web_search"}]),
    ],
)
def test_prefix_cache_incompatible_requests_are_not_modified(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manager = PrefixCacheManager(PluginStateStore(tmp_path / "state.sqlite3"), ttl_seconds=259200)
    body = _body()
    body[field] = value
    assert manager.build_plan(body, client_config=_client_config()) is None


@pytest.mark.asyncio
async def test_prefix_cache_create_then_reuse_sends_only_uncached_suffix(
    tmp_path: Path,
) -> None:
    captured: list[tuple[str, dict, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = _json_body(request)
        captured.append((request.url.path, body, request.headers.get("X-Client-Request-Id")))
        if request.url.path.endswith("/tokenization"):
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        if body.get("caching") == {"type": "enabled", "prefix": True}:
            return httpx.Response(200, json={"id": "resp_prefix", "status": "completed", "output": []})
        return httpx.Response(200, json=_completed_response())

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    first = await provider.get_response(_request())
    second = await provider.get_response(_request())
    await store.close()

    assert first["content"] == "回答"
    assert second["content"] == "回答"
    assert [path for path, _, _ in captured] == [
        "/api/v3/tokenization",
        "/api/v3/responses",
        "/api/v3/responses",
        "/api/v3/responses",
    ]
    tokenization_body = captured[0][1]
    create_body = captured[1][1]
    first_reuse_body = captured[2][1]
    second_reuse_body = captured[3][1]
    assert tokenization_body == {"model": "doubao-test", "text": "system prompt"}
    assert "stream" not in create_body
    assert "max_output_tokens" not in create_body
    assert create_body["input"][0]["role"] == "system"
    assert create_body["tools"][0]["name"] == "lookup"
    assert create_body["thinking"] == {"type": "disabled"}
    for reuse_body in (first_reuse_body, second_reuse_body):
        assert reuse_body["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "你好"}],
            }
        ]
        assert reuse_body["previous_response_id"] == "resp_prefix"
        assert reuse_body["stream"] is False
        assert reuse_body["max_output_tokens"] == 128
        assert reuse_body["thinking"] == {"type": "disabled"}
        assert "tools" not in reuse_body
        assert "caching" not in reuse_body
    request_ids = [request_id for _, _, request_id in captured]
    assert all(request_ids)
    assert len(set(request_ids)) == len(request_ids)


@pytest.mark.asyncio
async def test_short_prefix_uses_normal_request_without_creating_cache(
    tmp_path: Path,
) -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = _json_body(request)
        captured.append(body)
        if request.url.path.endswith("/tokenization"):
            return httpx.Response(200, json={"data": [{"total_tokens": 20}]})
        return httpx.Response(200, json=_completed_response())

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    await provider.get_response(_request())
    await provider.get_response(_request())
    await store.close()

    assert len(captured) == 3
    assert captured[0] == {"model": "doubao-test", "text": "system prompt"}
    for body in captured[1:]:
        assert [message["role"] for message in body["input"]] == ["system", "user"]
        assert "previous_response_id" not in body
        assert "caching" not in body


@pytest.mark.asyncio
async def test_same_key_concurrent_requests_create_cache_once(tmp_path: Path) -> None:
    tokenization_count = 0
    create_count = 0
    actual_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal actual_count, create_count, tokenization_count
        body = _json_body(request)
        if request.url.path.endswith("/tokenization"):
            tokenization_count += 1
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        if body.get("caching") == {"type": "enabled", "prefix": True}:
            create_count += 1
            return httpx.Response(200, json={"id": "resp_prefix", "status": "completed", "output": []})
        actual_count += 1
        return httpx.Response(200, json=_completed_response())

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    await asyncio.gather(provider.get_response(_request()), provider.get_response(_request()))
    await store.close()

    assert tokenization_count == 1
    assert create_count == 1
    assert actual_count == 2


@pytest.mark.asyncio
async def test_reuse_4xx_invalidates_entry_without_retrying_current_request(
    tmp_path: Path,
) -> None:
    tokenization_count = 0
    create_count = 0
    actual_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal actual_count, create_count, tokenization_count
        body = _json_body(request)
        if request.url.path.endswith("/tokenization"):
            tokenization_count += 1
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        if body.get("caching") == {"type": "enabled", "prefix": True}:
            create_count += 1
            return httpx.Response(
                200,
                json={
                    "id": f"resp_prefix_{create_count}",
                    "status": "completed",
                    "output": [],
                },
            )
        actual_count += 1
        if actual_count == 2:
            return httpx.Response(404, json={"error": {"message": "response not found"}})
        return httpx.Response(200, json=_completed_response())

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
            volcengine_max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    await provider.get_response(_request())
    with pytest.raises(HttpxProviderError, match="404"):
        await provider.get_response(_request())
    await provider.get_response(_request())
    await store.close()

    assert tokenization_count == 2
    assert create_count == 2
    assert actual_count == 3


@pytest.mark.asyncio
async def test_entry_inside_expiry_safety_window_is_recreated(tmp_path: Path) -> None:
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    manager = PrefixCacheManager(store, ttl_seconds=3600)
    config = _client_config()
    body = _body()
    plan = manager.build_plan(body, client_config=config)
    assert plan is not None
    await store.set(
        PREFIX_CACHE_NAMESPACE,
        plan.cache_key,
        {
            "response_id": "resp_old",
            "expires_at": 1050.0,
            "input_tokens": 300,
            "created_at": 900.0,
        },
        expires_at=1050.0,
        now=900.0,
    )
    tokenization_count = 0
    create_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_count, tokenization_count
        request_body = _json_body(request)
        if request.url.path.endswith("/tokenization"):
            tokenization_count += 1
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        assert request_body["caching"] == {"type": "enabled", "prefix": True}
        create_count += 1
        return httpx.Response(200, json={"id": "resp_new", "status": "completed", "output": []})

    async with httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    ) as client:
        resolution = await manager.resolve(
            client,
            responses_path="responses",
            tokenization_path="tokenization",
            body=body,
            headers={},
            query={},
            client_config=config,
            max_retries=0,
            retry_interval=0.0,
            now=1000.0,
        )
    await store.close()

    assert resolution.body["previous_response_id"] == "resp_new"
    assert tokenization_count == 1
    assert create_count == 1


@pytest.mark.asyncio
async def test_invalid_persisted_cache_schema_is_exposed(tmp_path: Path) -> None:
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    manager = PrefixCacheManager(store, ttl_seconds=3600)
    config = _client_config()
    body = _body()
    plan = manager.build_plan(body, client_config=config)
    assert plan is not None
    await store.set(
        PREFIX_CACHE_NAMESPACE,
        plan.cache_key,
        {
            "response_id": 123,
            "expires_at": 5000.0,
            "input_tokens": 300,
            "created_at": 900.0,
        },
        expires_at=5000.0,
        now=900.0,
    )

    async def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"不应发送网络请求: {request.url}")

    async with httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(unexpected_request),
    ) as client:
        with pytest.raises(ValueError):
            await manager.resolve(
                client,
                responses_path="responses",
                tokenization_path="tokenization",
                body=body,
                headers={},
                query={},
                client_config=config,
                max_retries=0,
                retry_interval=0.0,
                now=1000.0,
            )
    await store.close()


def test_prefix_cache_plan_handles_multiple_system_messages_and_missing_thinking(
    tmp_path: Path,
) -> None:
    manager = PrefixCacheManager(
        PluginStateStore(tmp_path / "state.sqlite3"),
        ttl_seconds=259200,
    )
    body = _body()
    body["input"].insert(
        1,
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "second system"}],
        },
    )

    plan = manager.build_plan(body, client_config=_client_config())
    assert plan is not None
    assert [cast(dict[str, object], message)["role"] for message in plan.prefix_input] == ["system", "system"]
    assert plan.tokenization_texts == ["system prompt", "second system"]
    assert [cast(dict[str, object], message)["role"] for message in plan.suffix_input] == ["user"]

    missing_thinking = deepcopy(body)
    missing_thinking.pop("thinking")
    missing_plan = manager.build_plan(
        missing_thinking,
        client_config=_client_config(),
    )
    assert missing_plan is not None
    assert missing_plan.cache_key != plan.cache_key


def test_short_prefix_negative_cache_is_bounded_and_expires(tmp_path: Path) -> None:
    manager = PrefixCacheManager(
        PluginStateStore(tmp_path / "state.sqlite3"),
        ttl_seconds=259200,
    )
    for index in range(PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES + 10):
        manager._remember_ineligible(f"key-{index}", now=1000.0)

    assert len(manager._ineligible_until) == PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES
    newest_key = f"key-{PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES + 9}"
    assert manager._is_ineligible(
        newest_key,
        now=1000.0 + PREFIX_CACHE_INELIGIBLE_TTL_SECONDS - 1,
    )
    assert not manager._is_ineligible(
        newest_key,
        now=1000.0 + PREFIX_CACHE_INELIGIBLE_TTL_SECONDS,
    )
    manager._prune_ineligible(now=1000.0 + PREFIX_CACHE_INELIGIBLE_TTL_SECONDS)
    assert manager._ineligible_until == {}


@pytest.mark.asyncio
async def test_different_prefix_keys_create_in_parallel_and_release_locks(
    tmp_path: Path,
) -> None:
    tokenizations_started = 0
    both_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tokenizations_started
        request_body = _json_body(request)
        if request.url.path.endswith("/tokenization"):
            tokenizations_started += 1
            if tokenizations_started == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        assert request_body["caching"] == {"type": "enabled", "prefix": True}
        return httpx.Response(
            200,
            json={
                "id": f"resp_{tokenizations_started}",
                "status": "completed",
                "output": [],
            },
        )

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    manager = PrefixCacheManager(store, ttl_seconds=259200)
    config = _client_config()
    async with httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    ) as client:
        first, second = await asyncio.gather(
            manager.resolve(
                client,
                responses_path="responses",
                tokenization_path="tokenization",
                body=_body(system_text="first prefix"),
                headers={},
                query={},
                client_config=config,
                max_retries=0,
                retry_interval=0.0,
            ),
            manager.resolve(
                client,
                responses_path="responses",
                tokenization_path="tokenization",
                body=_body(system_text="second prefix"),
                headers={},
                query={},
                client_config=config,
                max_retries=0,
                retry_interval=0.0,
            ),
        )
    await store.close()

    assert tokenizations_started == 2
    assert first.body["previous_response_id"]
    assert second.body["previous_response_id"]
    assert len(manager._locks) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caching", {"type": "enabled"}),
        ("previous_response_id", "resp_manual"),
    ],
)
async def test_manual_cache_parameters_are_not_rewritten_by_provider(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        captured.append(_json_body(request))
        return httpx.Response(200, json=_completed_response())

    request = _request()
    request["model_info"]["extra_params"][field] = value
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    await provider.get_response(request)
    await store.close()

    assert len(captured) == 1
    assert captured[0][field] == value
    assert [message["role"] for message in captured[0]["input"]] == ["system", "user"]
    assert captured[0]["tools"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_stage", "error_pattern"),
    [
        ("tokenization", "ARK.*system"),
        ("creation", "ARK.*ID"),
    ],
)
async def test_prefix_cache_protocol_errors_are_exposed(
    tmp_path: Path,
    failure_stage: str,
    error_pattern: str,
) -> None:
    actual_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal actual_requests
        request_body = _json_body(request)
        if request.url.path.endswith("/tokenization"):
            if failure_stage == "tokenization":
                return httpx.Response(200, json={"data": []})
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        if request_body.get("caching") == {"type": "enabled", "prefix": True}:
            assert failure_stage == "creation"
            return httpx.Response(200, json={"status": "completed", "output": []})
        actual_requests += 1
        return httpx.Response(200, json=_completed_response())

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
            volcengine_max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    with pytest.raises(ValueError, match=error_pattern):
        await provider.get_response(_request())
    await store.close()

    assert actual_requests == 0


@pytest.mark.asyncio
async def test_prefix_cache_preserves_stream_mode_only_for_actual_request(
    tmp_path: Path,
) -> None:
    captured: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body = _json_body(request)
        captured.append(request_body)
        if request.url.path.endswith("/tokenization"):
            return httpx.Response(200, json={"data": [{"total_tokens": 300}]})
        if request_body.get("caching") == {"type": "enabled", "prefix": True}:
            return httpx.Response(
                200,
                json={"id": "resp_prefix", "status": "completed", "output": []},
            )
        event = {
            "type": "response.completed",
            "response": _completed_response(),
        }
        return httpx.Response(
            200,
            content=(f"event: response.completed\ndata: {json.dumps(event, ensure_ascii=False)}\n\n").encode(),
            headers={"Content-Type": "text/event-stream"},
        )

    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_prefix_cache_enabled=True,
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    result = await provider.get_response(_request(stream=True))
    await store.close()

    assert result["content"] == "回答"
    assert len(captured) == 3
    assert "stream" not in captured[0]
    assert "stream" not in captured[1]
    actual_body = captured[2]
    assert actual_body["stream"] is True
    assert actual_body["max_output_tokens"] == 128
    assert actual_body["previous_response_id"] == "resp_prefix"
    assert [message["role"] for message in actual_body["input"]] == ["user"]
    assert "tools" not in actual_body


@pytest.mark.asyncio
async def test_prefix_cache_periodically_cleans_expired_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    manager = PrefixCacheManager(store, ttl_seconds=259200)
    cleanup_times: list[float | None] = []
    original_delete_expired = store.delete_expired

    async def delete_expired(
        namespace: str | None = None,
        *,
        now: float | None = None,
    ) -> int:
        cleanup_times.append(now)
        return await original_delete_expired(namespace, now=now)

    monkeypatch.setattr(store, "delete_expired", delete_expired)

    await manager._delete_expired_if_due(now=1000.0)
    await manager._delete_expired_if_due(
        now=1000.0 + PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS - 1,
    )
    await manager._delete_expired_if_due(
        now=1000.0 + PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS,
    )
    await store.close()

    assert cleanup_times == [
        1000.0,
        1000.0 + PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS,
    ]
