import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.providers.xiaomi_mimo_provider.provider import XiaomiMimoProvider


def _api_provider(api_key: str = "mimo-key") -> dict:
    return {
        "api_key": api_key,
        "base_url": "https://relay.example/v1",
        "default_headers": {},
        "default_query": {},
    }


def _request(
    *,
    messages: list[dict] | None = None,
    api_key: str = "mimo-key",
    stream: bool = False,
) -> dict:
    return {
        "model_info": {
            "model_identifier": "mimo-v2.5-pro",
            "force_stream_mode": stream,
            "extra_params": {},
        },
        "api_provider": _api_provider(api_key),
        "message_list": messages or [{"role": "user", "parts": [{"type": "text", "text": "查询天气"}]}],
        "tool_options": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查询天气",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def _tool_response(call_id: str = "call_1", reasoning: str = "先查询天气") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": reasoning,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": {},
    }


def _history(tool_call: dict) -> list[dict]:
    return [
        {"role": "user", "parts": [{"type": "text", "text": "查询天气"}]},
        {"role": "assistant", "parts": [], "tool_calls": [tool_call]},
        {
            "role": "tool",
            "parts": [{"type": "text", "text": "晴天"}],
            "tool_call_id": tool_call["id"],
            "tool_name": "weather",
        },
    ]


@pytest.mark.parametrize("remove_metadata", [False, True])
@pytest.mark.asyncio
async def test_mimo_reasoning_replays_from_metadata_or_sqlite(
    tmp_path: Path,
    remove_metadata: bool,
) -> None:
    request_bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        request_bodies.append(cast(dict, body))
        if len(request_bodies) == 1:
            return httpx.Response(200, json=_tool_response())
        assistant = next(message for message in body["messages"] if message["role"] == "assistant")
        assert assistant["reasoning_content"] == "先查询天气"
        return httpx.Response(200, json={"choices": [{"message": {"content": "晴天"}}], "usage": {}})

    store = PluginStateStore(tmp_path / "state.sqlite3")
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    first = await provider.get_response(_request())
    tool_call = deepcopy(first["tool_calls"][0])
    assert tool_call["extra_content"]["xiaomi_mimo"]["reasoning_content"] == "先查询天气"
    if remove_metadata:
        del tool_call["extra_content"]["xiaomi_mimo"]["reasoning_content"]

    second = await provider.get_response(_request(messages=_history(tool_call)))

    assert second["content"] == "晴天"
    assert len(request_bodies) == 2
    await store.close()


@pytest.mark.asyncio
async def test_mimo_reasoning_conflict_is_exposed_before_request(
    tmp_path: Path,
) -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=_tool_response())

    store = PluginStateStore(tmp_path / "state.sqlite3")
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    first = await provider.get_response(_request())
    tool_call = deepcopy(first["tool_calls"][0])
    tool_call["extra_content"]["xiaomi_mimo"]["reasoning_content"] = "冲突内容"

    with pytest.raises(ValueError, match="值不一致"):
        await provider.get_response(_request(messages=_history(tool_call)))
    assert request_count == 1
    await store.close()


@pytest.mark.asyncio
async def test_mimo_reasoning_is_isolated_by_credentials(tmp_path: Path) -> None:
    store = PluginStateStore(tmp_path / "state.sqlite3")
    first_provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_tool_response())),
        state_store=store,
    )
    first = await first_provider.get_response(_request(api_key="account-a"))
    tool_call = deepcopy(first["tool_calls"][0])
    del tool_call["extra_content"]["xiaomi_mimo"]["reasoning_content"]

    second_provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        state_store=store,
    )
    with pytest.raises(ValueError, match="call_1.*reasoning_content"):
        await second_provider.get_response(_request(messages=_history(tool_call), api_key="account-b"))
    await store.close()


@pytest.mark.asyncio
async def test_mimo_enabled_thinking_rejects_tool_call_without_reasoning(
    tmp_path: Path,
) -> None:
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_tool_response(reasoning=""))),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    with pytest.raises(ValueError, match="reasoning_content"):
        await provider.get_response(_request())


@pytest.mark.asyncio
async def test_mimo_stream_preserves_reasoning_on_tool_call(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content.decode())["stream"] is True
        content = "".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先思考"}}]}\n\n',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_stream","function":{"name":"weather","arguments":"{}"}}]}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"total_tokens":3}}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    store = PluginStateStore(tmp_path / "state.sqlite3")
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_force_disable_thinking=False),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    result = await provider.get_response(_request(stream=True))

    assert result["reasoning_content"] == "先思考"
    extra = result["tool_calls"][0]["extra_content"]["xiaomi_mimo"]
    assert extra["reasoning_content"] == "先思考"
    assert extra["raw_arguments"] == "{}"
    await store.close()


@pytest.mark.asyncio
async def test_mimo_reasoning_persistence_is_independent_from_hidden_host_reasoning(
    tmp_path: Path,
) -> None:
    store = PluginStateStore(tmp_path / "state.sqlite3")
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(
            mimo_force_disable_thinking=False,
            reasoning_parse_mode="none",
        ),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_tool_response())),
        state_store=store,
    )

    result = await provider.get_response(_request())

    assert "reasoning_content" not in result
    extra = result["tool_calls"][0]["extra_content"]["xiaomi_mimo"]
    assert extra["reasoning_content"] == "先查询天气"
    await store.close()


@pytest.mark.asyncio
async def test_mimo_stream_reasoning_persistence_is_independent_from_hidden_host_reasoning(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        content = "".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先思考"}}]}\n\n',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_stream_hidden","function":{"name":"weather","arguments":"{}"}}]}}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    store = PluginStateStore(tmp_path / "state.sqlite3")
    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(
            mimo_force_disable_thinking=False,
            reasoning_parse_mode="none",
        ),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )

    result = await provider.get_response(_request(stream=True))

    assert "reasoning_content" not in result
    extra = result["tool_calls"][0]["extra_content"]["xiaomi_mimo"]
    assert extra["reasoning_content"] == "先思考"
    await store.close()
