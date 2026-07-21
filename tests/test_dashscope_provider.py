import base64
import json
from typing import cast

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.core.parameter_policy import (
    ParameterPolicy,
    ParameterPolicyRegistry,
    ProviderCapabilityPolicies,
)
from src.providers.dashscope_provider.chat import (
    DASHSCOPE_GENERATION_ENDPOINT,
    DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
)
from src.providers.dashscope_provider.embeddings import (
    DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
    DASHSCOPE_TEXT_EMBEDDING_ENDPOINT,
)
from src.providers.dashscope_provider.provider import DashScopeProvider
from src.schemas.host_snapshots import AudioTranscriptionRequestSnapshot


def _response_request_with_image(
    *,
    with_image: bool = True,
    stream: bool = False,
    extra_params: dict | None = None,
) -> dict:
    parts: list[dict] = [{"type": "text", "text": "描述这张图片"}]
    if with_image:
        parts.append(
            {
                "type": "image",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lm8cAAAAASUVORK5CYII=",
                "image_format": "png",
            }
        )
    model_info: dict = {
        "model_identifier": "qwen-vl-plus",
        "force_stream_mode": stream,
        "extra_params": extra_params or {},
    }
    return {
        "model_info": model_info,
        "api_provider": _api_provider(),
        "message_list": [{"role": "user", "parts": parts}],
        "tool_options": [],
    }


def _api_provider(*, base_url: str | None = "https://dashscope.aliyuncs.com/api/v1") -> dict:
    return {
        "api_key": "dashscope-key",
        "auth_type": "bearer",
        "base_url": base_url,
        "default_headers": {},
        "default_query": {},
    }


def _response_request(
    *,
    stream: bool = False,
    base_url: str | None = "https://dashscope.aliyuncs.com/api/v1",
    extra_params: dict | None = None,
    response_format: dict | None = None,
    model_temperature: int | float | None = None,
    model_max_tokens: int | None = None,
    request_temperature: int | float | None = 0.2,
    request_max_tokens: int | None = 64,
) -> dict:
    model_info: dict = {
        "model_identifier": "qwen-plus",
        "force_stream_mode": stream,
        "extra_params": extra_params or {},
    }
    if model_temperature is not None:
        model_info["temperature"] = model_temperature
    if model_max_tokens is not None:
        model_info["max_tokens"] = model_max_tokens
    request: dict = {
        "model_info": model_info,
        "api_provider": _api_provider(base_url=base_url),
        "message_list": [{"role": "user", "parts": [{"type": "text", "text": "你好"}]}],
        "tool_options": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "look up data",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        "temperature": request_temperature,
        "max_tokens": request_max_tokens,
    }
    if response_format is not None:
        request["response_format"] = response_format
    return request


def _embedding_request(
    model: str,
    *,
    base_url: str | None = "https://dashscope.aliyuncs.com/api/v1",
    extra_params: dict | None = None,
) -> dict:
    return {
        "model_info": {"model_identifier": model, "extra_params": extra_params or {}},
        "api_provider": _api_provider(base_url=base_url),
        "embedding_input": "文本向量",
    }


def _as_list(value: object) -> list:
    assert isinstance(value, list)
    return cast(list, value)


def _as_mapping(value: object) -> dict:
    assert isinstance(value, dict)
    return cast(dict, value)


@pytest.mark.asyncio
async def test_dashscope_generation_uses_api_v1_base_body_and_tool_calls() -> None:
    captured_path: list[str] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "request_id": "req_1",
                "output": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "回答",
                                "reasoning_content": "思考",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"q":"x"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
        )

    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(extra_params={"top_p": 0.9}))

    assert captured_path == ["/api/v1/" + DASHSCOPE_GENERATION_ENDPOINT]
    body = captured_body[0]
    assert body["model"] == "qwen-plus"
    assert body["input"] == {"messages": [{"role": "user", "content": "你好"}]}
    parameters = _as_mapping(body["parameters"])
    assert parameters["result_format"] == "message"
    assert parameters["temperature"] == 0.2
    assert parameters["max_tokens"] == 64
    assert parameters["top_p"] == 0.9
    assert isinstance(parameters["tools"], list)
    assert result["content"] == "回答"
    assert result["reasoning_content"] == "思考"
    tool_calls = _as_list(result["tool_calls"])
    first_tool_call = _as_mapping(tool_calls[0])
    assert first_tool_call["id"] == "call_1"
    assert first_tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 5
    raw_data = _as_mapping(result["raw_data"])
    assert raw_data["request_id"] == "req_1"


@pytest.mark.asyncio
async def test_dashscope_generation_host_only_base_adds_api_v1_prefix() -> None:
    captured_path: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(base_url="https://dashscope.aliyuncs.com"))

    assert captured_path == ["/api/v1/" + DASHSCOPE_GENERATION_ENDPOINT]
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(base_url="https://relay.example/custom"))

    assert captured_url == [f"https://dashscope.aliyuncs.com/api/v1/{DASHSCOPE_GENERATION_ENDPOINT}"]
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(dashscope_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(base_url="https://relay.example/custom"))

    assert captured_url == [f"https://relay.example/custom/api/v1/{DASHSCOPE_GENERATION_ENDPOINT}"]
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_stream_sends_header_and_merges_incremental_chunks() -> None:
    captured_headers: list[dict[str, str | None]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(
            {
                "accept": request.headers.get("Accept"),
                "accel": request.headers.get("X-Accel-Buffering"),
                "sse": request.headers.get("X-DashScope-SSE"),
            }
        )
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        content = (
            "event: result\n"
            'data: {"output":{"choices":[{"message":{"content":"你"}}]},"usage":{"input_tokens":1}}\n\n'
            "event: result\n"
            'data: {"output":{"choices":[{"message":{"content":"好"}}]},"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}\n\n'
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(stream=True))

    assert captured_headers == [{"accept": "text/event-stream", "accel": "no", "sse": "enable"}]
    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["incremental_output"] is True
    assert parameters["stream"] is True
    assert result["content"] == "你好"
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_dashscope_generation_places_sdk_specific_params() -> None:
    captured_headers: list[dict[str, str | None]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append({"plugin": request.headers.get("X-DashScope-Plugin")})
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(
        _response_request(
            extra_params={
                "customized_model_id": "cm_1",
                "plugins": {"name": "search"},
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "enable_thinking": False,
                "n": 2,
            }
        )
    )

    assert result["content"] == "ok"
    assert captured_headers == [{"plugin": '{"name":"search"}'}]
    body = captured_body[0]
    assert body["input"] == {
        "messages": [{"role": "user", "content": "你好"}],
        "customized_model_id": "cm_1",
    }
    parameters = _as_mapping(body["parameters"])
    assert parameters["repetition_penalty"] == 1.1
    assert parameters["presence_penalty"] == 0.2
    assert parameters["enable_thinking"] is False
    assert parameters["n"] == 2
    assert "customized_model_id" not in parameters
    assert "plugins" not in parameters


@pytest.mark.asyncio
async def test_dashscope_generation_translates_host_response_format_json_object() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "{}"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(_response_request(response_format={"format_type": "json_object"}))

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["result_format"] == "message"
    assert parameters["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_dashscope_generation_rejects_unconfirmed_host_response_format_json_schema() -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )

    with pytest.raises(ValueError, match="json_schema"):
        await provider.get_response(
            _response_request(
                response_format={
                    "format_type": "json_schema",
                    "schema": {
                        "name": "answer",
                        "description": "Answer payload",
                        "schema": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                        "strict": True,
                    },
                }
            )
        )


@pytest.mark.asyncio
async def test_dashscope_generation_rejects_response_format_conflict() -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )

    with pytest.raises(ValueError, match="response_format"):
        await provider.get_response(
            _response_request(
                extra_params={"response_format": {"type": "json_object"}},
                response_format={"format_type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_dashscope_generation_keeps_result_format_separate_from_response_format() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "{}"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            extra_params={"result_format": "text"},
            response_format={"format_type": "json_object"},
        )
    )

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["result_format"] == "text"
    assert parameters["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_dashscope_generation_request_sampling_overrides_model_sampling() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            model_temperature=0.8,
            model_max_tokens=256,
            request_temperature=0.1,
            request_max_tokens=32,
        )
    )

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["temperature"] == 0.1
    assert parameters["max_tokens"] == 32


@pytest.mark.asyncio
async def test_dashscope_generation_falls_back_to_model_sampling_when_request_fields_absent() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            model_temperature=0.8,
            model_max_tokens=256,
            request_temperature=None,
            request_max_tokens=None,
        )
    )

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["temperature"] == 0.8
    assert parameters["max_tokens"] == 256


@pytest.mark.asyncio
async def test_dashscope_generation_policy_disabled_paths_and_override_parameters() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    policies = ParameterPolicyRegistry(
        dashscope=ProviderCapabilityPolicies(
            chat_completion=ParameterPolicy(
                disabled_paths=("unknown_field",),
                override_params={"body": {"parameters": {"enable_thinking": False}}},
            )
        )
    )
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(parameter_policies=policies),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(extra_params={"unknown_field": 1, "enable_thinking": True}))

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["enable_thinking"] is False
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_stream_event_error_and_status_raise() -> None:
    async def event_error_handler(request: httpx.Request) -> httpx.Response:
        del request
        content = 'event: error\ndata: {"message":"stream bad"}\n\n'
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    event_error_provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(event_error_handler),
    )
    with pytest.raises(Exception, match="stream bad"):
        await event_error_provider.get_response(_response_request(stream=True))

    async def status_error_handler(request: httpx.Request) -> httpx.Response:
        del request
        content = 'status: 500\ndata: {"message":"status bad"}\n\n'
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    status_error_provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(status_error_handler),
    )
    with pytest.raises(Exception, match="500"):
        await status_error_provider.get_response(_response_request(stream=True))


@pytest.mark.asyncio
async def test_dashscope_stream_handles_cumulative_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = (
            "event: result\n"
            'data: {"output":{"choices":[{"message":{"content":"你"}}]},"usage":{}}\n\n'
            "event: result\n"
            'data: {"output":{"choices":[{"message":{"content":"你好"}}]},"usage":{}}\n\n'
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(stream=True, extra_params={"incremental_output": False}))

    assert result["content"] == "你好"


@pytest.mark.asyncio
async def test_dashscope_xml_tool_fallback_when_no_native_tool_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {"message": {"content": '<tool_call><function=lookup>{"q":"x"}</function></tool_call>'}}
                    ]
                },
                "usage": {},
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request())

    tool_calls = _as_list(result["tool_calls"])
    first_tool_call = _as_mapping(tool_calls[0])
    assert first_tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}


@pytest.mark.asyncio
async def test_dashscope_error_payload_raises() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"code": "InvalidParameter", "message": "bad", "request_id": "req_1"},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    with pytest.raises(Exception, match="InvalidParameter"):
        await provider.get_response(_response_request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_endpoint", "expected_input"),
    [
        (
            "text-embedding-v4",
            DASHSCOPE_TEXT_EMBEDDING_ENDPOINT,
            {"texts": ["文本向量"]},
        ),
        (
            "qwen3.7-text-embedding",
            DASHSCOPE_TEXT_EMBEDDING_ENDPOINT,
            {"texts": ["文本向量"]},
        ),
        (
            "qwen2-vl-embedding",
            DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
            {"contents": [{"text": "文本向量"}]},
        ),
        (
            "multimodal-embedding-v1",
            DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
            {"contents": [{"text": "文本向量"}]},
        ),
        (
            "multimodal-embedding-one-peace-v1",
            DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
            {"contents": [{"text": "文本向量"}]},
        ),
        (
            "tongyi-embedding-vision-plus",
            DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
            {"contents": [{"text": "文本向量"}]},
        ),
    ],
)
async def test_dashscope_embedding_routes_model_families(
    model: str,
    expected_endpoint: str,
    expected_input: dict,
) -> None:
    captured_path: list[str] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "output": {"embeddings": [{"embedding": [1, "2.5"]}]},
                "usage": {"total_tokens": 2},
            },
        )

    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(_embedding_request(model))

    assert captured_path == ["/api/v1/" + expected_endpoint]
    assert captured_body[0]["input"] == expected_input
    assert result["embedding"] == [1.0, 2.5]
    assert result["raw_data"] == {
        "output": {"embeddings": [{"embedding": [1, "2.5"]}]},
        "usage": {"total_tokens": "***"},
    }


@pytest.mark.asyncio
async def test_dashscope_embedding_rejects_tongyi_vison_typo() -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )

    with pytest.raises(ValueError) as exc_info:
        await provider.get_embedding(_embedding_request("tongyi-embedding-vison-plus"))

    assert "tongyi-embedding-vision-*" in str(exc_info.value)
    assert "vison" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_dashscope_embedding_accepts_native_dimension() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"output": {"embeddings": [{"embedding": [1]}]}, "usage": {}})

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_embedding(_embedding_request("text-embedding-v4", extra_params={"dimension": 256}))

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["dimension"] == 256
    assert "dimensions" not in parameters


@pytest.mark.asyncio
async def test_dashscope_embedding_translates_dimensions_to_body_parameters_dimension() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"output": {"embeddings": [{"embedding": [1]}]}, "usage": {}})

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    # extra_params.dimensions is mapped to body.parameters.dimension (DashScope native)
    await provider.get_embedding(_embedding_request("text-embedding-v4", extra_params={"dimensions": 256}))

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["dimension"] == 256
    assert "dimensions" not in parameters


@pytest.mark.asyncio
async def test_dashscope_embedding_accepts_sdk_params() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"output": {"embeddings": [{"embedding": [1]}]}, "usage": {}})

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_embedding(
        _embedding_request(
            "qwen2-vl-embedding",
            extra_params={
                "output_type": "dense",
                "instruct": "query",
                "fps": 0.5,
                "res_level": 2,
                "max_video_frames": 8,
                "auto_truncation": True,
            },
        )
    )

    body = captured_body[0]
    assert body["input"] == {"contents": [{"text": "文本向量"}]}
    parameters = _as_mapping(body["parameters"])
    assert parameters["output_type"] == "dense"
    assert parameters["instruct"] == "query"
    assert parameters["fps"] == 0.5
    assert parameters["res_level"] == 2
    assert parameters["max_video_frames"] == 8
    assert parameters["auto_truncation"] is True


@pytest.mark.asyncio
async def test_dashscope_embedding_qwen3_defaults_enable_fusion_unless_overridden() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"output": {"embeddings": [{"embedding": [1]}]}, "usage": {}})

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_embedding(_embedding_request("qwen3-vl-embedding"))
    await provider.get_embedding(_embedding_request("qwen3-vl-embedding", extra_params={"enable_fusion": False}))

    first_parameters = _as_mapping(captured_body[0]["parameters"])
    second_parameters = _as_mapping(captured_body[1]["parameters"])
    assert first_parameters["enable_fusion"] is True
    assert second_parameters["enable_fusion"] is False


@pytest.mark.asyncio
async def test_dashscope_embedding_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "output": {"embeddings": [{"embedding": [1, "2.5"]}]},
                "usage": {"total_tokens": 2},
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_embedding(
        _embedding_request("text-embedding-v4", base_url="https://relay.example/custom")
    )

    assert captured_url == [f"https://dashscope.aliyuncs.com/api/v1/{DASHSCOPE_TEXT_EMBEDDING_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]


@pytest.mark.asyncio
async def test_dashscope_embedding_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "output": {"embeddings": [{"embedding": [1, "2.5"]}]},
                "usage": {"total_tokens": 2},
            },
        )

    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(dashscope_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(
        _embedding_request("text-embedding-v4", base_url="https://relay.example/custom")
    )

    assert captured_url == [f"https://relay.example/custom/api/v1/{DASHSCOPE_TEXT_EMBEDDING_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]


@pytest.mark.asyncio
async def test_dashscope_embedding_rejects_unsupported_model() -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )

    with pytest.raises(ValueError, match="text-embedding-v"):
        await provider.get_embedding(_embedding_request("unknown-embedding"))


# ── multimodal image content ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashscope_multimodal_image_content_builds_content_list() -> None:
    """用户消息含图片 part 时，content 应为 list[dict] 多模态格式"""
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "output": {"choices": [{"message": {"content": "描述完成"}}]},
                "usage": {},
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(_response_request_with_image())

    messages = captured_body[0]["input"]["messages"]
    user_msg = messages[0]
    assert isinstance(user_msg["content"], list)
    blocks = cast(list, user_msg["content"])

    text_blocks = [b for b in blocks if "text" in b]
    image_blocks = [b for b in blocks if "image" in b]
    assert len(text_blocks) >= 1
    assert len(image_blocks) == 1
    assert image_blocks[0]["image"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_dashscope_multimodal_image_empty_skips_invalid_image() -> None:
    """无效 base64 图片在 placeholder 策略下应生成占位文本"""
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "output": {"choices": [{"message": {"content": "..."}}]},
                "usage": {},
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(_response_request_with_image(extra_params={"image_format": None}))

    messages = captured_body[0]["input"]["messages"]
    user_content = messages[0]["content"]
    assert isinstance(user_content, list)
    # 图片无效，但 placeholder 策略会生成占位文本
    has_placeholder = any("[图片内容不可用]" in b.get("text", "") for b in cast(list, user_content))
    # 或者有图片 block（如果解码成功）
    has_image = any("image" in b for b in cast(list, user_content))
    assert has_placeholder or has_image


# ── audio transcription ───────────────────────────────────────────────


def test_dashscope_audio_transcription_build_request_format() -> None:
    """验证 build_audio_transcription_request 构建正确的 multimodal-generation 请求体"""
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    # 构造最小 WAV 文件的 base64（RIFF header + dummy data）
    wav_header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    request = AudioTranscriptionRequestSnapshot.model_validate(
        {
            "model_info": {"model_identifier": "qwen3-asr-flash", "extra_params": {}},
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(wav_header).decode(),
        }
    )
    body, headers, query = build_audio_transcription_request(request, options=ProviderRuntimeOptions())

    assert body["model"] == "qwen3-asr-flash"
    assert body["parameters"]["result_format"] == "message"
    assert headers == {}
    assert query == {}
    content = body["input"]["messages"][0]["content"]
    assert len(content) == 1
    audio_block = content[0]
    assert "audio" in audio_block
    assert audio_block["audio"].startswith("data:audio/wav;base64,")


def test_dashscope_audio_transcription_parse_response() -> None:
    """从 multimodal-generation 响应中提取转录文本"""
    from src.providers.dashscope_provider.audio_transcriptions import (
        parse_audio_transcription_response,
    )

    payload = {"output": {"choices": [{"message": {"content": "你好世界"}}]}}
    text, raw_data = parse_audio_transcription_response(payload, options=ProviderRuntimeOptions())
    assert text == "你好世界"
    assert raw_data is None


def test_dashscope_audio_transcription_parse_missing_content_raises() -> None:
    """缺少 choices[0].message.content 时应报错"""
    from src.providers.dashscope_provider.audio_transcriptions import (
        parse_audio_transcription_response,
    )

    with pytest.raises(ValueError, match="DashScope"):
        parse_audio_transcription_response({"output": {}}, options=ProviderRuntimeOptions())


def test_dashscope_audio_transcription_parse_include_raw_data() -> None:
    """include_raw_data=True 时应返回 raw_data"""
    from src.providers.dashscope_provider.audio_transcriptions import (
        parse_audio_transcription_response,
    )

    payload = {"output": {"choices": [{"message": {"content": "test"}}]}, "usage": {}}
    text, raw_data = parse_audio_transcription_response(payload, options=ProviderRuntimeOptions(include_raw_data=True))
    assert text == "test"
    assert raw_data is not None
    assert raw_data["output"]["choices"][0]["message"]["content"] == "test"


def test_dashscope_audio_transcription_builder_pipeline_injects_extra_params() -> None:
    """extra_params 通过 pipeline 注入 body（asr_options 级别）"""
    from src.config import MaiDockConfig, build_parameter_policies
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    wav_header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    request = AudioTranscriptionRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "qwen3-asr-flash",
                "extra_params": {"language": "en", "enable_itn": False},
            },
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(wav_header).decode(),
        }
    )
    config = MaiDockConfig()
    policies = build_parameter_policies(config)
    options = ProviderRuntimeOptions(parameter_policies=policies)
    body, _headers, _query = build_audio_transcription_request(request, options=options)

    assert body["model"] == "qwen3-asr-flash"
    assert body["parameters"]["result_format"] == "message"
    assert body["parameters"]["asr_options"]["language"] == "en"
    assert body["parameters"]["asr_options"]["enable_itn"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "visual", "expected_endpoint"),
    [
        ("qwen3.7-max", False, DASHSCOPE_GENERATION_ENDPOINT),
        ("qwen3.7-max-2026-05-20", False, DASHSCOPE_GENERATION_ENDPOINT),
        ("qwen3.6-max-preview", False, DASHSCOPE_GENERATION_ENDPOINT),
        ("qwen3.7-max-2026-06-08", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen3.7-plus-latest", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen3.6-plus", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen3.5-flash", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen-vl-plus", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qvq-max", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen-audio-turbo", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen-omni-turbo", False, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
        ("qwen3.7-unconfirmed", False, DASHSCOPE_GENERATION_ENDPOINT),
        ("unknown-visual", True, DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT),
    ],
)
async def test_dashscope_endpoint_model_matrix(
    model: str,
    visual: bool,
    expected_endpoint: str,
) -> None:
    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    request = _response_request()
    request["model_info"]["model_identifier"] = model
    request["model_info"]["visual"] = visual
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(request)

    assert captured_paths == ["/api/v1/" + expected_endpoint]
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_text_only_model_rejects_actual_image_locally() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        del request
        called = True
        return httpx.Response(200)

    request = _response_request_with_image()
    request["model_info"]["model_identifier"] = "qwen3.7-max"
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    with pytest.raises(ValueError, match="qwen3.7-max"):
        await provider.get_response(request)

    assert called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "first_endpoint", "second_endpoint"),
    [
        (
            "unknown-model",
            DASHSCOPE_GENERATION_ENDPOINT,
            DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
        ),
        (
            "qwen3.7-plus",
            DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT,
            DASHSCOPE_GENERATION_ENDPOINT,
        ),
    ],
)
async def test_dashscope_auto_detects_endpoint_in_both_directions(
    model: str,
    first_endpoint: str,
    second_endpoint: str,
) -> None:
    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        if len(captured_paths) == 1:
            return httpx.Response(
                400,
                json={
                    "code": "InvalidParameter",
                    "message": "url error: wrong endpoint",
                    "request_id": "req_probe",
                },
            )
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    request = _response_request()
    request["model_info"]["model_identifier"] = model
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(request)

    assert captured_paths == ["/api/v1/" + first_endpoint, "/api/v1/" + second_endpoint]
    assert result["content"] == "ok"


@pytest.mark.asyncio
async def test_dashscope_endpoint_cache_keeps_kind_across_base_urls() -> None:
    captured_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        if len(captured_urls) == 1:
            return httpx.Response(
                200,
                json={
                    "code": "InvalidParameter",
                    "message": "url error",
                    "request_id": "req_1",
                },
            )
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    options = ProviderRuntimeOptions(dashscope_force_official_endpoint=False)
    provider = DashScopeProvider(options=options, transport=httpx.MockTransport(handler))
    first = _response_request(base_url="https://workspace-a.example/custom")
    first["model_info"]["model_identifier"] = "probe-model"
    second = _response_request(base_url="https://workspace-b.example/other")
    second["model_info"]["model_identifier"] = "probe-model"

    await provider.get_response(first)
    await provider.get_response(second)

    assert captured_urls == [
        f"https://workspace-a.example/custom/api/v1/{DASHSCOPE_GENERATION_ENDPOINT}",
        f"https://workspace-a.example/custom/api/v1/{DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT}",
        f"https://workspace-b.example/other/api/v1/{DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT}",
    ]


@pytest.mark.asyncio
async def test_dashscope_image_request_never_falls_back_to_text_endpoint() -> None:
    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "url error",
                "request_id": "req_image",
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    with pytest.raises(Exception, match="req_image"):
        await provider.get_response(_response_request_with_image())

    assert captured_paths == ["/api/v1/" + DASHSCOPE_MULTIMODAL_GENERATION_ENDPOINT]


@pytest.mark.asyncio
@pytest.mark.parametrize("http_status", [200, 400])
async def test_dashscope_json_errors_preserve_structured_fields(
    http_status: int,
) -> None:
    from src.providers.dashscope_provider.errors import DashScopeApiError

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            http_status,
            json={
                "code": "BadRequest",
                "message": "structured bad",
                "request_id": "req_structured",
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    with pytest.raises(DashScopeApiError) as captured:
        await provider.get_response(_response_request())

    assert captured.value.code == "BadRequest"
    assert captured.value.upstream_message == "structured bad"
    assert captured.value.request_id == "req_structured"
    assert captured.value.status_code == http_status


@pytest.mark.asyncio
async def test_dashscope_sse_error_preserves_structured_fields() -> None:
    from src.providers.dashscope_provider.errors import DashScopeApiError

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = (
            "event: error\n"
            "status: 422\n"
            'data: {"code":"InvalidParameter","message":"stream bad","request_id":"req_stream"}\n\n'
        )
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    with pytest.raises(DashScopeApiError) as captured:
        await provider.get_response(_response_request(stream=True))

    assert captured.value.code == "InvalidParameter"
    assert captured.value.upstream_message == "stream bad"
    assert captured.value.request_id == "req_stream"
    assert captured.value.status_code == 422


def _stream_payload(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tools: list | None = None,
) -> dict:
    message: dict = {}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tools is not None:
        message["tool_calls"] = tools
    return {"output": {"choices": [{"message": message}]}, "usage": {}}


def _tool_chunk(
    *,
    index: int | None = None,
    call_id: str | None = None,
    name: str = "",
    arguments: str = "",
) -> dict:
    result: dict = {"function": {"name": name, "arguments": arguments}}
    if index is not None:
        result["index"] = index
    if call_id is not None:
        result["id"] = call_id
    return result


def test_dashscope_stream_accepts_official_no_index_tool_chunks() -> None:
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(call_id="call_1", name="lookup", arguments="{")]))
    accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(call_id="", name="", arguments='"q":"x"}')]))

    result = accumulator.to_provider_response().to_host_dict()
    tool_call = _as_mapping(_as_list(result["tool_calls"])[0])
    assert tool_call["id"] == "call_1"
    assert tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}


def test_dashscope_stream_merges_parallel_ordinal_chunks_and_split_ids() -> None:
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(call_id="call_", name="look", arguments="{"),
                _tool_chunk(call_id="call_", name="weather", arguments="{"),
            ]
        )
    )
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(call_id="1", name="up", arguments='"q":"x"}'),
                _tool_chunk(call_id="2", name="", arguments='"city":"hz"}'),
            ]
        )
    )

    result = accumulator.to_provider_response().to_host_dict()
    tool_calls = _as_list(result["tool_calls"])
    assert _as_mapping(tool_calls[0])["id"] == "call_1"
    assert _as_mapping(tool_calls[0])["function"] == {
        "name": "lookup",
        "arguments": {"q": "x"},
    }
    assert _as_mapping(tool_calls[1])["id"] == "call_2"
    assert _as_mapping(tool_calls[1])["function"] == {
        "name": "weather",
        "arguments": {"city": "hz"},
    }


def test_dashscope_stream_binds_late_index_and_replays_merged_slots() -> None:
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(arguments='{"q":')]))
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(arguments=""),
                _tool_chunk(call_id="call_late", name="lookup", arguments=""),
            ]
        )
    )
    accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(index=0, call_id="call_late", arguments='"x"}')]))

    result = accumulator.to_provider_response().to_host_dict()
    tool_calls = _as_list(result["tool_calls"])
    assert len(tool_calls) == 1
    assert _as_mapping(tool_calls[0])["id"] == "call_late"
    assert _as_mapping(tool_calls[0])["function"] == {
        "name": "lookup",
        "arguments": {"q": "x"},
    }


def test_dashscope_stream_rejects_only_real_identity_conflict() -> None:
    from src.providers.common.httpx import HttpxProviderParseError
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(index=0, call_id="call_a", name="a", arguments="{}"),
                _tool_chunk(index=1, call_id="call_b", name="b", arguments="{}"),
            ]
        )
    )

    with pytest.raises(HttpxProviderParseError, match="index/id"):
        accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(index=0, call_id="call_b", arguments="")]))


def test_dashscope_stream_orders_explicit_indexes_before_provisional_slots() -> None:
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(index=2, call_id="call_two", name="two", arguments="{}"),
                _tool_chunk(call_id="call_provisional", name="provisional", arguments="{}"),
                _tool_chunk(index=0, call_id="call_zero", name="zero", arguments="{}"),
            ]
        )
    )

    result = accumulator.to_provider_response().to_host_dict()
    assert [_as_mapping(item)["id"] for item in _as_list(result["tool_calls"])] == [
        "call_zero",
        "call_two",
        "call_provisional",
    ]


def test_dashscope_stream_rejects_unidentified_fragment_with_multiple_pending_slots() -> None:
    from src.providers.common.httpx import HttpxProviderParseError
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    accumulator.merge_payload(
        _stream_payload(
            tools=[
                _tool_chunk(name="a", arguments="{"),
                _tool_chunk(name="b", arguments="{"),
                _tool_chunk(name="c", arguments="{"),
            ]
        )
    )
    accumulator.merge_payload(_stream_payload(tools=[_tool_chunk(index=0, arguments="")]))

    with pytest.raises(HttpxProviderParseError, match="多个可能归属"):
        accumulator.merge_payload(
            _stream_payload(
                tools=[
                    _tool_chunk(index=0, arguments=""),
                    _tool_chunk(arguments="x"),
                ]
            )
        )


@pytest.mark.parametrize(
    "reasoning_chunks",
    [
        ["思", "考"],
        ["思", "思考"],
    ],
)
def test_dashscope_stream_merges_reasoning_without_extra_newline(
    reasoning_chunks: list[str],
) -> None:
    from src.providers.dashscope_provider.streaming import DashScopeStreamAccumulator

    accumulator = DashScopeStreamAccumulator(options=ProviderRuntimeOptions())
    for reasoning in reasoning_chunks:
        accumulator.merge_payload(_stream_payload(reasoning=reasoning))
    accumulator.merge_payload(_stream_payload(content="answer"))

    result = accumulator.to_provider_response().to_host_dict()
    assert result["reasoning_content"] == "思考"
    assert result["content"] == "answer"


@pytest.mark.asyncio
async def test_dashscope_new_native_parameters_are_validated_and_forwarded() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            content=('event: result\ndata: {"output":{"choices":[{"message":{"content":"ok"}}]},"usage":{}}\n\n'),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(
        _response_request(
            stream=True,
            extra_params={
                "max_completion_tokens": 128,
                "thinking_budget": 64,
                "reasoning_effort": "xhigh",
                "tool_stream": True,
                "enable_code_interpreter": False,
                "search_options": {"forced_search": True},
                "vl_high_resolution_images": True,
            },
        )
    )

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert "max_tokens" not in parameters
    assert parameters["max_completion_tokens"] == 128
    assert parameters["thinking_budget"] == 64
    assert parameters["reasoning_effort"] == "xhigh"
    assert parameters["tool_stream"] is True
    assert parameters["enable_code_interpreter"] is False
    assert parameters["search_options"] == {"forced_search": True}
    assert parameters["vl_high_resolution_images"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_field"),
    [
        ("qwen3.7-max", "max_completion_tokens"),
        ("qwen3.6-plus", "max_completion_tokens"),
        ("qwen3.5-flash", "max_completion_tokens"),
        ("kimi-k2.5", "max_completion_tokens"),
        ("glm-5", "max_completion_tokens"),
        ("MiniMax-M2.5", "max_completion_tokens"),
        ("deepseek-v3.2", "max_completion_tokens"),
        ("deepseek-r1-0528", "max_completion_tokens"),
        ("qwen3.6-max-preview", "max_tokens"),
        ("qwen3.5-omni", "max_tokens"),
        ("qwen-plus", "max_tokens"),
        ("moonshot/kimi-k2.5", "max_tokens"),
        ("custom-model", "max_tokens"),
    ],
)
async def test_dashscope_host_max_tokens_uses_official_model_capability(
    model: str,
    expected_field: str,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    request = _response_request()
    request["model_info"]["model_identifier"] = model
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(request)

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters[expected_field] == 64
    unexpected_field = "max_tokens" if expected_field == "max_completion_tokens" else "max_completion_tokens"
    assert unexpected_field not in parameters


@pytest.mark.asyncio
async def test_dashscope_explicit_legacy_max_tokens_disables_automatic_translation() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    request = _response_request(extra_params={"max_tokens": 96}, request_max_tokens=None)
    request["model_info"]["model_identifier"] = "qwen3.7-max"
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(request)

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["max_tokens"] == 96
    assert "max_completion_tokens" not in parameters


@pytest.mark.asyncio
async def test_dashscope_rejects_two_explicit_native_token_limits() -> None:
    provider = DashScopeProvider(options=ProviderRuntimeOptions())
    request = _response_request(
        extra_params={"max_tokens": 96, "max_completion_tokens": 128},
        request_max_tokens=None,
    )

    with pytest.raises(ValueError, match="max_tokens.*max_completion_tokens"):
        await provider.get_response(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "model", "expected_field", "expected_value"),
    [
        (
            ParameterPolicy(disabled_paths=("body.parameters.max_completion_tokens",)),
            "qwen3.7-max",
            "max_tokens",
            64,
        ),
        (
            ParameterPolicy(rejected_paths=("body.parameters.max_completion_tokens",)),
            "qwen3.7-max",
            "max_tokens",
            64,
        ),
        (
            ParameterPolicy(override_params={"body": {"parameters": {"max_completion_tokens": 192}}}),
            "qwen-plus",
            "max_completion_tokens",
            192,
        ),
    ],
)
async def test_dashscope_automatic_token_target_respects_parameter_policy(
    policy: ParameterPolicy,
    model: str,
    expected_field: str,
    expected_value: int,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    policies = ParameterPolicyRegistry(dashscope=ProviderCapabilityPolicies(chat_completion=policy))
    request = _response_request()
    request["model_info"]["model_identifier"] = model
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(parameter_policies=policies),
        transport=httpx.MockTransport(handler),
    )
    await provider.get_response(request)

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters[expected_field] == expected_value
    unexpected_field = "max_tokens" if expected_field == "max_completion_tokens" else "max_completion_tokens"
    assert unexpected_field not in parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_completion_tokens", 0),
        ("thinking_budget", -1),
        ("reasoning_effort", "minimal"),
        ("tool_stream", "true"),
        ("enable_code_interpreter", 1),
        ("search_options", []),
        ("vl_high_resolution_images", "false"),
    ],
)
async def test_dashscope_new_native_parameters_reject_invalid_values(field: str, value: object) -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )

    with pytest.raises((TypeError, ValueError), match=field):
        await provider.get_response(_response_request(extra_params={field: value}))


@pytest.mark.asyncio
async def test_dashscope_tool_choice_required_uses_final_single_tool() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    final_tool = {
        "type": "function",
        "function": {"name": "final_lookup", "parameters": {"type": "object"}},
    }
    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(_response_request(extra_params={"tools": [final_tool], "tool_choice": "required"}))

    parameters = _as_mapping(captured_body[0]["parameters"])
    assert parameters["tools"] == [final_tool]
    assert parameters["tool_choice"] == {
        "type": "function",
        "function": {"name": "final_lookup"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_count", [0, 2])
async def test_dashscope_tool_choice_required_rejects_non_single_final_tools(
    tool_count: int,
) -> None:
    request = _response_request(extra_params={"tool_choice": "required"})
    if tool_count == 0:
        request["tool_options"] = []
    else:
        request["tool_options"].append(
            {
                "type": "function",
                "function": {
                    "name": "second",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )

    with pytest.raises(ValueError, match="required"):
        await provider.get_response(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "lookup"}},
    ],
)
async def test_dashscope_thinking_mode_rejects_forced_tool_choice(
    tool_choice: object,
) -> None:
    provider = DashScopeProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )

    with pytest.raises(ValueError, match="auto/none"):
        await provider.get_response(
            _response_request(extra_params={"enable_thinking": True, "tool_choice": tool_choice})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_choice", ["auto", "none"])
async def test_dashscope_thinking_mode_accepts_unforced_tool_choice(
    tool_choice: str,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": "ok"}}]}, "usage": {}},
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_response(_response_request(extra_params={"enable_thinking": True, "tool_choice": tool_choice}))

    assert captured_body[0]["parameters"]["tool_choice"] == tool_choice


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["embedding", "audio_transcription"])
async def test_dashscope_auxiliary_capabilities_use_structured_errors(
    capability: str,
) -> None:
    from src.providers.dashscope_provider.errors import DashScopeApiError

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            400,
            json={
                "code": "BadRequest",
                "message": "auxiliary bad",
                "request_id": "req_aux",
            },
        )

    provider = DashScopeProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    with pytest.raises(DashScopeApiError) as captured:
        if capability == "embedding":
            await provider.get_embedding(_embedding_request("qwen3.7-text-embedding"))
        else:
            await provider.get_audio_transcriptions(
                {
                    "model_info": {
                        "model_identifier": "qwen3-asr-flash",
                        "extra_params": {},
                    },
                    "api_provider": _api_provider(),
                    "audio_base64": base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode(),
                }
            )

    assert captured.value.code == "BadRequest"
    assert captured.value.request_id == "req_aux"
    assert captured.value.status_code == 400


def _audio_request(audio: bytes, *, extra_params: dict | None = None) -> AudioTranscriptionRequestSnapshot:
    return AudioTranscriptionRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "qwen3-asr-flash",
                "extra_params": extra_params or {},
            },
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(audio).decode(),
        }
    )


@pytest.mark.parametrize(
    ("signature", "mime"),
    [
        (b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
        (b"ID3payload", "audio/mpeg"),
        (b"\xff\xf1payload", "audio/aac"),
        (b"fLaCpayload", "audio/flac"),
        (b"OggSpayload", "audio/ogg"),
    ],
)
def test_dashscope_asr_accepts_documented_audio_formats(signature: bytes, mime: str) -> None:
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    body, _headers, _query = build_audio_transcription_request(
        _audio_request(signature),
        options=ProviderRuntimeOptions(),
    )

    audio_url = body["input"]["messages"][0]["content"][0]["audio"]
    assert audio_url.startswith(f"data:{mime};base64,")


@pytest.mark.parametrize(
    ("audio_base64", "extra_params", "error_match"),
    [
        ("not-base64!", {}, "Base64"),
        (base64.b64encode(b"unknown").decode(), {}, "格式|format"),
        (
            base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode(),
            {"format": "mp3"},
            "不一致",
        ),
        (base64.b64encode(b"\x00\x00\x00\x00ftypM4A ").decode(), {}, "m4a"),
    ],
)
def test_dashscope_asr_rejects_invalid_audio(
    audio_base64: str,
    extra_params: dict,
    error_match: str,
) -> None:
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    request = AudioTranscriptionRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "qwen3-asr-flash",
                "extra_params": extra_params,
            },
            "api_provider": _api_provider(),
            "audio_base64": audio_base64,
        }
    )

    with pytest.raises((TypeError, ValueError), match=error_match):
        build_audio_transcription_request(request, options=ProviderRuntimeOptions())


def test_dashscope_asr_rejects_base64_larger_than_ten_mib() -> None:
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    request = AudioTranscriptionRequestSnapshot.model_validate(
        {
            "model_info": {"model_identifier": "qwen3-asr-flash", "extra_params": {}},
            "api_provider": _api_provider(),
            "audio_base64": "A" * (10 * 1024 * 1024 + 4),
        }
    )

    with pytest.raises(ValueError, match="10 MiB"):
        build_audio_transcription_request(request, options=ProviderRuntimeOptions())


def test_dashscope_asr_consumes_format_hints_locally() -> None:
    from src.providers.dashscope_provider.audio_transcriptions import (
        build_audio_transcription_request,
    )

    body, _headers, _query = build_audio_transcription_request(
        _audio_request(
            b"RIFF\x00\x00\x00\x00WAVE",
            extra_params={"format": "wav", "audio_format": "wave"},
        ),
        options=ProviderRuntimeOptions(),
    )

    assert "format" not in body
    assert "audio_format" not in body
    assert "format" not in body["parameters"]
    assert "audio_format" not in body["parameters"]
