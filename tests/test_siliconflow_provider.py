import base64
import json
from dataclasses import replace
from typing import cast

import httpx
import pytest

from src.config import MaiDockConfig, build_runtime_options
from src.core.common import ProviderRuntimeOptions
from src.host_adapters.siliconflow_provider.chat import (
    SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT,
)
from src.host_adapters.siliconflow_provider.embeddings import (
    SILICONFLOW_EMBEDDINGS_ENDPOINT,
)
from tests.support.host_adapters import SiliconFlowProvider


def _overrides_options(
    provider: str,
    capability: str,
    overrides: dict,
    *,
    include_raw_data: bool = False,
) -> ProviderRuntimeOptions:
    """构造带指定覆写目录的运行时选项。"""

    config = MaiDockConfig.model_validate({provider: {capability: {"overrides": overrides}}})
    options = build_runtime_options(config)
    if include_raw_data:
        return replace(options, include_raw_data=True)
    return options


def _api_provider(*, base_url: str | None = "https://api.siliconflow.cn/v1") -> dict:
    return {
        "api_key": "siliconflow-key",
        "auth_type": "bearer",
        "base_url": base_url,
        "default_headers": {},
        "default_query": {},
    }


def _response_request(
    *,
    stream: bool = False,
    base_url: str | None = "https://api.siliconflow.cn/v1",
    extra_params: dict | None = None,
    response_format: dict | None = None,
    with_image: bool = False,
) -> dict:
    user_parts: list[dict] = [{"type": "text", "text": "你好"}]
    if with_image:
        user_parts.append(
            {
                "type": "image",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lm8cAAAAASUVORK5CYII=",
                "image_format": "png",
            }
        )
    request: dict = {
        "model_info": {
            "model_identifier": "Pro/zai-org/GLM-4.7",
            "force_stream_mode": stream,
            "extra_params": extra_params or {},
        },
        "api_provider": _api_provider(base_url=base_url),
        "message_list": [{"role": "user", "parts": user_parts}],
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
        "temperature": 0.2,
        "max_tokens": 64,
    }
    if response_format is not None:
        request["response_format"] = response_format
    return request


def _embedding_request(
    model: str,
    *,
    extra_params: dict | None = None,
    base_url: str | None = "https://api.siliconflow.cn/v1",
) -> dict:
    return {
        "model_info": {"model_identifier": model, "extra_params": extra_params or {}},
        "api_provider": _api_provider(base_url=base_url),
        "embedding_input": "文本向量",
    }


def _as_mapping(value: object) -> dict:
    assert isinstance(value, dict)
    return cast(dict, value)


def _as_list(value: object) -> list:
    assert isinstance(value, list)
    return cast(list, value)


@pytest.mark.asyncio
async def test_siliconflow_non_stream_response_posts_chat_completions_body_and_parses_reasoning_tools() -> None:
    captured_path: list[str] = []
    captured_headers: list[dict[str, str]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        captured_headers.append(
            {
                "Authorization": request.headers["Authorization"],
                "User-Agent": request.headers["User-Agent"],
            }
        )
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "model": "Pro/zai-org/GLM-4.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "回答",
                            "reasoning_content": "先想",
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
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            },
        )

    provider = SiliconFlowProvider(
        options=ProviderRuntimeOptions(siliconflow_user_agent="SiliconFlow-UA/1", include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(
        _response_request(
            extra_params={"top_p": 0.9},
            response_format={
                "format_type": "json_schema",
                "schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
            with_image=True,
        )
    )

    assert captured_path == ["/v1/" + SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT]
    assert captured_headers == [{"Authorization": "Bearer siliconflow-key", "User-Agent": "SiliconFlow-UA/1"}]
    body = captured_body[0]
    assert body["model"] == "Pro/zai-org/GLM-4.7"
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    # extra_params 完全无效：top_p 不会进入请求体。
    assert "top_p" not in body
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
    }
    messages = _as_list(body["messages"])
    first_message = _as_mapping(messages[0])
    content = _as_list(first_message["content"])
    assert content[0] == {"type": "text", "text": "你好"}
    image_part = _as_mapping(content[1])
    assert image_part["type"] == "image_url"
    image_url = _as_mapping(image_part["image_url"])
    assert cast(str, image_url["url"]).startswith("data:image/png;base64,")
    assert image_url["detail"] == "auto"
    tools = _as_list(body["tools"])
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look up data",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    assert result["content"] == "回答"
    assert result["reasoning_content"] == "先想"
    tool_calls = _as_list(result["tool_calls"])
    first_tool_call = _as_mapping(tool_calls[0])
    assert first_tool_call["id"] == "call_1"
    assert first_tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}
    assert first_tool_call["extra_content"] == {
        "provider": "siliconflow",
        "siliconflow": {"raw_arguments": '{"q":"x"}'},
        "tool_call_source": "reasoning",
    }
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 7
    raw_data = _as_mapping(result["raw_data"])
    assert raw_data["id"] == "chatcmpl_1"


@pytest.mark.asyncio
async def test_siliconflow_tools_override_appends_after_host_function_tools() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}},
        )

    provider = SiliconFlowProvider(
        options=_overrides_options(
            "siliconflow",
            "chat_completion",
            {"tools": '[{"type":"vendor_native","name":"native"}]'},
        ),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request())

    tools = _as_list(captured_body[0]["tools"])
    assert _as_mapping(tools[0])["function"]["name"] == "lookup"
    assert tools[1] == {"type": "vendor_native", "name": "native"}


@pytest.mark.asyncio
async def test_siliconflow_chat_ignores_stale_model_sampling_fields() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            },
        )

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    request = _response_request()
    model_info = _as_mapping(request["model_info"])
    model_info["temperature"] = 0.8
    model_info["max_tokens"] = 256
    request["temperature"] = 0.1
    request["max_tokens"] = 32

    await provider.get_response(request)

    body = captured_body[0]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 32


@pytest.mark.asyncio
async def test_siliconflow_stream_accumulates_delta_reasoning_and_tool_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        assert body["stream"] is True
        content = "".join(
            [
                'data: {"choices":[{"delta":{"content":"你","reasoning_content":"想"}}],"usage":{"prompt_tokens":1}}\n\n',
                'data: {"choices":[{"delta":{"content":"好","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":\\"x\\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    tool_calls = _as_list(result["tool_calls"])
    first_tool_call = _as_mapping(tool_calls[0])
    assert first_tool_call["id"] == "call_1"
    assert first_tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}
    assert _as_mapping(first_tool_call["extra_content"])["tool_call_source"] == "reasoning"
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_siliconflow_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [{"message": {"role": "assistant", "content": "回答"}}],
                "usage": {"total_tokens": 1},
            },
        )

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(_response_request(base_url="https://relay.example/custom"))

    assert captured_url == [f"https://api.siliconflow.cn/v1/{SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT}"]
    assert result["content"] == "回答"


@pytest.mark.asyncio
async def test_siliconflow_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [{"message": {"role": "assistant", "content": "回答"}}],
                "usage": {"total_tokens": 1},
            },
        )

    provider = SiliconFlowProvider(
        options=ProviderRuntimeOptions(siliconflow_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(base_url="https://relay.example/custom"))

    assert captured_url == [f"https://relay.example/custom/v1/{SILICONFLOW_CHAT_COMPLETIONS_ENDPOINT}"]
    assert result["content"] == "回答"


@pytest.mark.asyncio
async def test_siliconflow_embedding_posts_to_embeddings_and_qwen_dimensions_only() -> None:
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
                "data": [{"embedding": [1, "2.5"]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = SiliconFlowProvider(
        options=_overrides_options("siliconflow", "embeddings", {"dimensions": "1024"}, include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    qwen_result = await provider.get_embedding(_embedding_request("Qwen/Qwen3-Embedding-8B"))

    # Non-Qwen models should raise an explicit error when dimensions is requested
    with pytest.raises(ValueError, match="dimensions"):
        await provider.get_embedding(_embedding_request("BAAI/bge-m3"))

    assert captured_path == ["/v1/" + SILICONFLOW_EMBEDDINGS_ENDPOINT]
    assert captured_body[0] == {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": "文本向量",
        "encoding_format": "float",
        "dimensions": 1024,
    }
    assert qwen_result["embedding"] == [1.0, 2.5]
    qwen_usage = _as_mapping(qwen_result["usage"])
    qwen_raw_data = _as_mapping(qwen_result["raw_data"])
    assert qwen_usage["prompt_tokens"] == 2
    assert qwen_raw_data["data"] == [{"embedding": [1, "2.5"]}]


@pytest.mark.asyncio
async def test_siliconflow_embedding_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [1, "2.5"]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_embedding(
        _embedding_request("Qwen/Qwen3-Embedding-8B", base_url="https://relay.example/custom")
    )

    assert captured_url == [f"https://api.siliconflow.cn/v1/{SILICONFLOW_EMBEDDINGS_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]


@pytest.mark.asyncio
async def test_siliconflow_embedding_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [1, "2.5"]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = SiliconFlowProvider(
        options=ProviderRuntimeOptions(siliconflow_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(
        _embedding_request("Qwen/Qwen3-Embedding-8B", base_url="https://relay.example/custom")
    )

    assert captured_url == [f"https://relay.example/custom/v1/{SILICONFLOW_EMBEDDINGS_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]


@pytest.mark.asyncio
async def test_siliconflow_embedding_override_replaces_encoding_format() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"data": [{"embedding": [1, "2.5"]}], "usage": {}})

    provider = SiliconFlowProvider(
        options=_overrides_options("siliconflow", "embeddings", {"encoding_format": "base64"}),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(
        _embedding_request("Qwen/Qwen3-Embedding-8B", extra_params={"dimensions": 1024}),
    )

    # 覆写 encoding_format 覆盖默认 float；extra_params.dimensions 完全无效。
    assert captured_body[0]["encoding_format"] == "base64"
    assert "dimensions" not in captured_body[0]
    assert result["embedding"] == [1.0, 2.5]


# ── audio transcription ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_siliconflow_audio_transcription_posts_multipart_and_parses_text() -> None:
    """SF 音频转录 POST 音频并成功解析 text 字段"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "你好世界"})

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    result = await provider.get_audio_transcriptions(
        {
            "model_info": {"model_identifier": "whisper-1", "extra_params": {}},
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(b"fake-audio-data").decode(),
        }
    )
    assert result["content"] == "你好世界"


@pytest.mark.asyncio
async def test_siliconflow_audio_transcription_ignores_extra_params() -> None:
    """模型级 extra_params 不进入 multipart form。"""
    captured_request: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request.append(request)
        return httpx.Response(200, json={"text": "transcribed"})

    provider = SiliconFlowProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    result = await provider.get_audio_transcriptions(
        {
            "model_info": {
                "model_identifier": "whisper-1",
                "extra_params": {"language": "zh", "temperature": 0.2},
            },
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(b"fake-audio-data").decode(),
        }
    )
    assert result["content"] == "transcribed"
    assert b'name="language"' not in captured_request[0].content
    assert b'name="temperature"' not in captured_request[0].content


def test_siliconflow_audio_form_field_value_encodes_types() -> None:
    from src.host_adapters.siliconflow_provider.audio_transcriptions import (
        _form_field_value,
    )

    assert _form_field_value("hello") == "hello"
    assert _form_field_value(True) == "true"
    assert _form_field_value(False) == "false"
    assert _form_field_value(3) == "3"
    assert _form_field_value(0.2) == "0.2"
    assert _form_field_value(["a", "b"]) == '["a","b"]'
    assert _form_field_value({"key": 1}) == '{"key":1}'


@pytest.mark.asyncio
async def test_siliconflow_audio_transcription_builder_pipeline_injects_overrides() -> None:
    """builder 通过 pipeline 将覆写目录注入 form_data。"""
    captured_form: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_form.append({"content_type": request.headers.get("Content-Type", "")})
        return httpx.Response(200, json={"text": "ok"})

    options = _overrides_options(
        "siliconflow",
        "audio_transcription",
        {"language": "ja", "temperature": "0.3"},
    )
    provider = SiliconFlowProvider(options=options, transport=httpx.MockTransport(handler))
    result = await provider.get_audio_transcriptions(
        {
            "model_info": {"model_identifier": "whisper-1"},
            "api_provider": _api_provider(),
            "audio_base64": base64.b64encode(b"fake-audio-data").decode(),
        }
    )
    assert result["content"] == "ok"
    assert captured_form[0]["content_type"] != ""
