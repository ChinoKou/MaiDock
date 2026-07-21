import json
from typing import cast

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.providers.anthropic_messages_provider.provider import AnthropicMessagesProvider


def _api_provider(default_headers: dict | None = None) -> dict:
    return {
        "api_key": "test-key",
        "auth_type": "bearer",
        "base_url": "https://example.com/v1",
        "default_headers": default_headers or {},
    }


def _response_request(
    *,
    stream: bool = False,
    extra_params: dict | None = None,
    request_temperature: int | float | None = 0.2,
    request_max_tokens: int | None = 128,
    model_temperature: int | float | None = None,
    model_max_tokens: int | None = None,
) -> dict:
    model_info: dict = {
        "model_identifier": "claude-test",
        "force_stream_mode": stream,
        "extra_params": extra_params or {},
    }
    if model_temperature is not None:
        model_info["temperature"] = model_temperature
    if model_max_tokens is not None:
        model_info["max_tokens"] = model_max_tokens
    return {
        "model_info": model_info,
        "api_provider": _api_provider(),
        "message_list": [
            {"role": "system", "parts": [{"type": "text", "text": "你是助手"}]},
            {"role": "user", "parts": [{"type": "text", "text": "你好"}]},
        ],
        "tool_options": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "查天气",
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


@pytest.mark.asyncio
async def test_anthropic_non_stream_response_posts_messages_body_and_parses_reasoning_and_tools() -> None:
    captured_path: list[str] = []
    captured_headers: list[dict[str, str]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        captured_headers.append(
            {
                "User-Agent": request.headers["User-Agent"],
                "x-api-key": request.headers["x-api-key"],
                "anthropic-version": request.headers["anthropic-version"],
            }
        )
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "output_tokens": 4},
                "content": [
                    {"type": "thinking", "thinking": "先想"},
                    {"type": "text", "text": "回答"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"q": "x"},
                    },
                ],
            },
        )

    provider = AnthropicMessagesProvider(
        options=ProviderRuntimeOptions(anthropic_user_agent="Anthropic-UA/1"),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(extra_params={"top_p": 0.8}))

    assert captured_path == ["/v1/messages"]
    assert captured_headers == [
        {
            "User-Agent": "Anthropic-UA/1",
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
        }
    ]
    body = captured_body[0]
    assert body["model"] == "claude-test"
    assert body["stream"] is False
    assert body["system"] == "你是助手"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128
    assert body["top_p"] == 0.8
    assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
    assert body["tools"] == [
        {
            "name": "lookup",
            "description": "查天气",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
    ]
    assert result["content"] == "回答"
    assert result["reasoning_content"] == "先想"
    assert result["tool_calls"][0]["id"] == "toolu_1"
    assert result["tool_calls"][0]["function"]["name"] == "lookup"
    assert result["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert result["usage"]["prompt_tokens"] == 3
    assert result["usage"]["completion_tokens"] == 4
    assert result["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_anthropic_response_ignores_stale_model_sampling_fields() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )

    provider = AnthropicMessagesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            request_temperature=0.1,
            request_max_tokens=32,
            model_temperature=0.8,
            model_max_tokens=256,
        )
    )

    body = captured_body[0]
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 32


@pytest.mark.asyncio
async def test_anthropic_response_falls_back_to_model_max_tokens() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            },
        )

    provider = AnthropicMessagesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            request_temperature=None,
            request_max_tokens=None,
            model_temperature=0.3,
            model_max_tokens=256,
        )
    )

    assert captured_body[0]["max_tokens"] == 256
    assert captured_body[0]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_anthropic_stream_response_accumulates_text_reasoning_tool_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        assert body["stream"] is True
        assert request.headers["Accept"] == "text/event-stream"
        content = "".join(
            [
                "event: message_start\n"
                'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"usage":{"input_tokens":2}}}\n\n',
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你"}}\n\n',
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"好"}}\n\n',
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}\n\n',
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"想"}}\n\n',
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}\n\n',
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\"x\\"}"}}\n\n',
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":2}\n\n',
                "event: message_delta\n"
                'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","usage":{"output_tokens":3}}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ]
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = AnthropicMessagesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    assert result["tool_calls"][0]["id"] == "toolu_1"
    assert result["tool_calls"][0]["function"]["name"] == "lookup"
    assert result["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert result["usage"]["prompt_tokens"] == 2
    assert result["usage"]["completion_tokens"] == 3
    assert result["usage"]["total_tokens"] == 5
