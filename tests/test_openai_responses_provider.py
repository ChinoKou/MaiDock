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
from src.providers.openai_responses_provider.provider import OpenAIResponsesProvider


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
    request_temperature: int | float | None = 0.3,
    request_max_tokens: int | None = 128,
    model_temperature: int | float | None = None,
    model_max_tokens: int | None = None,
) -> dict:
    model_info: dict = {
        "model_identifier": "gpt-test",
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
        "message_list": [{"role": "user", "parts": [{"type": "text", "text": "你好"}]}],
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


def _embedding_request() -> dict:
    return {
        "model_info": {"model_identifier": "text-embedding-3-small"},
        "api_provider": _api_provider(),
        "embedding_input": "hello",
    }


def _audio_request(extra_params: dict | None = None) -> dict:
    return {
        "model_info": {
            "model_identifier": "gpt-4o-mini-transcribe",
            "extra_params": extra_params or {},
        },
        "api_provider": _api_provider(),
        "audio_base64": base64.b64encode(b"fake-wav-data").decode("utf-8"),
    }


@pytest.mark.asyncio
async def test_openai_provider_uses_configured_user_agent() -> None:
    captured_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(request.headers["User-Agent"])
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
            },
        )

    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(openai_user_agent="OpenAI-UA/1"),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request())

    assert captured_headers == ["OpenAI-UA/1"]


@pytest.mark.asyncio
async def test_openai_non_stream_response_posts_responses_body_and_parses_tool_calls() -> None:
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
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning_summary",
                        "summary": [{"type": "summary_text", "text": "先想"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                        "status": "completed",
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        )

    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(extra_params={"top_p": 0.8}))

    assert captured_path == ["/v1/responses"]
    body = captured_body[0]
    assert body["model"] == "gpt-test"
    assert body["stream"] is False
    assert body["temperature"] == 0.3
    assert body["max_output_tokens"] == 128
    assert body["top_p"] == 0.8
    assert result["content"] == "回答"
    assert result["reasoning_content"] == "先想"
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][0]["function"]["name"] == "lookup"
    assert result["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_openai_response_ignores_stale_model_sampling_fields() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
            },
        )

    provider = OpenAIResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

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
    assert body["max_output_tokens"] == 32


@pytest.mark.asyncio
async def test_openai_response_does_not_fall_back_to_model_sampling_fields() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
            },
        )

    provider = OpenAIResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        _response_request(
            request_temperature=None,
            request_max_tokens=None,
            model_temperature=0.8,
            model_max_tokens=256,
        )
    )

    body = captured_body[0]
    assert body["temperature"] == 0.8
    assert body["max_output_tokens"] == 256


@pytest.mark.asyncio
async def test_openai_response_parameter_policy_can_drop_reject_and_override_transport_fields() -> None:
    captured_body: list[dict] = []
    captured_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        captured_headers.append(request.headers.get("X-Policy"))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
            },
        )

    policies = ParameterPolicyRegistry(
        openai_responses=ProviderCapabilityPolicies(
            response=ParameterPolicy(
                rejected_paths=("headers.Authorization-Override",),
                disabled_paths=("body.temperature",),
                override_params={
                    "body": {"reasoning": {"effort": "low"}},
                    "headers": {"X-Policy": "yes"},
                },
            )
        )
    )
    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(parameter_policies=policies),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request(extra_params={"body": {"temperature": 1.0}}))

    body = captured_body[0]
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert captured_headers == ["yes"]

    with pytest.raises(ValueError, match="headers.Authorization-Override"):
        await provider.get_response(_response_request(extra_params={"headers": {"Authorization-Override": "blocked"}}))


@pytest.mark.asyncio
async def test_openai_stream_response_accumulates_delta_tool_reasoning_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        assert body["stream"] is True
        content = "".join(
            [
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"你"}\n\n',
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"好"}\n\n',
                "event: response.reasoning_summary_text.delta\n"
                'data: {"type":"response.reasoning_summary_text.delta","delta":"想"}\n\n',
                "event: response.output_item.added\n"
                'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"lookup"}}\n\n',
                "event: response.function_call_arguments.done\n"
                'data: {"type":"response.function_call_arguments.done","item_id":"item_1","arguments":"{\\"q\\":\\"x\\"}"}\n\n',
                "event: response.completed\n"
                'data: {"type":"response.completed","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}\n\n',
            ]
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    assert result["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert result["usage"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_openai_embedding_posts_to_embeddings_endpoint() -> None:
    captured_path: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"embedding": [1, "2.5"]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(_embedding_request())

    assert captured_path == ["/v1/embeddings"]
    assert result["embedding"] == [1.0, 2.5]
    assert result["usage"]["prompt_tokens"] == 2
    assert result["raw_data"] == {
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": "***", "total_tokens": "***"},
    }


@pytest.mark.asyncio
async def test_openai_audio_transcription_posts_multipart_and_parses_text() -> None:
    captured_path: list[str] = []
    captured_content_type: list[str] = []
    captured_body: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        captured_content_type.append(request.headers["Content-Type"])
        captured_body.append(request.content)
        return httpx.Response(200, json={"text": "转写文本"})

    provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(include_raw_data=True),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_audio_transcriptions(_audio_request(extra_params={"body": {"language": "zh"}}))

    assert captured_path == ["/v1/audio/transcriptions"]
    assert "multipart/form-data" in captured_content_type[0]
    assert b"gpt-4o-mini-transcribe" in captured_body[0]
    assert b"language" in captured_body[0]
    assert b"zh" in captured_body[0]
    assert result["content"] == "转写文本"


def test_extract_embedding_reports_non_numeric_item_index() -> None:
    payload = {"data": [{"embedding": [1, "2.5", {"bad": True}]}]}

    with pytest.raises(ValueError, match=r"embedding\[2\].*dict"):
        OpenAIResponsesProvider._extract_embedding(payload)


@pytest.mark.asyncio
async def test_openai_response_format_maps_to_responses_text_config() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {},
            },
        )

    provider = OpenAIResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(
        {
            **_response_request(),
            "response_format": {
                "format_type": "json_schema",
                "schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            },
        }
    )

    assert result["content"] == "ok"
    assert captured_body[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }
    }


@pytest.mark.asyncio
async def test_openai_failed_response_and_stream_error_raise_clear_errors() -> None:
    async def non_stream_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "status": "failed",
                "error": {"message": "bad"},
                "output": [],
                "usage": {},
            },
        )

    non_stream_provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(non_stream_handler),
    )
    with pytest.raises(ValueError, match="failed"):
        await non_stream_provider.get_response(_response_request())

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        del request
        content = 'event: error\ndata: {"type":"error","message":"stream bad"}\n\n'
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    stream_provider = OpenAIResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(stream_handler),
    )
    with pytest.raises(Exception, match="stream bad"):
        await stream_provider.get_response(_response_request(stream=True))
