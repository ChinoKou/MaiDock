import json
import logging
from collections.abc import Callable

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.providers.common.httpx import HttpxProviderError
from src.providers.openai_responses_provider.responses import (
    create_responses_mapper as create_openai_mapper,
)
from src.providers.responses_family.responses import ResponsesMapper
from src.providers.volcengine_ark_provider.provider import (
    VolcengineArkResponsesProvider,
)
from src.providers.volcengine_ark_provider.responses import (
    create_responses_mapper as create_ark_mapper,
)

type MapperFactory = Callable[..., ResponsesMapper]


def _length_payload(*, include_text: bool = True, include_reasoning: bool = True) -> dict:
    output: list[dict] = []
    if include_reasoning:
        output.append(
            {
                "type": "reasoning_summary",
                "summary": [{"type": "summary_text", "text": "截断思考"}],
            }
        )
    if include_text:
        output.append(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "截断回答"}],
            }
        )
    output.append(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"x"}',
            "status": "incomplete",
        }
    )
    return {
        "id": "resp_incomplete",
        "model": "test-model",
        "status": "incomplete",
        "incomplete_details": {"reason": "length"},
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.parametrize("factory", [create_openai_mapper, create_ark_mapper])
def test_length_incomplete_returns_partial_text_reasoning_tools_and_usage(
    factory: MapperFactory,
) -> None:
    mapper = factory(options=ProviderRuntimeOptions(), logger=logging.getLogger("test"))

    result = mapper.convert_response(_length_payload())

    assert result.content == "截断回答"
    assert result.reasoning_content == "截断思考"
    assert result.tool_calls[0].id == "call_1"
    assert result.usage is not None
    assert result.usage.total_tokens == 15


def test_length_incomplete_allows_reasoning_only_but_rejects_fully_empty() -> None:
    mapper = create_ark_mapper(options=ProviderRuntimeOptions(), logger=logging.getLogger("test"))
    reasoning_only = _length_payload(include_text=False)
    reasoning_only["output"] = reasoning_only["output"][:1]

    result = mapper.convert_response(reasoning_only)
    assert result.content == ""
    assert result.reasoning_content == "截断思考"

    empty = _length_payload(include_text=False, include_reasoning=False)
    empty["output"] = []
    with pytest.raises(ValueError, match="Volcengine Ark"):
        mapper.convert_response(empty)


def test_non_length_incomplete_still_raises() -> None:
    mapper = create_ark_mapper(options=ProviderRuntimeOptions(), logger=logging.getLogger("test"))
    payload = _length_payload()
    payload["incomplete_details"] = {"reason": "content_filter"}

    with pytest.raises(ValueError, match="incomplete"):
        mapper.convert_response(payload)


def _stream_request() -> dict:
    return {
        "model_info": {
            "model_identifier": "doubao-test",
            "force_stream_mode": True,
            "extra_params": {},
        },
        "api_provider": {
            "api_key": "ark-key",
            "auth_type": "bearer",
            "base_url": "https://ark.example/api/v3",
            "default_headers": {},
            "default_query": {},
        },
        "message_list": [{"role": "user", "parts": [{"type": "text", "text": "你好"}]}],
        "tool_options": [],
    }


@pytest.mark.asyncio
async def test_stream_length_incomplete_uses_accumulated_delta_when_terminal_output_is_empty() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"截断内容"}\n\n'
            "event: response.incomplete\n"
            'data: {"type":"response.incomplete","response":'
            '{"id":"resp_1","model":"doubao-test","status":"incomplete",'
            '"incomplete_details":{"reason":"length"},"output":[],"usage":{"output_tokens":5}}}\n\n'
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_stream_request())

    assert result["content"] == "截断内容"
    assert result["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_non_length_incomplete_still_raises() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = (
            "event: response.incomplete\n"
            'data: {"type":"response.incomplete","response":'
            '{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}}\n\n'
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(HttpxProviderError, match="content_filter"):
        await provider.get_response(_stream_request())


@pytest.mark.asyncio
async def test_stream_length_incomplete_merges_reasoning_tools_and_split_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        events = [
            (
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "截断思考",
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": "截断回答",
                },
            ),
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "id": "item_1",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            ),
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "item_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "delta": '{"q":"x"}',
                    "usage": {"input_tokens": 10},
                },
            ),
            (
                "response.incomplete",
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_1",
                        "model": "doubao-test",
                        "status": "incomplete",
                        "incomplete_details": {"reason": "length"},
                        "output": [],
                        "usage": {
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                },
            ),
        ]
        content = "".join(
            f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            for event_type, payload in events
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_stream_request())

    assert result["content"] == "截断回答"
    assert result["reasoning_content"] == "截断思考"
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": {"q": "x"},
    }
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
    assert result["usage"]["total_tokens"] == 15
