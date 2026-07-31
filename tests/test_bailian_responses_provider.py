import json
from typing import cast

import httpx
import pytest

from src.config import MaiDockConfig, build_runtime_options
from src.clients.dashscope import DashScopeClientError
from src.core.common import ProviderRuntimeOptions
from src.host_adapters.bailian_responses_provider.adapter import bailian_responses_path
from src.version import DEFAULT_USER_AGENT
from tests.support.assertions import json_int_at, json_str_at
from tests.support.host_adapters import BailianResponsesProvider

BAILIAN_V1_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _api_provider(*, base_url: str = BAILIAN_V1_BASE_URL) -> dict:
    return {
        "api_key": "bailian-key",
        "auth_type": "bearer",
        "base_url": base_url,
        "default_headers": {},
        "default_query": {},
    }


def _response_request(
    *,
    stream: bool = False,
    base_url: str = BAILIAN_V1_BASE_URL,
    extra_params: dict | None = None,
    request_temperature: int | float | None = 0.3,
    request_max_tokens: int | None = 128,
) -> dict:
    return {
        "model_info": {
            "model_identifier": "qwen-plus",
            "force_stream_mode": stream,
            "extra_params": extra_params or {},
        },
        "api_provider": _api_provider(base_url=base_url),
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


def _completed_response(**extra: object) -> dict:
    payload: dict = {
        "id": "resp_1",
        "model": "qwen-plus",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "回答"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    }
    payload.update(extra)
    return payload


def _bailian_options(overrides: dict | None = None) -> ProviderRuntimeOptions:
    """构造带百炼覆写目录的运行时选项；normalize 填充 store=false 默认。"""

    from src.config import normalize_maidock_config_data

    raw, _ = normalize_maidock_config_data({})
    if overrides:
        raw["bailian_responses"]["response"]["overrides"].update(overrides)
    return build_runtime_options(MaiDockConfig.model_validate(raw))


@pytest.mark.asyncio
async def test_bailian_sync_response_posts_to_v1_responses_with_defaults() -> None:
    captured_path: list[str] = []
    captured_headers: list[dict[str, str]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        captured_headers.append(
            {
                "Authorization": request.headers.get("Authorization", ""),
                "User-Agent": request.headers.get("User-Agent", ""),
            }
        )
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json=_completed_response())

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request())

    assert captured_path == ["/compatible-mode/v1/responses"]
    assert captured_headers[0]["Authorization"] == "Bearer bailian-key"
    assert captured_headers[0]["User-Agent"] == DEFAULT_USER_AGENT
    body = captured_body[0]
    assert body["model"] == "qwen-plus"
    assert body["stream"] is False
    assert body["temperature"] == 0.3
    # Host max_tokens 转译为 OpenAI Responses 规范的 max_output_tokens。
    assert body["max_output_tokens"] == 128
    # 默认 store=false：无状态 Host 链路不产生远端会话存储。
    assert body["store"] is False
    assert isinstance(body["input"], list)
    assert isinstance(body["tools"], list)
    # 百炼 Function 工具省略未声明的 strict 字段。
    assert "strict" not in body["tools"][0]
    assert result["content"] == "回答"
    assert json_int_at(result, "usage", "total_tokens") == 7


@pytest.mark.asyncio
async def test_bailian_extra_params_are_ignored() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(200, json=_completed_response())

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request(extra_params={"top_p": 0.8, "store": True}))

    body = captured_body[0]
    # 两级 extra_params 完全无效；store 仍为默认 false。
    assert "top_p" not in body
    assert body["store"] is False


@pytest.mark.asyncio
async def test_bailian_overrides_win_over_host_fields() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(200, json=_completed_response())

    provider = BailianResponsesProvider(
        options=_bailian_options(
            {
                "max_tokens": "512",
                "temperature": "0.9",
                "text": '{"instructions":"keep"}',
            }
        ),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request(request_temperature=0.1, request_max_tokens=32))

    body = captured_body[0]
    assert body["max_output_tokens"] == 512
    assert body["temperature"] == 0.9
    assert body["text"] == {"instructions": "keep"}
    # 覆写未涉及的 Host 字段仍保留。
    assert body["store"] is False


@pytest.mark.asyncio
async def test_bailian_response_format_maps_to_text_format_and_merge_leaf() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(cast(dict, json.loads(request.content.decode("utf-8"))))
        return httpx.Response(200, json=_completed_response())

    provider = BailianResponsesProvider(
        options=_bailian_options({"text": '{"instructions":"keep"}'}),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(
        {
            **_response_request(),
            "response_format": {"format_type": "json_object"},
        }
    )

    # Host response_format 转译为 text.format；text 覆写只替换同名叶子，保留其他字段。
    assert captured_body[0]["text"] == {"instructions": "keep", "format": {"type": "json_object"}}


@pytest.mark.asyncio
async def test_bailian_stream_accumulates_delta_tool_reasoning_and_usage() -> None:
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

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    assert json_str_at(result, "tool_calls", 0, "id") == "call_1"
    assert json_str_at(result, "tool_calls", 0, "function", "name") == "lookup"
    assert json_int_at(result, "usage", "total_tokens") == 3


@pytest.mark.asyncio
async def test_bailian_unknown_output_items_are_tolerated() -> None:
    """未知 output item（如服务端原生工具）不误报为 Host Function Call，也不丢最终文本。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json=_completed_response(
                output=[
                    {"type": "server_tool_call", "id": "st_1", "name": "web_search"},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "最终回答"}],
                    },
                ]
            ),
        )

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request())

    assert result["content"] == "最终回答"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_bailian_error_payload_raises_structured_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid parameter",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                },
                "request_id": "req_err",
            },
        )

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DashScopeClientError, match="invalid_api_key") as exc_info:
        await provider.get_response(_response_request())
    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.request_id == "req_err"


@pytest.mark.asyncio
async def test_bailian_dashscope_top_level_error_preserves_structure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            429,
            json={
                "code": "Throttling",
                "message": "Too many requests",
                "requestId": "req_top_level",
            },
        )

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DashScopeClientError, match="百炼 Responses") as exc_info:
        await provider.get_response(_response_request())
    assert exc_info.value.code == "Throttling"
    assert exc_info.value.request_id == "req_top_level"
    assert exc_info.value.status_code == 429
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_bailian_sse_error_uses_same_top_level_mapping() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content=(
                b'event: error\ndata: {"code":"InvalidParameter","message":"bad stream","request_id":"req_sse"}\n\n'
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DashScopeClientError, match="InvalidParameter") as exc_info:
        await provider.get_response(_response_request(stream=True))
    assert exc_info.value.code == "InvalidParameter"
    assert exc_info.value.request_id == "req_sse"


@pytest.mark.asyncio
async def test_bailian_error_message_is_truncated() -> None:
    """上游超长错误消息被截断，避免膨胀用户可见错误与日志。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "x" * 5000,
                    "type": "invalid_request_error",
                    "code": "bad_request",
                },
            },
        )

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(Exception) as exc_info:
        await provider.get_response(_response_request())
    assert len(str(exc_info.value)) < 500
    assert "…" in str(exc_info.value)


def test_bailian_responses_path_validates_base_url() -> None:
    assert bailian_responses_path(BAILIAN_V1_BASE_URL) == "responses"
    assert bailian_responses_path("https://dashscope.aliyuncs.com/compatible-mode/v1/") == "responses"
    assert bailian_responses_path("https://dashscope-intl.aliyuncs.com/compatible-mode/v1") == "responses"

    with pytest.raises(ValueError, match="/v1"):
        bailian_responses_path("https://dashscope.aliyuncs.com/compatible-mode/v1/responses")
    with pytest.raises(ValueError, match="/api/v1"):
        bailian_responses_path("https://dashscope.aliyuncs.com/api/v1")
    with pytest.raises(ValueError, match="/v1"):
        bailian_responses_path("https://relay.example/custom")


@pytest.mark.asyncio
async def test_bailian_uses_host_base_url_and_rejects_full_endpoint() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(200, json=_completed_response())

    provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request())
    assert captured_url == [f"{BAILIAN_V1_BASE_URL}/responses"]

    bad_provider = BailianResponsesProvider(
        options=_bailian_options(),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="/v1"):
        await bad_provider.get_response(
            _response_request(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/responses")
        )
