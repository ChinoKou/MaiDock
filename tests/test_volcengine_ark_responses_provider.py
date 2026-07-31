import json
from dataclasses import replace
from typing import cast

import httpx
import pytest

from src.config import MaiDockConfig, build_runtime_options
from src.core.common import ArkBuiltinEndpointMode, ProviderRuntimeOptions
from tests.support.host_adapters import VolcengineArkResponsesProvider
from src.host_adapters.volcengine_ark_provider.responses import (
    ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT,
    ARK_RESPONSES_ENDPOINT,
    VOLCENGINE_ARK_AGENT_PLAN_BASE_URL,
    VOLCENGINE_ARK_BASE_URL,
    VOLCENGINE_ARK_CODING_PLAN_BASE_URL,
)


def _ark_overrides_options(overrides: dict, *, capability: str = "response") -> ProviderRuntimeOptions:
    """构造带 ARK 覆写目录的运行时选项。"""

    config = MaiDockConfig.model_validate({"volcengine_ark": {capability: {"overrides": overrides}}})
    return build_runtime_options(config)


def _api_provider(*, base_url: str | None = VOLCENGINE_ARK_BASE_URL) -> dict:
    return {
        "api_key": "ark-key",
        "auth_type": "bearer",
        "base_url": base_url,
        "default_headers": {},
        "default_query": {},
    }


def _response_request(*, stream: bool = False, extra_params: dict | None = None) -> dict:
    return {
        "model_info": {
            "model_identifier": "doubao-test",
            "force_stream_mode": stream,
            "extra_params": extra_params or {},
        },
        "api_provider": _api_provider(),
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
        "temperature": 0.3,
        "max_tokens": 128,
    }


def _embedding_request(*, extra_params: dict | None = None) -> dict:
    return {
        "model_info": {
            "model_identifier": "doubao-embedding",
            "extra_params": extra_params or {},
        },
        "api_provider": _api_provider(),
        "embedding_input": "文本向量",
    }


def _as_list(value: object) -> list:
    assert isinstance(value, list)
    return cast(list, value)


def _as_mapping(value: object) -> dict:
    assert isinstance(value, dict)
    return cast(dict, value)


@pytest.mark.asyncio
async def test_ark_non_stream_response_posts_responses_body_and_parses_text_tool_reasoning() -> None:
    captured_path: list[str] = []
    captured_request_id: list[str | None] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_path.append(request.url.path)
        captured_request_id.append(request.headers.get("X-Client-Request-Id"))
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "doubao-test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning_summary",
                        "summary": [{"type": "summary_text", "text": "先思考"}],
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

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(extra_params={"top_p": 0.8}))

    assert captured_path == ["/api/v3/" + ARK_RESPONSES_ENDPOINT]
    assert captured_request_id[0]
    body = captured_body[0]
    assert body["model"] == "doubao-test"
    assert body["stream"] is False
    assert body["temperature"] == 0.3
    assert body["max_output_tokens"] == 128
    # extra_params 完全无效：top_p 不会进入请求体。
    assert "top_p" not in body
    assert isinstance(body["input"], list)
    assert isinstance(body["tools"], list)
    assert result["content"] == "回答"
    assert result["reasoning_content"] == "先思考"
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "function": {"name": "lookup", "arguments": {"q": "x"}},
            "extra_content": {
                "provider": "volcengine_ark_responses",
                "tool_call_source": "reasoning",
                "openai_responses": {
                    "item_id": None,
                    "status": "completed",
                    "raw_arguments": '{"q":"x"}',
                    "generated_call_id": False,
                },
            },
        }
    ]
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 7


@pytest.mark.asyncio
async def test_ark_extra_tools_append_after_host_tools_and_enable_beta_headers() -> None:
    captured_headers: list[dict[str, str | None]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.append(
            {
                "web_search": request.headers.get("ark-beta-web-search"),
                "mcp": request.headers.get("ark-beta-mcp"),
            }
        )
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "doubao-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=_ark_overrides_options({"tools": '[{"type":"web_search"},{"type":"mcp","server_label":"local"}]'}),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request())

    assert result["content"] == "回答"
    assert captured_headers == [{"web_search": "true", "mcp": "true"}]
    tools = _as_list(captured_body[0]["tools"])
    assert _as_mapping(tools[0])["name"] == "lookup"
    assert tools[1:] == [{"type": "web_search"}, {"type": "mcp", "server_label": "local"}]


@pytest.mark.asyncio
async def test_ark_stream_accumulates_delta_tool_reasoning_and_usage() -> None:
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
                "event: response.function_call_arguments.delta\n"
                'data: {"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"{\\"q\\":"}\n\n',
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

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "你好"
    assert result["reasoning_content"] == "想"
    tool_calls = _as_list(result["tool_calls"])
    first_tool_call = _as_mapping(tool_calls[0])
    assert first_tool_call["id"] == "call_1"
    assert first_tool_call["function"] == {"name": "lookup", "arguments": {"q": "x"}}
    usage = _as_mapping(result["usage"])
    assert usage["total_tokens"] == 3


@pytest.mark.asyncio
async def test_ark_trailing_assistant_message_is_marked_partial() -> None:
    """planner 形状的请求：末位 assistant 预填必须带上 ARK 续写模式必填的 partial。"""

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
                        "content": [{"type": "output_text", "text": "分析结论"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(
        {
            **_response_request(),
            "message_list": [
                {"role": "system", "parts": [{"type": "text", "text": "你是规划器"}]},
                {"role": "user", "parts": [{"type": "text", "text": "现在怎么办"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "我需要输出对发言的分析"}]},
            ],
        }
    )

    input_items = _as_list(captured_body[0]["input"])
    assert _as_mapping(input_items[-1]) == {
        "role": "assistant",
        "content": "我需要输出对发言的分析",
        "partial": True,
    }
    assert result["content"] == "分析结论"


@pytest.mark.asyncio
async def test_ark_non_trailing_assistant_message_has_no_partial_flag() -> None:
    """assistant 只在末位才触发续写模式；历史位置的 assistant 不得带 partial。"""

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
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    await provider.get_response(
        {
            **_response_request(),
            "message_list": [
                {"role": "user", "parts": [{"type": "text", "text": "你好"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "上轮回复"}]},
                {"role": "user", "parts": [{"type": "text", "text": "继续"}]},
            ],
        }
    )

    input_items = _as_list(captured_body[0]["input"])
    assert _as_mapping(input_items[1]) == {"role": "assistant", "content": "上轮回复"}


@pytest.mark.asyncio
async def test_ark_stream_trailing_assistant_message_is_marked_partial() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        content = "".join(
            [
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"好"}\n\n',
                'event: response.completed\ndata: {"type":"response.completed","usage":{"total_tokens":1}}\n\n',
            ]
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = VolcengineArkResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(
        {
            **_response_request(stream=True),
            "message_list": [
                {"role": "user", "parts": [{"type": "text", "text": "现在怎么办"}]},
                {"role": "assistant", "parts": [{"type": "text", "text": "我需要输出分析"}]},
            ],
        }
    )

    input_items = _as_list(captured_body[0]["input"])
    assert _as_mapping(input_items[-1]) == {
        "role": "assistant",
        "content": "我需要输出分析",
        "partial": True,
    }
    assert result["content"] == "好"


@pytest.mark.asyncio
async def test_ark_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_response(
        {
            **_response_request(),
            "api_provider": _api_provider(base_url="https://relay.example/custom"),
        }
    )

    assert captured_url == [f"{VOLCENGINE_ARK_BASE_URL}/{ARK_RESPONSES_ENDPOINT}"]
    assert result["content"] == "回答"


@pytest.mark.asyncio
async def test_ark_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(
        {
            **_response_request(),
            "api_provider": _api_provider(base_url="https://relay.example/custom"),
        }
    )

    assert captured_url == [f"https://relay.example/custom/api/v3/{ARK_RESPONSES_ENDPOINT}"]
    assert result["content"] == "回答"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_base"),
    [
        ("standard", VOLCENGINE_ARK_BASE_URL),
        ("agent_plan", VOLCENGINE_ARK_AGENT_PLAN_BASE_URL),
        ("coding_plan", VOLCENGINE_ARK_CODING_PLAN_BASE_URL),
    ],
)
async def test_ark_builtin_endpoint_mode_selects_base_url(mode: ArkBuiltinEndpointMode, expected_base: str) -> None:
    """三种内置端点各自拼出单前缀 URL——prefix 与 base 不同步会拼出 /api/plan/v3/api/v3/responses。"""

    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_builtin_endpoint_mode=mode),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(_response_request())

    assert captured_url == [f"{expected_base}/{ARK_RESPONSES_ENDPOINT}"]


@pytest.mark.asyncio
async def test_ark_builtin_endpoint_mode_is_inert_without_force_official_endpoint() -> None:
    """关闭原生 endpoint 时订阅端点选项不得生效，Host base_url 行为与既有逐字节一致。"""

    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_force_official_endpoint=False,
            volcengine_builtin_endpoint_mode="agent_plan",
        ),
        transport=httpx.MockTransport(handler),
    )

    await provider.get_response(
        {
            **_response_request(),
            "api_provider": _api_provider(base_url="https://relay.example/custom"),
        }
    )

    assert captured_url == [f"https://relay.example/custom/api/v3/{ARK_RESPONSES_ENDPOINT}"]


@pytest.mark.asyncio
async def test_ark_plan_endpoint_disables_prefix_cache() -> None:
    """订阅端点上 tokenization/caching 无文档；前缀缓存必须整体停用，不发辅助请求。"""

    captured_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "回答"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(
            volcengine_prefix_cache_enabled=True,
            volcengine_builtin_endpoint_mode="coding_plan",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(
        {
            **_response_request(),
            "message_list": [
                {"role": "system", "parts": [{"type": "text", "text": "系统前缀" * 200}]},
                {"role": "user", "parts": [{"type": "text", "text": "你好"}]},
            ],
        }
    )

    # 只应有一次 responses 主请求：没有 tokenization、没有缓存创建请求。
    assert captured_paths == ["/api/coding/v3/" + ARK_RESPONSES_ENDPOINT]
    assert result["content"] == "回答"


@pytest.mark.asyncio
async def test_ark_stream_prefers_completed_response_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        content = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"ignored"}\n\n'
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"status":"completed","output_text":"final","output":[],"usage":{}}}\n\n'
        )
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_response(_response_request(stream=True))

    assert result["content"] == "final"


@pytest.mark.asyncio
async def test_ark_failed_response_and_stream_error_raise_clear_errors() -> None:
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

    non_stream_provider = VolcengineArkResponsesProvider(
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

    stream_provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(stream_handler),
    )
    with pytest.raises(Exception, match="stream bad"):
        await stream_provider.get_response(_response_request(stream=True))

    async def bare_error_handler(request: httpx.Request) -> httpx.Response:
        del request
        content = 'data: {"error":{"message":"bare bad"}}\n\n'
        return httpx.Response(
            200,
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    bare_error_provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(bare_error_handler),
    )
    with pytest.raises(Exception, match="bare bad"):
        await bare_error_provider.get_response(_response_request(stream=True))


@pytest.mark.asyncio
async def test_ark_embedding_uses_multimodal_endpoint_and_object_data_shape() -> None:
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
                "data": {"object": "embedding", "embedding": [1, "2.5"]},
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=replace(
            _ark_overrides_options({"dimensions": "128", "sparse_embedding": "true"}, capability="embeddings"),
            include_raw_data=True,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(_embedding_request())

    assert captured_path == ["/api/v3/" + ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT]
    assert captured_body == [
        {
            "model": "doubao-embedding",
            "encoding_format": "float",
            "dimensions": 128,
            "sparse_embedding": {"type": "enabled"},
            "input": [{"type": "text", "text": "文本向量"}],
        }
    ]
    assert result["embedding"] == [1.0, 2.5]
    usage = _as_mapping(result["usage"])
    raw_data = _as_mapping(result["raw_data"])
    assert usage["prompt_tokens"] == 2
    assert raw_data["data"] == {"object": "embedding", "embedding": [1, "2.5"]}


@pytest.mark.asyncio
async def test_ark_embedding_default_force_official_endpoint_ignores_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": {"object": "embedding", "embedding": [1, "2.5"]},
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = VolcengineArkResponsesProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_embedding(
        {
            **_embedding_request(),
            "api_provider": _api_provider(base_url="https://relay.example/custom"),
        }
    )

    assert captured_url == [f"{VOLCENGINE_ARK_BASE_URL}/{ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]


@pytest.mark.asyncio
async def test_ark_embedding_force_official_endpoint_false_uses_host_base_url() -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": {"object": "embedding", "embedding": [1, "2.5"]},
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(volcengine_force_official_endpoint=False),
        transport=httpx.MockTransport(handler),
    )

    result = await provider.get_embedding(
        {
            **_embedding_request(),
            "api_provider": _api_provider(base_url="https://relay.example/custom"),
        }
    )

    assert captured_url == [f"https://relay.example/custom/api/v3/{ARK_MULTIMODAL_EMBEDDINGS_ENDPOINT}"]
    assert result["embedding"] == [1.0, 2.5]
