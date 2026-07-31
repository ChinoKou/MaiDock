import base64
import json
from dataclasses import replace
from typing import cast

import httpx
import pytest

from src.config import MaiDockConfig, build_runtime_options, normalize_maidock_config_data
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.host_adapters.xiaomi_mimo_provider.chat import MIMO_CHAT_COMPLETIONS_ENDPOINT
from tests.support.host_adapters import XiaomiMimoProvider


def _mimo_options(**chat_overrides: str) -> ProviderRuntimeOptions:
    """按生产路径构造 Mimo 运行时选项：normalize 填充默认覆写后再构建。"""

    raw, _ = normalize_maidock_config_data({})
    overrides = raw["xiaomi_mimo"]["chat_completion"]["overrides"]
    overrides.update(chat_overrides)
    return build_runtime_options(MaiDockConfig.model_validate(raw))


def _api_provider(*, base_url: str | None = "https://relay.example/v1") -> dict:
    return {
        "api_key": "mimo-key",
        "auth_type": "header",
        "base_url": base_url,
        "default_headers": {},
        "default_query": {},
    }


def _response_request(
    *,
    base_url: str | None = "https://relay.example/v1",
    model_extra_params: dict | None = None,
    request_extra_params: dict | None = None,
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
    return {
        "model_info": {
            "model_identifier": "mimo-vl",
            "extra_params": model_extra_params or {},
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
        "extra_params": request_extra_params or {},
        "temperature": 0.2,
        "max_tokens": 64,
    }


def _audio_request(
    *,
    extra_params: dict | None = None,
    model_extra_params: dict | None = None,
    model: str = "mimo-v2.5-asr",
    audio_bytes: bytes = b"RIFF\x00\x00\x00\x00WAVEfmt ",
) -> dict:
    return {
        "model_info": {
            "model_identifier": model,
            "extra_params": model_extra_params or {},
        },
        "api_provider": _api_provider(),
        "extra_params": extra_params or {},
        "audio_base64": base64.b64encode(audio_bytes).decode(),
    }


def _as_mapping(value: object) -> dict:
    assert isinstance(value, dict)
    return cast(dict, value)


def _as_list(value: object) -> list:
    assert isinstance(value, list)
    return cast(list, value)


@pytest.mark.asyncio
async def test_mimo_chat_uses_api_key_header_and_default_thinking_disabled(
    tmp_path,
) -> None:
    captured_url: list[str] = []
    captured_headers: list[dict[str, str]] = []
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        captured_headers.append(
            {
                "api-key": request.headers["api-key"],
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
                "choices": [
                    {
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
                        }
                    }
                ],
                "usage": {"total_tokens": 7},
            },
        )

    provider = XiaomiMimoProvider(
        options=replace(_mimo_options(), mimo_user_agent="Mimo-UA/1"),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    result = await provider.get_response(
        _response_request(
            with_image=True,
        )
    )

    assert captured_url == [f"https://relay.example/v1/{MIMO_CHAT_COMPLETIONS_ENDPOINT}"]
    assert captured_headers == [{"api-key": "mimo-key", "User-Agent": "Mimo-UA/1"}]
    body = captured_body[0]
    # 默认覆写目录启用 thinking=disabled。
    assert body["thinking"] == {"type": "disabled"}
    assert body["temperature"] == 0.2
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    content = _as_list(_as_mapping(_as_list(body["messages"])[0])["content"])
    assert content[0] == {"type": "text", "text": "你好"}
    assert _as_mapping(content[1])["type"] == "image_url"
    assert _as_list(body["tools"])[0]["function"]["name"] == "lookup"
    assert result["content"] == "回答"
    first_tool_call = _as_mapping(_as_list(result["tool_calls"])[0])
    assert first_tool_call["extra_content"] == {
        "provider": "xiaomi_mimo",
        "xiaomi_mimo": {"raw_arguments": '{"q":"x"}', "reasoning_content": "先想"},
        "tool_call_source": "reasoning",
    }


@pytest.mark.asyncio
async def test_mimo_chat_allows_native_thinking_via_override(
    tmp_path,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    config = MaiDockConfig.model_validate(
        {"xiaomi_mimo": {"chat_completion": {"overrides": {"thinking": '{"type":"enabled"}'}}}}
    )
    provider = XiaomiMimoProvider(
        options=build_runtime_options(config),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    await provider.get_response(_response_request())

    assert captured_body[0]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_mimo_chat_uses_host_base_url(
    tmp_path,
) -> None:
    captured_url: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_url.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    provider = XiaomiMimoProvider(
        options=_mimo_options(),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    await provider.get_response(_response_request(base_url="https://relay.example/v1"))

    assert captured_url == [f"https://relay.example/v1/{MIMO_CHAT_COMPLETIONS_ENDPOINT}"]


@pytest.mark.asyncio
async def test_mimo_chat_accepts_official_max_completion_tokens_alias(
    tmp_path,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    config = MaiDockConfig.model_validate({"xiaomi_mimo": {"chat_completion": {"overrides": {"max_tokens": "96"}}}})
    request = _response_request()
    request["max_tokens"] = None
    provider = XiaomiMimoProvider(
        options=build_runtime_options(config),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )
    await provider.get_response(request)

    assert captured_body[0]["max_completion_tokens"] == 96
    assert "max_tokens" not in captured_body[0]


@pytest.mark.asyncio
async def test_mimo_chat_override_max_tokens_wins_over_host_field(
    tmp_path,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    provider = XiaomiMimoProvider(
        options=_mimo_options(max_tokens="128"),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    await provider.get_response(_response_request())

    assert captured_body[0]["max_completion_tokens"] == 128
    assert "max_tokens" not in captured_body[0]


@pytest.mark.asyncio
async def test_mimo_chat_extra_params_are_ignored(
    tmp_path,
) -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    request = _response_request(
        request_extra_params={"max_completion_tokens": 96},
        model_extra_params={"max_completion_tokens": 32},
    )
    request["max_tokens"] = None
    provider = XiaomiMimoProvider(
        options=_mimo_options(),
        transport=httpx.MockTransport(handler),
        state_store=PluginStateStore(tmp_path / "state.sqlite3"),
    )

    await provider.get_response(request)

    assert "max_completion_tokens" not in captured_body[0]


@pytest.mark.asyncio
async def test_mimo_audio_transcription_uses_asr_request_without_prompt() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好世界"}}], "usage": {}})

    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_audio_transcriptions(_audio_request())

    assert result["content"] == "你好世界"
    body = captured_body[0]
    assert body["model"] == "mimo-v2.5-asr"
    assert body["stream"] is False
    assert body["asr_options"] == {"language": "auto"}
    content = _as_list(_as_mapping(_as_list(body["messages"])[0])["content"])
    assert content == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:audio/wav;base64,{_audio_request()['audio_base64']}",
                "format": "wav",
            },
        }
    ]
    for unsupported_field in ("max_completion_tokens", "max_tokens", "prompt", "thinking"):
        assert unsupported_field not in body


@pytest.mark.asyncio
async def test_mimo_audio_transcription_alias_uses_mp3_asr_request() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured_body.append(cast(dict, body))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": [{"text": "转写"}]}}],
                "usage": {},
            },
        )

    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))

    result = await provider.get_audio_transcriptions(
        _audio_request(
            model="relay-mimo-asr",
            extra_params={
                "format": "mp3",
                "max_completion_tokens": 256,
                "prompt": "旧转录提示词",
            },
            audio_bytes=b"ID3\x04\x00\x00\x00\x00\x00\x00",
        )
    )

    assert result["content"] == "转写"
    body = captured_body[0]
    assert body["model"] == "relay-mimo-asr"
    content = _as_list(_as_mapping(_as_list(body["messages"])[0])["content"])
    assert len(content) == 1
    input_audio = _as_mapping(_as_mapping(content[0])["input_audio"])
    assert input_audio["data"].startswith("data:audio/mpeg;base64,")
    assert input_audio["format"] == "mp3"
    for unsupported_field in ("max_completion_tokens", "max_tokens", "prompt", "thinking"):
        assert unsupported_field not in body


@pytest.mark.asyncio
async def test_mimo_audio_transcription_rejects_missing_or_invalid_audio_base64() -> None:
    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions())

    with pytest.raises(ValueError, match="audio_base64"):
        await provider.get_audio_transcriptions(
            {
                "model_info": {"model_identifier": "mimo-audio"},
                "api_provider": _api_provider(),
            }
        )

    with pytest.raises(ValueError, match="Base64"):
        await provider.get_audio_transcriptions(
            {
                "model_info": {"model_identifier": "mimo-audio"},
                "api_provider": _api_provider(),
                "audio_base64": "not-base64",
            }
        )
