import base64
import json
from typing import cast

import httpx
import pytest

from src.core.common import ProviderRuntimeOptions
from src.providers.common import audio as audio_module
from src.providers.common.audio import detect_audio_format, prepare_base64_audio
from src.providers.volcengine_ark_provider.provider import (
    VolcengineArkResponsesProvider,
)
from src.providers.xiaomi_mimo_provider.provider import XiaomiMimoProvider


def _api_provider(api_key: str) -> dict:
    return {
        "api_key": api_key,
        "base_url": "https://relay.example/v1",
        "default_headers": {},
        "default_query": {},
    }


def _wav_bytes() -> bytes:
    return b"RIFF\x00\x00\x00\x00WAVEfmt "


def _audio_request(model: str, *, extra_params: dict | None = None, max_tokens: int | None = None) -> dict:
    return {
        "model_info": {"model_identifier": model, "extra_params": {}},
        "api_provider": _api_provider("audio-key"),
        "audio_base64": base64.b64encode(_wav_bytes()).decode(),
        "extra_params": extra_params or {},
        "max_tokens": max_tokens,
    }


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (_wav_bytes(), "wav"),
        (b"ID3\x04\x00\x00", "mp3"),
        (b"\xff\xf1\x50\x80", "aac"),
        (b"\x00\x00\x00\x18ftypM4A ", "m4a"),
        (b"fLaC\x00", "flac"),
        (b"OggS\x00", "ogg"),
    ],
)
def test_detect_audio_format(data: bytes, expected: str) -> None:
    assert detect_audio_format(data) == expected


def test_audio_format_conflict_and_unknown_are_exposed() -> None:
    encoded = base64.b64encode(_wav_bytes()).decode()
    with pytest.raises(ValueError, match="mp3.*wav"):
        prepare_base64_audio(
            encoded,
            {"format": "mp3"},
            provider_label="test",
            allowed_formats={"mp3", "wav"},
        )
    with pytest.raises(ValueError, match="format.*audio_format"):
        prepare_base64_audio(
            base64.b64encode(b"unknown").decode(),
            {},
            provider_label="test",
            allowed_formats={"wav"},
        )


def test_audio_size_limit_is_checked_before_base64_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decode(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("超限输入不应进入 Base64 解码")

    monkeypatch.setattr(audio_module.base64, "b64decode", fail_decode)
    with pytest.raises(ValueError, match="Base64"):
        prepare_base64_audio(
            "A" * 12,
            {},
            provider_label="test",
            allowed_formats={"wav"},
            max_base64_chars=8,
        )
    with pytest.raises(ValueError, match="test.*0 MiB"):
        prepare_base64_audio(
            "A" * 12,
            {},
            provider_label="test",
            allowed_formats={"wav"},
            max_decoded_bytes=3,
        )


@pytest.mark.asyncio
async def test_ark_audio_transcription_uses_responses_input_audio() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured_body.append(cast(dict, body))
        assert request.url.path == "/api/v3/responses"
        return httpx.Response(
            200,
            json={
                "id": "resp_audio",
                "model": "doubao-audio",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "转录结果"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        )

    provider = VolcengineArkResponsesProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )
    result = await provider.get_audio_transcriptions(_audio_request("doubao-audio", max_tokens=128))

    assert result["content"] == "转录结果"
    body = captured_body[0]
    assert body["stream"] is False
    assert body["max_output_tokens"] == 128
    assert "caching" not in body
    content = body["input"][0]["content"]
    assert content[0]["type"] == "input_audio"
    assert content[0]["audio_url"].startswith("data:audio/wav;base64,")
    assert content[1] == {
        "type": "input_text",
        "text": "请识别音频中的内容，以文字形式返回识别结果。",
    }


@pytest.mark.asyncio
async def test_mimo_dedicated_asr_has_single_audio_and_language() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": "专用转录"}}], "usage": {}})

    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(mimo_audio_transcription_language="zh"),
        transport=httpx.MockTransport(handler),
    )
    result = await provider.get_audio_transcriptions(_audio_request("mimo-v2.5-asr", max_tokens=128))

    assert result["content"] == "专用转录"
    body = captured_body[0]
    assert body["asr_options"] == {"language": "zh"}
    assert "max_completion_tokens" not in body
    assert "thinking" not in body
    content = body["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_audio"
    assert content[0]["input_audio"]["format"] == "wav"


@pytest.mark.asyncio
async def test_mimo_dedicated_asr_preserves_forwarded_future_options() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "专用转录"}}], "usage": {}})

    provider = XiaomiMimoProvider(
        options=ProviderRuntimeOptions(),
        transport=httpx.MockTransport(handler),
    )
    await provider.get_audio_transcriptions(
        _audio_request(
            "mimo-v2.5-asr",
            extra_params={
                "body": {
                    "asr_options": {"language": "zh", "future_option": True},
                    "future_top_level": "keep",
                    "temperature": 0.2,
                    "tools": [{"type": "function"}],
                }
            },
        )
    )

    body = captured_body[0]
    assert body["asr_options"] == {"language": "zh", "future_option": True}
    assert body["future_top_level"] == "keep"
    assert "temperature" not in body
    assert "tools" not in body


@pytest.mark.asyncio
async def test_mimo_asr_rejects_invalid_asr_options_schema_for_model_alias() -> None:
    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions())

    with pytest.raises(TypeError, match="asr_options.*list"):
        await provider.get_audio_transcriptions(
            _audio_request(
                "relay-mimo-audio",
                extra_params={"body": {"asr_options": []}},
            )
        )


@pytest.mark.asyncio
async def test_mimo_asr_uses_dedicated_protocol_for_model_alias() -> None:
    captured_body: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured_body.append(cast(dict, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": "别名转录"}}], "usage": {}})

    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions(), transport=httpx.MockTransport(handler))
    await provider.get_audio_transcriptions(
        _audio_request(
            "relay-mimo-audio",
            extra_params={"max_completion_tokens": 512, "prompt": "只返回文字"},
            max_tokens=256,
        )
    )

    body = captured_body[0]
    assert body["model"] == "relay-mimo-audio"
    assert body["asr_options"] == {"language": "auto"}
    assert body["messages"][0]["content"] == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:audio/wav;base64,{base64.b64encode(_wav_bytes()).decode()}",
                "format": "wav",
            },
        }
    ]
    for unsupported_field in ("max_completion_tokens", "max_tokens", "prompt", "thinking"):
        assert unsupported_field not in body


@pytest.mark.parametrize(
    "audio_bytes",
    [
        b"fLaC\x00",
        b"\x00\x00\x00\x18ftypM4A ",
        b"OggS\x00",
    ],
    ids=["flac", "m4a", "ogg"],
)
@pytest.mark.asyncio
async def test_mimo_asr_rejects_non_asr_audio_formats_for_model_alias(audio_bytes: bytes) -> None:
    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions())
    request = _audio_request("relay-mimo-audio")
    request["audio_base64"] = base64.b64encode(audio_bytes).decode()

    with pytest.raises(ValueError, match="mp3, wav"):
        await provider.get_audio_transcriptions(request)


@pytest.mark.asyncio
async def test_mimo_asr_rejects_oversized_base64_for_model_alias() -> None:
    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions())
    request = _audio_request("relay-mimo-audio")
    request["audio_base64"] = "A" * (10 * 1024 * 1024 + 4)

    with pytest.raises(ValueError, match="10 MiB"):
        await provider.get_audio_transcriptions(request)


@pytest.mark.asyncio
async def test_mimo_asr_rejects_invalid_language_for_model_alias() -> None:
    provider = XiaomiMimoProvider(options=ProviderRuntimeOptions())

    with pytest.raises(ValueError, match="auto/zh/en"):
        await provider.get_audio_transcriptions(_audio_request("relay-mimo-audio", extra_params={"language": "ja"}))
