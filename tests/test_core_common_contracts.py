import base64
import io
import logging

import pytest
from PIL import Image

from src.core.common import (
    ImageProcessingLimits,
    InvalidImagePolicy,
    ProviderRuntimeOptions,
    build_audio_file,
    build_openai_compatible_client_config,
    build_usage,
    build_usage_from_snapshot,
    image_data_url,
    image_media_type,
    merge_extra_params,
    message_text,
    normalize_auth_type,
    normalize_base_url,
    pop_json_object,
    read_api_key,
    read_model_identifier,
    read_timeout,
    require_string_dict,
    resolve_max_retries,
    resolve_retry_interval,
    split_request_overrides,
    with_default_user_agent,
)
from src.schemas import (
    AudioTranscriptionRequestSnapshot,
    GenericUsageSnapshot,
    MessagePartImage,
    MessageSnapshot,
    ModelInfoSnapshot,
    ObjectFields,
    ResponseRequestSnapshot,
)
from src.schemas.provider_contracts import ProviderUsage

from .support.http import make_api_provider


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        pytest.param("example.com/v1/", "https://example.com/v1", id="add-https"),
        pytest.param(" https://example.com/v1/ ", "https://example.com/v1", id="trim"),
        pytest.param("http://localhost:8080", "http://localhost:8080", id="http"),
    ],
)
def test_normalize_base_url_accepts_host_urls_and_removes_trailing_slash(
    raw_url: str,
    expected: str,
) -> None:
    assert normalize_base_url(raw_url) == expected


@pytest.mark.parametrize("raw_url", [None, "", "  ", "https://"])
def test_normalize_base_url_rejects_missing_or_unsupported_urls(raw_url: str | None) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(raw_url)


@pytest.mark.parametrize(
    ("auth_type", "api_key", "expected_client_key", "expected_headers", "expected_query"),
    [
        pytest.param("bearer", "secret", "secret", {}, {}, id="bearer-default"),
        pytest.param(
            "bearer",
            "secret",
            "",
            {"X-Token": "Token secret"},
            {},
            id="bearer-custom-header",
        ),
        pytest.param("header", "secret", "", {"X-Key": "secret"}, {}, id="header"),
        pytest.param("query", "secret", "", {}, {"key": "secret"}, id="query"),
        pytest.param("none", "", "", {}, {}, id="none"),
    ],
)
def test_openai_compatible_client_config_covers_auth_modes(
    auth_type: str,
    api_key: str,
    expected_client_key: str,
    expected_headers: dict[str, str],
    expected_query: dict[str, str],
) -> None:
    provider = make_api_provider(
        auth_type=auth_type,
        api_key=api_key,
        auth_header_name=(
            "Authorization"
            if auth_type == "bearer" and expected_client_key
            else "X-Token"
            if auth_type == "bearer"
            else "X-Key"
        ),
        auth_header_prefix=(
            "Bearer" if auth_type == "bearer" and expected_client_key else "Token" if auth_type == "bearer" else ""
        ),
        auth_query_name="key",
    )

    config = build_openai_compatible_client_config(provider)

    assert config.api_key == expected_client_key
    assert {key: value for key, value in config.default_headers.items() if key != "User-Agent"} == expected_headers
    assert config.default_query == expected_query


def test_openai_compatible_client_config_preserves_defaults_and_requires_key() -> None:
    provider = make_api_provider(
        default_headers={"X-Default": "yes"},
        default_query={"version": "1"},
    )
    config = build_openai_compatible_client_config(provider, user_agent="Custom/1")
    assert config.default_headers == {"X-Default": "yes", "User-Agent": "Custom/1"}
    assert config.default_query == {"version": "1"}

    with pytest.raises(ValueError):
        build_openai_compatible_client_config(make_api_provider(api_key=""))


@pytest.mark.parametrize(
    ("raw_auth_type", "expected"),
    [
        pytest.param(None, "bearer", id="default"),
        pytest.param(" HEADER ", "header", id="trim-lower"),
        pytest.param("none", "none", id="none"),
    ],
)
def test_normalize_auth_type(raw_auth_type: str | None, expected: str) -> None:
    assert normalize_auth_type(raw_auth_type) == expected


def test_normalize_auth_type_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        normalize_auth_type("cookie")


def test_model_key_timeout_and_string_mapping_boundaries() -> None:
    assert read_model_identifier(ModelInfoSnapshot(model_identifier="  model-a  ")) == "model-a"
    assert read_model_identifier(ModelInfoSnapshot(name="fallback")) == "fallback"
    with pytest.raises(ValueError):
        read_model_identifier(ModelInfoSnapshot())

    provider = make_api_provider(api_key="  key  ", timeout=3)
    assert read_api_key(provider) == "key"
    assert read_timeout(provider) == 3.0
    assert read_timeout(make_api_provider(timeout=0)) is None
    with pytest.raises(ValueError):
        read_api_key(make_api_provider(api_key=""))
    assert require_string_dict({"ok": "value"}, field_name="headers") == {"ok": "value"}
    with pytest.raises(TypeError):
        require_string_dict({"bad": 1}, field_name="headers")


@pytest.mark.parametrize(
    ("force", "host", "config", "expected"),
    [
        pytest.param(False, 2, 5, 2, id="host"),
        pytest.param(False, None, 5, 5, id="config"),
        pytest.param(True, 2, 5, 5, id="force"),
        pytest.param(False, -1, -2, 3, id="defaults"),
    ],
)
def test_retry_resolution_obeys_force_and_fallback_precedence(
    force: bool,
    host: int | None,
    config: int,
    expected: int,
) -> None:
    provider = make_api_provider(max_retry=host, retry_interval=host)
    assert resolve_max_retries(provider, config_value=config, force=force, default=3) == expected
    assert resolve_retry_interval(
        provider,
        config_value=float(config),
        force=force,
        default=3.0,
    ) == float(expected if expected != 3 or config == 3 else 3)


def test_extra_params_merge_split_and_pop_boundaries() -> None:
    request = ResponseRequestSnapshot(
        model_info=ModelInfoSnapshot(extra_params=ObjectFields(fields={"shared": "model", "none": None, "model": 1})),
        extra_params=ObjectFields(fields={"shared": "request", "request": True}),
    )
    assert merge_extra_params(request) == {"shared": "request", "model": 1, "request": True}

    overrides = split_request_overrides(
        {
            "headers": {"X-Test": "yes"},
            "query": {"version": 1},
            "body": {"nested": True},
            "temperature": 0.5,
            "reserved": "ignored",
        },
        direct_body_keys={"temperature"},
        reserved_body_keys={"reserved"},
    )
    assert overrides.extra_headers == {"X-Test": "yes"}
    assert overrides.extra_query == {"version": 1}
    assert overrides.extra_body == {"nested": True}
    assert overrides.direct_params == {"temperature": 0.5}
    assert pop_json_object({}, "missing") == {}


def test_message_and_media_helpers() -> None:
    message = MessageSnapshot.model_validate(
        {
            "parts": [
                {"type": "text", "text": "hello"},
                {"type": "image", "image_base64": "ignored"},
            ]
        }
    )
    assert message_text(message) == "hello"
    assert image_media_type(".jpg") == "image/jpeg"
    assert image_media_type("unknown") == "image/png"
    assert with_default_user_agent({}, "  Custom/2  ") == {"User-Agent": "Custom/2"}


def _encoded_image(image_format: str, *, frames: int = 1) -> str:
    images = [Image.new("RGB", (2, 2), (index * 50, 20, 20)) for index in range(frames)]
    output = io.BytesIO()
    images[0].save(
        output,
        format=image_format,
        save_all=frames > 1,
        append_images=images[1:],
        duration=50,
        loop=0,
    )
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    ("image_format", "expected_format"),
    [
        pytest.param("PNG", "png", id="png"),
        pytest.param("BMP", "png", id="bmp-converts"),
        pytest.param("GIF", "webp", id="gif-converts"),
    ],
)
def test_image_data_url_converts_supported_and_static_formats(
    image_format: str,
    expected_format: str,
) -> None:
    encoded = _encoded_image(image_format, frames=2 if image_format == "GIF" else 1)
    part = MessagePartImage(image_base64=encoded, image_format=image_format.lower())

    result = image_data_url(part, logging.getLogger("maidock-test"), "error")

    assert result is not None
    assert result.startswith(f"data:image/{expected_format};base64,")


@pytest.mark.parametrize("policy", ["placeholder", "skip"])
def test_image_data_url_invalid_policy_returns_none(policy: InvalidImagePolicy) -> None:
    part = MessagePartImage(image_base64="not-base64")
    assert image_data_url(part, logging.getLogger("maidock-test"), policy) is None


def test_image_limits_and_error_policy_are_enforced() -> None:
    part = MessagePartImage(image_base64=_encoded_image("PNG"))
    limits = ImageProcessingLimits(max_base64_chars=1)
    assert image_data_url(part, logging.getLogger("maidock-test"), "skip", limits) is None
    with pytest.raises(ValueError):
        image_data_url(MessagePartImage(image_base64="invalid"), logging.getLogger("maidock-test"), "error")


def test_usage_builder_covers_explicit_derived_and_cache_tokens() -> None:
    assert build_usage(prompt_tokens=2, completion_tokens=3).total_tokens == 5
    explicit = build_usage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=99,
        prompt_cache_hit_tokens=1,
        prompt_cache_miss_tokens=1,
    )
    assert explicit == ProviderUsage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=99,
        prompt_cache_hit_tokens=1,
        prompt_cache_miss_tokens=1,
    )

    usage = GenericUsageSnapshot.model_validate(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 3},
        }
    )
    assert build_usage_from_snapshot(usage) == ProviderUsage(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_cache_hit_tokens=3,
        prompt_cache_miss_tokens=7,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param("aGVsbG8=", b"hello", id="valid"),
        pytest.param("", None, id="missing"),
        pytest.param("invalid", None, id="invalid"),
    ],
)
def test_build_audio_file(payload: str, expected: bytes | None) -> None:
    request = AudioTranscriptionRequestSnapshot(audio_base64=payload)
    if expected is None:
        with pytest.raises(ValueError):
            build_audio_file(request)
        return
    filename, file_object = build_audio_file(request)
    assert filename == "audio.wav"
    assert file_object.read() == expected


def test_payload_summary_logging_obeys_options(caplog: pytest.LogCaptureFixture) -> None:
    from src.core.common import log_request_summary, log_response_summary

    logger = logging.getLogger("maidock-summary")
    usage = ProviderUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    with caplog.at_level(logging.DEBUG, logger="maidock-summary"):
        log_request_summary(
            logger,
            provider_label="test",
            model="model",
            messages=1,
            tools=0,
            extra={"secret": "value"},
            options=ProviderRuntimeOptions(log_payload_debug=True),
        )
        log_response_summary(
            logger,
            provider_label="test",
            content="ok",
            tool_calls=[],
            usage=usage,
            options=ProviderRuntimeOptions(),
        )
    assert "test" in caplog.text

    caplog.clear()
    log_request_summary(
        logger,
        provider_label="test",
        model="model",
        options=ProviderRuntimeOptions(log_payload_summary=False),
    )
    assert caplog.text == ""
