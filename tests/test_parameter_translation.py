from src.core.parameter_catalog import get_parameter_catalog
from src.core.parameter_policy import ParameterOverrideSet
from src.host_adapters.common.parameter_translation import (
    NormalizedHostParameters,
    TranslationContext,
    TranslationEnvelope,
    build_normalized_host_parameters,
    get_target_value,
    has_target_value,
    pop_target_value,
    run_translators,
    set_target_value,
)
from src.schemas.host_snapshots import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    ResponseRequestSnapshot,
)


def _response_request(
    *,
    request_temperature: int | float | None = None,
    request_max_tokens: int | None = None,
    response_format: dict | None = None,
) -> ResponseRequestSnapshot:
    payload: dict = {
        "model_info": {"model_identifier": "test-model"},
        "api_provider": {"api_key": "test-key"},
    }
    if request_temperature is not None:
        payload["temperature"] = request_temperature
    if request_max_tokens is not None:
        payload["max_tokens"] = request_max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    return ResponseRequestSnapshot.model_validate(payload)


def _embedding_request(*, dimensions: int | None = None) -> EmbeddingRequestSnapshot:
    payload: dict = {
        "model_info": {"model_identifier": "embedding-model"},
        "api_provider": {"api_key": "test-key"},
        "embedding_input": "hello",
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return EmbeddingRequestSnapshot.model_validate(payload)


def _audio_request(*, max_tokens: int | None = None) -> AudioTranscriptionRequestSnapshot:
    payload: dict = {
        "model_info": {"model_identifier": "audio-model"},
        "api_provider": {"api_key": "test-key"},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return AudioTranscriptionRequestSnapshot.model_validate(payload)


def test_build_normalized_host_parameters_collects_typed_fields() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(request_temperature=0.2, request_max_tokens=64)

    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )

    assert normalized.fields["temperature"] == 0.2
    assert normalized.fields["max_tokens"] == 64
    assert normalized.sources["temperature"] == "request.temperature"
    assert normalized.sources["max_tokens"] == "request.max_tokens"


def test_build_normalized_host_parameters_does_not_read_model_fallback() -> None:
    """请求字段为 None 时不再读取 model_info 补值。"""

    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")
    request = ResponseRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "test-model",
                "temperature": 0.4,
                "max_tokens": 512,
            },
            "api_provider": {"api_key": "test-key"},
        }
    )

    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(),
        catalog=catalog,
        provider_label="Anthropic Messages",
        capability="chat_completion",
    )

    assert normalized.fields == {}


def test_build_normalized_host_parameters_merges_overrides_after_typed_fields() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(request_temperature=0.2, request_max_tokens=64)

    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(values={"temperature": 0.9, "store": False}),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )

    assert normalized.fields["temperature"] == 0.9
    assert normalized.fields["max_tokens"] == 64
    assert normalized.fields["store"] is False
    assert normalized.sources["temperature"] == "overrides.temperature"


def test_build_normalized_host_parameters_embedding_dimensions() -> None:
    catalog = get_parameter_catalog("dashscope", "embeddings")
    request = _embedding_request(dimensions=1024)

    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(values={"dimensions": 2048}),
        catalog=catalog,
        provider_label="DashScope",
        capability="embeddings",
    )

    assert normalized.fields["dimensions"] == 2048


def test_build_normalized_host_parameters_audio_max_tokens() -> None:
    catalog = get_parameter_catalog("volcengine_ark", "audio_transcription")
    request = _audio_request(max_tokens=128)

    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(),
        catalog=catalog,
        provider_label="Volcengine Ark",
        capability="audio_transcription",
    )

    assert normalized.fields["max_tokens"] == 128


def test_target_path_helpers_round_trip_nested_values() -> None:
    envelope = TranslationEnvelope()

    set_target_value(envelope, ("body", "parameters", "temperature"), 0.3)
    set_target_value(envelope, ("headers", "X-Test"), "1")
    set_target_value(envelope, ("query", "trace"), True)

    assert has_target_value(envelope, ("body", "parameters", "temperature")) is True
    assert get_target_value(envelope, ("body", "parameters", "temperature")) == 0.3
    assert get_target_value(envelope, ("headers", "X-Test")) == "1"
    assert pop_target_value(envelope, ("query", "trace")) is True
    assert has_target_value(envelope, ("query", "trace")) is False


def test_run_translators_writes_catalog_target_for_missing_translator() -> None:
    """目录字段没有专用 translator 时按目标路径直写。"""

    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request()
    context = TranslationContext(
        request=request,
        provider_label="OpenAI Responses",
        provider="openai_responses",
        capability="response",
        catalog=catalog,
        overrides=ParameterOverrideSet(),
        normalized=NormalizedHostParameters(fields={"top_p": 0.5, "store": True}),
        model="test-model",
    )
    envelope = TranslationEnvelope()

    run_translators(context, envelope, {"top_p": _set_top_p})

    assert envelope.body == {"top_p": 0.5, "store": True}


def _set_top_p(_context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
    set_target_value(envelope, ("body", "top_p"), value)


def test_run_translators_ignores_unknown_non_catalog_fields() -> None:
    """不在目录中的字段不会转发到请求体。"""

    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")
    request = _response_request()
    context = TranslationContext(
        request=request,
        provider_label="Anthropic Messages",
        provider="anthropic_messages",
        capability="chat_completion",
        catalog=catalog,
        overrides=ParameterOverrideSet(),
        normalized=NormalizedHostParameters(fields={"future_field": True}),
        model="test-model",
    )

    envelope = TranslationEnvelope()
    run_translators(context, envelope, {})
    assert envelope.body == {}


def test_run_translators_order_applies_overrides_last() -> None:
    """同一目标路径先 Host 类型字段后覆写值，后者覆盖实现叶级合并。"""

    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(response_format={"format_type": "json_object"})
    normalized = NormalizedHostParameters(
        fields={
            "response_format": request.response_format,
            "text": {"instructions": "keep"},
        }
    )
    context = TranslationContext(
        request=request,
        provider_label="OpenAI Responses",
        provider="openai_responses",
        capability="response",
        catalog=catalog,
        overrides=ParameterOverrideSet(),
        normalized=normalized,
        model="test-model",
    )

    from src.host_adapters.responses_family.parameter_translation import apply_responses_parameters

    envelope = TranslationEnvelope()
    apply_responses_parameters(context, envelope)

    # response_format 转译为 text.format；text 覆写对象只替换同名叶子，保留其他字段。
    assert envelope.body["text"] == {"instructions": "keep", "format": {"type": "json_object"}}
