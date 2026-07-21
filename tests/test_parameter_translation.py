import pytest

from src.core.parameter_catalog import get_parameter_catalog
from src.core.parameter_policy import ParameterPolicy
from src.providers.common.parameter_translation import (
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
from src.schemas.host_snapshots import EmbeddingRequestSnapshot, ResponseRequestSnapshot


def _response_request(
    *,
    model_extra_params: dict | None = None,
    request_extra_params: dict | None = None,
    request_temperature: int | float | None = None,
    request_max_tokens: int | None = None,
    model_temperature: int | float | None = None,
    model_max_tokens: int | None = None,
    response_format: dict | None = None,
) -> ResponseRequestSnapshot:
    model_info: dict = {
        "model_identifier": "test-model",
        "extra_params": model_extra_params or {},
    }
    if model_temperature is not None:
        model_info["temperature"] = model_temperature
    if model_max_tokens is not None:
        model_info["max_tokens"] = model_max_tokens
    payload: dict = {
        "model_info": model_info,
        "api_provider": {"api_key": "test-key"},
        "extra_params": request_extra_params or {},
    }
    if request_temperature is not None:
        payload["temperature"] = request_temperature
    if request_max_tokens is not None:
        payload["max_tokens"] = request_max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    return ResponseRequestSnapshot.model_validate(payload)


def _embedding_request(
    *,
    model_extra_params: dict | None = None,
    request_extra_params: dict | None = None,
    dimensions: int | None = None,
) -> EmbeddingRequestSnapshot:
    payload: dict = {
        "model_info": {
            "model_identifier": "embedding-model",
            "extra_params": model_extra_params or {},
        },
        "api_provider": {"api_key": "test-key"},
        "extra_params": request_extra_params or {},
        "embedding_input": "hello",
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return EmbeddingRequestSnapshot.model_validate(payload)


def test_build_normalized_host_parameters_prefers_request_typed_fields() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(
        model_extra_params={"top_p": 0.8},
        request_extra_params={"metadata": {"source": "request"}},
        request_temperature=0.2,
        request_max_tokens=64,
        model_temperature=0.9,
        model_max_tokens=256,
    )

    normalized = build_normalized_host_parameters(
        request,
        policy=ParameterPolicy(),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )

    assert normalized.fields["temperature"] == 0.2
    assert normalized.fields["max_tokens"] == 64
    assert normalized.fields["top_p"] == 0.8
    assert normalized.fields["metadata"] == {"source": "request"}
    assert normalized.sources["temperature"] == "request.temperature"
    assert normalized.sources["max_tokens"] == "request.max_tokens"


def test_build_normalized_host_parameters_uses_model_fallback_once() -> None:
    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")
    request = _response_request(model_temperature=0.4, model_max_tokens=512)

    normalized = build_normalized_host_parameters(
        request,
        policy=ParameterPolicy(),
        catalog=catalog,
        provider_label="Anthropic Messages",
        capability="chat_completion",
    )

    assert normalized.fields["temperature"] == 0.4
    assert normalized.fields["max_tokens"] == 512
    assert normalized.sources["temperature"] == "model_info.temperature"
    assert normalized.sources["max_tokens"] == "model_info.max_tokens"


def test_build_normalized_host_parameters_maps_source_aliases() -> None:
    catalog = get_parameter_catalog("dashscope", "embeddings")
    request = _embedding_request(request_extra_params={"dimension": 1024})

    normalized = build_normalized_host_parameters(
        request,
        policy=ParameterPolicy(),
        catalog=catalog,
        provider_label="DashScope",
        capability="embeddings",
    )

    assert normalized.fields == {"dimensions": 1024}
    assert normalized.sources["dimensions"] == "request.extra_params.dimension"
    assert catalog.field_by_safe_key("body_parameters_dimension") is not None


def test_build_normalized_host_parameters_rejects_conflicting_typed_and_alias_values() -> None:
    catalog = get_parameter_catalog("dashscope", "embeddings")
    request = _embedding_request(request_extra_params={"dimension": 1024}, dimensions=2048)

    with pytest.raises(ValueError, match="dimensions"):
        build_normalized_host_parameters(
            request,
            policy=ParameterPolicy(),
            catalog=catalog,
            provider_label="DashScope",
            capability="embeddings",
        )


def test_disabled_target_path_drops_matching_host_field() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(response_format={"format_type": "json_object"})

    normalized = build_normalized_host_parameters(
        request,
        policy=ParameterPolicy(disabled_paths=("body.text.format",)),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )

    assert "response_format" not in normalized.fields


def test_rejected_target_path_rejects_matching_host_field() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request(response_format={"format_type": "json_object"})

    with pytest.raises(ValueError, match="body.text.format"):
        build_normalized_host_parameters(
            request,
            policy=ParameterPolicy(rejected_paths=("body.text.format",)),
            catalog=catalog,
            provider_label="OpenAI Responses",
            capability="response",
        )


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


def test_run_translators_merges_transport_roots_and_forwards_unknown_fields() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    request = _response_request()
    normalized = NormalizedHostParameters(
        fields={
            "temperature": 0.3,
            "body": {"nested": True},
            "headers": {"X-Test": "1"},
            "future_field": {"enabled": True},
        }
    )
    context = TranslationContext(
        request=request,
        provider_label="OpenAI Responses",
        provider="openai_responses",
        capability="response",
        catalog=catalog,
        policy=ParameterPolicy(unknown_extra_params="forward"),
        normalized=normalized,
        model="test-model",
    )
    envelope = TranslationEnvelope()

    run_translators(
        context,
        envelope,
        {
            "temperature": lambda _context, current_envelope, value: set_target_value(
                current_envelope, ("body", "temperature"), value
            )
        },
    )

    assert envelope.body == {
        "nested": True,
        "temperature": 0.3,
        "future_field": {"enabled": True},
    }
    assert envelope.headers == {"X-Test": "1"}


def test_run_translators_rejects_unknown_fields_when_policy_requires() -> None:
    catalog = get_parameter_catalog("anthropic_messages", "chat_completion")
    request = _response_request()
    context = TranslationContext(
        request=request,
        provider_label="Anthropic Messages",
        provider="anthropic_messages",
        capability="chat_completion",
        catalog=catalog,
        policy=ParameterPolicy(unknown_extra_params="reject"),
        normalized=NormalizedHostParameters(fields={"future_field": True}),
        model="test-model",
    )

    with pytest.raises(ValueError, match="future_field"):
        run_translators(context, TranslationEnvelope(), {})
