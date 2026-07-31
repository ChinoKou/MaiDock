from src.schemas import (
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    MessagePartImage,
    MessagePartText,
    MessagePartUnknown,
    ResponseFormatSchemaSnapshot,
    ResponseRequestSnapshot,
    ToolOptionSnapshot,
)
from tests.support.assertions import as_json_list, as_json_object
from tests.support.core_payloads import (
    build_audio_transcription_payload,
    build_embedding_payload,
    build_response_payload,
    load_provider_result_payload,
)


def test_response_fixture_preserves_complete_core_request_contract() -> None:
    payload = build_response_payload()
    request = ResponseRequestSnapshot.model_validate(payload)

    assert request.request_kind == "response"
    assert request.model_info.model_identifier == "contract-response-model"
    assert request.model_info.name == "脱敏响应模型"
    assert request.model_info.api_provider == "contract-provider"
    assert request.model_info.temperature == 0.25
    assert request.model_info.max_tokens == 512
    assert request.model_info.force_stream_mode is True
    assert request.model_info.visual is True
    assert "extra_params" not in request.model_info.model_dump(mode="python")

    api_provider = request.api_provider
    assert api_provider.name == "contract-provider"
    assert api_provider.api_key == "<redacted>"
    assert api_provider.base_url == "https://gateway.example/v1"
    assert api_provider.client_type == "openai_responses"
    assert api_provider.auth_type == "header"
    assert api_provider.auth_header_name == "X-API-Key"
    assert api_provider.auth_header_prefix == ""
    assert api_provider.auth_query_name == "key"
    assert api_provider.default_headers.to_plain_dict() == {"X-Contract": "snapshot"}
    assert api_provider.default_query.to_plain_dict() == {"api-version": "2026-01-01"}
    assert api_provider.organization == "org-redacted"
    assert api_provider.project == "project-redacted"
    assert api_provider.model_list_endpoint == "/models"
    assert api_provider.reasoning_parse_mode == "strict"
    assert api_provider.tool_argument_parse_mode == "repair"
    assert api_provider.timeout == 42.5
    assert api_provider.max_retry == 2
    assert api_provider.retry_interval == 1

    assert request.temperature == 0.4
    assert request.max_tokens == 1024
    assert "extra_params" not in request.model_dump(mode="python")
    assert "future_request_field" not in request.model_fields_set


def test_response_fixture_preserves_messages_tools_and_response_format() -> None:
    request = ResponseRequestSnapshot.model_validate(build_response_payload())

    assert [message.role for message in request.message_list] == ["system", "user", "assistant", "tool"]
    assert isinstance(request.message_list[0].parts[0], MessagePartText)
    assert request.message_list[0].parts[0].text == "脱敏系统消息"
    assert isinstance(request.message_list[1].parts[1], MessagePartImage)
    assert request.message_list[1].parts[1].image_base64 == "aW1hZ2U="
    assert request.message_list[1].parts[1].image_format == "png"
    assert isinstance(request.message_list[2].parts[0], MessagePartUnknown)
    assert request.message_list[2].parts[0].type == "future_part"

    tool_call = request.message_list[2].tool_calls[0]
    assert tool_call.resolved_call_id() == "tool-id-wins"
    assert tool_call.function.name == "lookup_weather"
    assert tool_call.function.arguments is not None
    assert not isinstance(tool_call.function.arguments, str)
    assert tool_call.function.arguments.to_plain_dict() == {"city": "上海"}
    assert tool_call.extra_content.to_plain_dict() == {"trace": "redacted"}
    assert request.message_list[3].tool_call_id == "tool-id-wins"
    assert request.message_list[3].tool_name == "lookup_weather"

    nested = request.tool_options[0].function_definition()
    flat = request.tool_options[1].function_definition()
    assert nested.name == "lookup_weather"
    assert nested.parameters.to_plain_dict()["required"] == ["city"]
    assert flat.name == "read_clock"
    assert flat.parameters.to_plain_dict() == {"type": "object", "properties": {}}

    assert request.response_format is not None
    assert request.response_format.format_type == "json_schema"
    assert isinstance(request.response_format.schema_, ResponseFormatSchemaSnapshot)
    assert request.response_format.schema_.name == "weather_result"
    assert request.response_format.schema_.strict is True
    assert request.response_format.schema_.schema_ is not None
    assert request.response_format.schema_.schema_.to_plain_dict()["required"] == ["summary"]


def test_embedding_fixture_ignores_extra_params() -> None:
    payload = build_embedding_payload()
    request = EmbeddingRequestSnapshot.model_validate(payload)

    assert request.request_kind == "embedding"
    assert request.embedding_input == "脱敏向量文本"
    assert request.dimensions is None
    assert "dimensions" not in payload
    assert "extra_params" not in request.model_info.model_dump(mode="python")
    assert "extra_params" not in request.model_dump(mode="python")
    assert request.api_provider.auth_type == "bearer"
    assert request.api_provider.retry_interval == 0


def test_audio_fixture_preserves_audio_and_provider_contract() -> None:
    request = AudioTranscriptionRequestSnapshot.model_validate(build_audio_transcription_payload())

    assert request.request_kind == "audio_transcription"
    assert request.audio_base64 == "UklGRgAAAABXQVZF"
    assert request.max_tokens == 96
    assert request.model_info.model_identifier == "contract-audio-model"
    assert "extra_params" not in request.model_info.model_dump(mode="python")
    assert request.api_provider.auth_type == "query"
    assert request.api_provider.auth_query_name == "token"
    assert request.api_provider.default_query.to_plain_dict() == {"locale": "zh-CN"}
    assert "extra_params" not in request.model_dump(mode="python")


def test_contract_metadata_records_sources_version_and_sanitization() -> None:
    for payload in (
        build_response_payload(),
        build_embedding_payload(),
        build_audio_transcription_payload(),
        load_provider_result_payload(),
    ):
        contract = as_json_object(payload["_contract"])
        sources = as_json_list(contract["source_symbols"])
        assert contract["version"] == 1
        assert contract["sanitized"] is True
        assert len(sources) >= 3
        assert all(isinstance(source, str) and source.startswith("src.") for source in sources)


def test_payload_constructors_return_isolated_mutable_copies() -> None:
    first = build_response_payload()
    first_model = as_json_object(first["model_info"])
    first_model["name"] = "mutated"

    second_model = as_json_object(build_response_payload()["model_info"])
    assert second_model["name"] == "脱敏响应模型"


def test_payload_constructor_applies_typed_top_level_overrides() -> None:
    payload = build_response_payload(message_list=[], max_tokens=64, response_format=None)
    request = ResponseRequestSnapshot.model_validate(payload)

    assert request.message_list == []
    assert request.max_tokens == 64
    assert request.response_format is None


def test_flat_tool_helper_uses_default_schema_for_missing_parameters() -> None:
    option = ToolOptionSnapshot.model_validate({"name": "no_parameters"})

    assert option.function_definition().parameters.to_plain_dict() == {
        "type": "object",
        "properties": {},
    }
