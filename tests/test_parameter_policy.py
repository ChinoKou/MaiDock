import pytest

from src.config import MaiDockConfig, build_parameter_overrides, build_runtime_options, parse_override_value
from src.core.parameter_catalog import get_parameter_catalog
from src.core.parameter_policy import ParameterOverrideRegistry, ParameterOverrideSet
from src.host_adapters.common.parameter_translation import build_normalized_host_parameters
from src.schemas import ResponseRequestSnapshot


def _request(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model_extra_params: dict | None = None,
    request_extra_params: dict | None = None,
) -> ResponseRequestSnapshot:
    return ResponseRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "test-model",
                "extra_params": model_extra_params or {},
            },
            "api_provider": {"api_key": "test-key"},
            "extra_params": request_extra_params or {},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )


def test_override_set_keeps_only_typed_values() -> None:
    overrides = ParameterOverrideSet(values={"top_p": 0.5, "store": False, "include": ["a", "b"]})
    assert bool(overrides)
    assert "top_p" in overrides
    assert "missing" not in overrides


def test_parse_override_value_types() -> None:
    assert parse_override_value(raw_value="", value_kind="string", field_label="f") is None
    assert parse_override_value(raw_value="  ", value_kind="json", field_label="f") is None
    assert parse_override_value(raw_value="auto", value_kind="string", field_label="f") == "auto"
    assert parse_override_value(raw_value="0.8", value_kind="number", field_label="f") == 0.8
    assert parse_override_value(raw_value="1024", value_kind="integer", field_label="f") == 1024
    assert parse_override_value(raw_value="true", value_kind="boolean", field_label="f") is True
    assert parse_override_value(raw_value="false", value_kind="boolean", field_label="f") is False
    assert parse_override_value(raw_value='{"type":"disabled"}', value_kind="json", field_label="f") == {
        "type": "disabled"
    }
    assert parse_override_value(raw_value='["a","b"]', value_kind="string_list", field_label="f") == ["a", "b"]


def test_parse_override_value_rejects_wrong_types() -> None:
    with pytest.raises(ValueError, match="整数"):
        parse_override_value(raw_value="0.5", value_kind="integer", field_label="f")
    with pytest.raises(ValueError, match="布尔"):
        parse_override_value(raw_value="0.5", value_kind="boolean", field_label="f")
    with pytest.raises(ValueError, match="数值"):
        parse_override_value(raw_value="true", value_kind="number", field_label="f")
    with pytest.raises(ValueError, match="字符串数组"):
        parse_override_value(raw_value="[1,2]", value_kind="string_list", field_label="f")
    with pytest.raises(ValueError, match="JSON"):
        parse_override_value(raw_value="{broken", value_kind="json", field_label="f")
    # NaN/Infinity 不是标准 JSON，拒绝而非生成非法请求体。
    with pytest.raises(ValueError, match="NaN"):
        parse_override_value(raw_value="NaN", value_kind="number", field_label="f")
    with pytest.raises(ValueError, match="Infinity"):
        parse_override_value(raw_value="Infinity", value_kind="number", field_label="f")
    # 1e999 会解析为 inf 而不触发 parse_constant，必须在数值校验中拒绝。
    with pytest.raises(ValueError, match="有限"):
        parse_override_value(raw_value="1e999", value_kind="number", field_label="f")
    with pytest.raises(ValueError, match="有限"):
        parse_override_value(raw_value='{"x": 1e999}', value_kind="json", field_label="f")


def test_build_parameter_overrides_rejects_unknown_catalog_fields() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    config = MaiDockConfig.model_validate(
        {
            "openai_responses": {
                "response": {
                    "overrides": {
                        "temperature": "0.4",
                        "store": "true",
                        "unknown_future_key": "1",
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match=r"openai_responses.*response.*unknown_future_key"):
        build_parameter_overrides(config.openai_responses.response, catalog)


def test_normalized_host_parameters_ignore_extra_params() -> None:
    """两级 extra_params 即使含冲突或非法值也完全无效。"""

    request = _request(
        temperature=0.7,
        max_tokens=128,
        model_extra_params={"temperature": 0.1, "top_p": "not-a-number"},
        request_extra_params={"temperature": 999, "body": {"top_p": [1, 2]}},
    )
    catalog = get_parameter_catalog("openai_responses", "response")
    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )
    assert normalized.fields == {"temperature": 0.7, "max_tokens": 128}
    assert normalized.sources["temperature"] == "request.temperature"


def test_normalized_host_parameters_none_fields_are_not_injected() -> None:
    request = _request(temperature=None, max_tokens=None)
    catalog = get_parameter_catalog("openai_responses", "response")
    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )
    assert normalized.fields == {}


def test_overrides_override_host_typed_fields() -> None:
    request = _request(temperature=0.7, max_tokens=128)
    catalog = get_parameter_catalog("openai_responses", "response")
    normalized = build_normalized_host_parameters(
        request,
        overrides=ParameterOverrideSet(values={"temperature": 0.2, "max_tokens": 512}),
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )
    # 覆写值在 normalized.fields 中排在 Host 类型字段之后（后写覆盖）。
    assert normalized.fields["temperature"] == 0.2
    assert normalized.fields["max_tokens"] == 512
    assert normalized.sources["temperature"] == "overrides.temperature"


def test_runtime_options_build_override_registry() -> None:
    config = MaiDockConfig.model_validate(
        {"dashscope": {"chat_completion": {"overrides": {"result_format": "message", "enable_search": "true"}}}}
    )
    options = build_runtime_options(config)
    registry = options.parameter_overrides
    assert isinstance(registry, ParameterOverrideRegistry)
    dashscope_chat = registry.get("dashscope", "chat_completion")
    assert dashscope_chat.values == {"result_format": "message", "enable_search": True}
    # 未配置的能力保持空覆写。
    assert registry.get("openai_responses", "response").values == {}
