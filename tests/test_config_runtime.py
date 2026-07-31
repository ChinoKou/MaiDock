import pytest
from pydantic import ValidationError

from src.config import (
    MaiDockConfig,
    build_parameter_overrides,
    build_runtime_options,
    normalize_maidock_config_data,
    normalize_user_agent,
)
from src.config_schema import build_maidock_config_schema
from src.core.parameter_catalog import get_parameter_catalog
from src.version import DEFAULT_USER_AGENT, __version__


def test_default_config_contains_provider_sections() -> None:
    config = MaiDockConfig()

    assert config.plugin.config_version == __version__
    assert config.openai_responses.user_agent == ""
    assert config.anthropic_messages.user_agent == ""
    assert config.volcengine_ark.user_agent == ""
    assert config.volcengine_ark.force_official_endpoint is True
    assert config.dashscope.user_agent == ""
    assert config.dashscope.force_official_endpoint is True
    assert config.siliconflow.user_agent == ""
    assert config.siliconflow.force_official_endpoint is True
    assert config.xiaomi_mimo.user_agent == ""

    assert config.xiaomi_mimo.reasoning_retention_days == 30
    assert config.openai_responses.response.overrides == {}
    assert config.volcengine_ark.embeddings.overrides == {}
    assert config.dashscope.chat_completion.overrides == {}
    assert config.siliconflow.embeddings.overrides == {}


def test_old_config_data_gets_provider_section_defaults() -> None:
    config = MaiDockConfig.model_validate(
        {
            "plugin": {"enabled": True, "config_version": "1.0.0"},
            "diagnostics": {"include_raw_data": True},
        }
    )

    assert config.openai_responses.user_agent == ""
    assert config.anthropic_messages.user_agent == ""
    assert config.volcengine_ark.user_agent == ""
    assert config.volcengine_ark.force_official_endpoint is True
    assert config.dashscope.user_agent == ""
    assert config.dashscope.force_official_endpoint is True
    assert config.siliconflow.user_agent == ""
    assert config.siliconflow.force_official_endpoint is True
    assert config.xiaomi_mimo.user_agent == ""

    assert config.xiaomi_mimo.reasoning_retention_days == 30
    assert config.openai_responses.response.overrides == {}
    assert config.dashscope.chat_completion.overrides == {}
    assert config.diagnostics.include_raw_data is True


def test_blank_provider_user_agents_use_default_runtime_user_agent() -> None:
    config = MaiDockConfig.model_validate(
        {
            "openai_responses": {"user_agent": "   "},
            "anthropic_messages": {"user_agent": ""},
            "volcengine_ark": {"user_agent": "\t"},
            "dashscope": {"user_agent": "  "},
            "siliconflow": {"user_agent": "  "},
            "xiaomi_mimo": {"user_agent": "  "},
        }
    )

    options = build_runtime_options(config)

    assert options.openai_user_agent == DEFAULT_USER_AGENT
    assert options.anthropic_user_agent == DEFAULT_USER_AGENT
    assert options.volcengine_user_agent == DEFAULT_USER_AGENT
    assert options.dashscope_user_agent == DEFAULT_USER_AGENT
    assert options.siliconflow_user_agent == DEFAULT_USER_AGENT
    assert options.mimo_user_agent == DEFAULT_USER_AGENT


def test_provider_user_agents_are_normalized_independently() -> None:
    config = MaiDockConfig.model_validate(
        {
            "openai_responses": {"user_agent": "  OpenAI-UA/1  "},
            "anthropic_messages": {"user_agent": "Anthropic-UA/1"},
            "volcengine_ark": {"user_agent": "  Ark-UA/1  "},
            "dashscope": {"user_agent": "DashScope-UA/1"},
            "siliconflow": {"user_agent": "  SiliconFlow-UA/1  "},
            "xiaomi_mimo": {"user_agent": "  Mimo-UA/1  "},
        }
    )

    options = build_runtime_options(config)

    assert options.openai_user_agent == "OpenAI-UA/1"
    assert options.anthropic_user_agent == "Anthropic-UA/1"
    assert options.volcengine_user_agent == "Ark-UA/1"
    assert options.dashscope_user_agent == "DashScope-UA/1"
    assert options.siliconflow_user_agent == "SiliconFlow-UA/1"
    assert options.mimo_user_agent == "Mimo-UA/1"


def test_force_official_endpoint_flags_map_independently_to_runtime_options() -> None:
    config = MaiDockConfig.model_validate(
        {
            "volcengine_ark": {"force_official_endpoint": True},
            "dashscope": {"force_official_endpoint": False},
            "siliconflow": {"force_official_endpoint": True},
        }
    )

    options = build_runtime_options(config)

    assert options.volcengine_force_official_endpoint is True
    assert options.dashscope_force_official_endpoint is False
    assert options.siliconflow_force_official_endpoint is True


def test_runtime_options_default_to_force_official_endpoints() -> None:
    options = build_runtime_options()

    assert options.volcengine_force_official_endpoint is True
    assert options.dashscope_force_official_endpoint is True
    assert options.siliconflow_force_official_endpoint is True
    assert options.volcengine_builtin_endpoint_mode == "standard"


def test_ark_builtin_endpoint_mode_defaults_and_maps_to_runtime_options() -> None:
    assert MaiDockConfig().volcengine_ark.builtin_endpoint_mode == "standard"

    config = MaiDockConfig.model_validate({"volcengine_ark": {"builtin_endpoint_mode": " Agent_Plan "}})
    assert config.volcengine_ark.builtin_endpoint_mode == "agent_plan"

    options = build_runtime_options(config)
    assert options.volcengine_builtin_endpoint_mode == "agent_plan"

    coding = build_runtime_options(
        MaiDockConfig.model_validate({"volcengine_ark": {"builtin_endpoint_mode": "coding_plan"}})
    )
    assert coding.volcengine_builtin_endpoint_mode == "coding_plan"


def test_ark_builtin_endpoint_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"volcengine_ark": {"builtin_endpoint_mode": "premium"}})


def test_normalize_user_agent_uses_default_for_empty_values() -> None:
    assert normalize_user_agent(None) == DEFAULT_USER_AGENT
    assert normalize_user_agent("") == DEFAULT_USER_AGENT
    assert normalize_user_agent("  ") == DEFAULT_USER_AGENT
    assert normalize_user_agent("  Custom-UA  ") == "Custom-UA"


def test_parameter_overrides_config_maps_to_runtime_options() -> None:
    config = MaiDockConfig.model_validate(
        {
            "dashscope": {
                "chat_completion": {
                    "overrides": {
                        "top_p": "0.4",
                        "enable_search": "true",
                        "result_format": "message",
                    }
                }
            }
        }
    )

    overrides = build_runtime_options(config).parameter_overrides.get("dashscope", "chat_completion")

    assert overrides.values == {
        "top_p": 0.4,
        "enable_search": True,
        "result_format": "message",
    }


def test_generated_override_defaults_map_to_runtime_options() -> None:
    config = MaiDockConfig.model_validate({"dashscope": {"chat_completion": {"overrides": {"top_p": "0.6"}}}})

    overrides = build_runtime_options(config).parameter_overrides.get("dashscope", "chat_completion")

    assert overrides.values == {"top_p": 0.6}


def test_embedding_catalogs_expose_provider_target_paths() -> None:
    chat_catalog = get_parameter_catalog("dashscope", "chat_completion")
    dashscope_embeddings = get_parameter_catalog("dashscope", "embeddings")
    siliconflow_embeddings = get_parameter_catalog("siliconflow", "embeddings")
    ark_embeddings = get_parameter_catalog("volcengine_ark", "embeddings")

    response_format = chat_catalog.field_by_safe_key("response_format")
    dashscope_dimensions = dashscope_embeddings.field_by_safe_key("dimensions")
    siliconflow_dimensions = siliconflow_embeddings.field_by_safe_key("dimensions")
    ark_dimensions = ark_embeddings.field_by_safe_key("dimensions")

    assert response_format is not None
    assert response_format.target_path == ("body", "parameters", "response_format")
    assert dashscope_dimensions is not None
    assert dashscope_dimensions.target_path == ("body", "parameters", "dimension")
    assert dashscope_dimensions.config_key == "dimensions"
    assert siliconflow_dimensions is not None
    assert siliconflow_dimensions.target_path == ("body", "dimensions")
    assert ark_dimensions is not None
    assert ark_dimensions.target_path == ("body", "dimensions")


def test_maidock_config_normalization_fills_override_defaults() -> None:
    normalized, changed = normalize_maidock_config_data(MaiDockConfig().model_dump(mode="python"))

    dashscope_chat = normalized["dashscope"]["chat_completion"]

    assert changed is True
    assert dashscope_chat["overrides"]["result_format"] == "message"
    assert "fields" not in dashscope_chat
    assert "temperature" not in dashscope_chat["overrides"]


def test_maidock_config_normalization_migrates_legacy_field_controls() -> None:
    normalized, changed = normalize_maidock_config_data(
        {
            "plugin": {"enabled": True, "config_version": __version__},
            "dashscope": {
                "chat_completion": {
                    "fields": {
                        "body_parameters_enable_thinking_enabled": False,
                        "body_parameters_top_p_override_enabled": True,
                        "body_parameters_top_p_override_value": "0.6",
                        "body_parameters_top_k_override_enabled": False,
                        "body_parameters_top_k_override_value": "10",
                    }
                }
            },
        }
    )

    chat_completion = normalized["dashscope"]["chat_completion"]

    assert changed is True
    # 只迁移启用状态的覆写；关闭状态遗留值不得生效。
    assert chat_completion["overrides"]["top_p"] == "0.6"
    assert "top_k" not in chat_completion["overrides"]
    assert "fields" not in chat_completion


def test_maidock_config_normalization_is_idempotent() -> None:
    normalized, _ = normalize_maidock_config_data(MaiDockConfig().model_dump(mode="python"))
    second, changed = normalize_maidock_config_data(normalized)
    assert changed is False
    assert second == normalized


def test_maidock_webui_schema_uses_dotted_scalar_sections() -> None:
    schema = build_maidock_config_schema(plugin_id="maidock")
    sections = schema["sections"]
    response_overrides = sections["openai_responses_response_overrides"]
    dashscope_overrides = sections["dashscope_chat_completion_overrides"]

    assert response_overrides["name"] == "openai_responses.response.overrides"
    assert dashscope_overrides["name"] == "dashscope.chat_completion.overrides"
    dashscope_embedding_overrides = sections["dashscope_embeddings_overrides"]["fields"]
    siliconflow_embedding_overrides = sections["siliconflow_embeddings_overrides"]["fields"]
    ark_embedding_overrides = sections["volcengine_ark_embeddings_overrides"]["fields"]

    # 每个参数只有一个全宽 textarea 覆写框。
    assert response_overrides["fields"]["top_p"]["ui_type"] == "textarea"
    assert response_overrides["fields"]["store"]["ui_type"] == "textarea"
    top_p = response_overrides["fields"]["top_p"]
    store = response_overrides["fields"]["store"]
    assert top_p["label"] == "top_p · number"
    assert "body.top_p" in top_p["hint"]
    assert "0..1" in top_p["hint"]
    assert "https://platform.openai.com/docs/api-reference/responses/create" in top_p["hint"]
    assert store["label"] == "store · boolean"
    assert "false" in store["hint"]
    assert "dimensions" in dashscope_embedding_overrides
    assert "dimensions" in siliconflow_embedding_overrides
    assert "dimensions" in ark_embedding_overrides
    assert "_enabled" not in response_overrides["fields"]
    for section in sections.values():
        for field in section["fields"].values():
            assert field["ui_type"] != "json"
            assert field["type"] != "object"


def test_parameter_overrides_config_normalizes_scalar_values() -> None:
    """配置中的布尔/数字覆写值会被字符串化后进入统一字符串目录。"""

    config = MaiDockConfig.model_validate(
        {"openai_responses": {"response": {"overrides": {"top_p": 0.5, "store": True}}}}
    )
    assert config.openai_responses.response.overrides == {"top_p": "0.5", "store": "true"}

    with pytest.raises(TypeError):
        MaiDockConfig.model_validate({"openai_responses": {"response": {"overrides": {"top_p": ["0.5"]}}}})


def test_config_normalization_rejects_unknown_override_key() -> None:
    with pytest.raises(ValueError, match=r"openai_responses.*response.*temprature"):
        normalize_maidock_config_data({"openai_responses": {"response": {"overrides": {"temprature": "0.5"}}}})


def test_audio_transcription_catalogs_expose_provider_target_paths() -> None:
    dashscope_audio = get_parameter_catalog("dashscope", "audio_transcription")
    siliconflow_audio = get_parameter_catalog("siliconflow", "audio_transcription")
    ark_audio = get_parameter_catalog("volcengine_ark", "audio_transcription")
    mimo_audio = get_parameter_catalog("xiaomi_mimo", "audio_transcription")

    assert dashscope_audio.title == "阿里云百炼 DashScope 语音转录参数"
    assert dashscope_audio.reserved_body_keys == frozenset({"input", "model", "parameters"})
    ds_language = dashscope_audio.field_by_safe_key("language")
    assert ds_language is not None
    assert ds_language.target_path == ("body", "parameters", "asr_options", "language")
    assert ds_language.value_kind == "string"
    ds_enable_itn = dashscope_audio.field_by_safe_key("enable_itn")
    assert ds_enable_itn is not None
    assert ds_enable_itn.target_path == (
        "body",
        "parameters",
        "asr_options",
        "enable_itn",
    )
    assert ds_enable_itn.value_kind == "boolean"
    assert ds_enable_itn.label == "逆文本正则化"
    ds_format = dashscope_audio.field_by_safe_key("format")
    ds_audio_format = dashscope_audio.field_by_safe_key("audio_format")
    assert ds_format is not None
    assert ds_format.target_path == ("body", "format")
    assert ds_audio_format is not None
    assert ds_audio_format.target_path == ("body", "audio_format")
    assert dashscope_audio.field_by_safe_key("result_format") is None

    assert siliconflow_audio.title == "SiliconFlow 语音转录参数"
    assert siliconflow_audio.reserved_body_keys == frozenset({"file", "model"})
    assert len(siliconflow_audio.fields) == 8
    sf_temperature = siliconflow_audio.field_by_safe_key("temperature")
    assert sf_temperature is not None
    assert sf_temperature.target_path == ("body", "temperature")
    assert sf_temperature.value_kind == "number"
    sf_stream = siliconflow_audio.field_by_safe_key("stream")
    assert sf_stream is not None
    assert sf_stream.value_kind == "boolean"
    assert sf_stream.label == "流式输出"
    ark_max_output_tokens = ark_audio.field_by_safe_key("max_tokens")
    assert ark_max_output_tokens is not None
    assert ark_max_output_tokens.target_path == (
        "body",
        "max_output_tokens",
    )
    mimo_language = mimo_audio.field_by_safe_key("language")
    assert mimo_language is not None
    assert mimo_language.target_path == (
        "body",
        "asr_options",
        "language",
    )
    assert [field.key for field in mimo_audio.fields] == ["language", "format", "audio_format"]
    assert mimo_audio.field_by_safe_key("max_tokens") is None
    assert mimo_audio.field_by_safe_key("prompt") is None


@pytest.mark.parametrize("retention_days", [0, 366])
def test_mimo_reasoning_retention_validation(retention_days: int) -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"xiaomi_mimo": {"reasoning_retention_days": retention_days}})


def test_mimo_audio_language_override_validation() -> None:
    from src.host_adapters.common.parameter_translation import (
        TranslationEnvelope,
        build_translation_context,
    )
    from src.host_adapters.xiaomi_mimo_provider.parameter_translation import _translate_mimo_audio_language
    from src.schemas import AudioTranscriptionRequestSnapshot

    catalog = get_parameter_catalog("xiaomi_mimo", "audio_transcription")
    config = MaiDockConfig.model_validate({"xiaomi_mimo": {"audio_transcription": {"overrides": {"language": "ja"}}}})
    overrides = build_parameter_overrides(config.xiaomi_mimo.audio_transcription, catalog)
    request = AudioTranscriptionRequestSnapshot.model_validate(
        {"model_info": {"model_identifier": "mimo-v2.5-asr"}, "api_provider": {"api_key": "k"}}
    )
    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label="Xiaomi Mimo",
        provider="xiaomi_mimo",
        capability="audio_transcription",
        model="mimo-v2.5-asr",
    )
    with pytest.raises(ValueError, match="auto/zh/en"):
        _translate_mimo_audio_language(context, TranslationEnvelope(), "ja")


def test_audio_transcription_catalogs_appear_in_provider_catalogs() -> None:
    from src.core.parameter_catalog import provider_catalogs

    ds_catalogs = provider_catalogs("dashscope")
    ds_audio = next(c for c in ds_catalogs if c.capability == "audio_transcription")
    assert ds_audio is not None
    assert len(ds_audio.fields) == 4

    sf_catalogs = provider_catalogs("siliconflow")
    sf_audio = next(c for c in sf_catalogs if c.capability == "audio_transcription")
    assert sf_audio is not None
    assert len(sf_audio.fields) == 8


def test_per_provider_retry_config_defaults() -> None:
    """验证 build_runtime_options(None) 输出中 retry 字段均为默认值。"""
    from src.config import build_runtime_options

    opts = build_runtime_options(None)

    assert opts.openai_max_retries == 3
    assert opts.openai_force_max_retries is False
    assert opts.openai_retry_interval == 5.0
    assert opts.openai_force_retry_interval is False

    assert opts.anthropic_max_retries == 3
    assert opts.anthropic_force_max_retries is False
    assert opts.anthropic_retry_interval == 5.0
    assert opts.anthropic_force_retry_interval is False

    assert opts.dashscope_max_retries == 3
    assert opts.dashscope_retry_interval == 5.0

    assert opts.siliconflow_max_retries == 3
    assert opts.siliconflow_retry_interval == 5.0

    assert opts.volcengine_max_retries == 3
    assert opts.volcengine_retry_interval == 5.0

    assert opts.mimo_max_retries == 3
    assert opts.mimo_retry_interval == 5.0


def test_per_provider_retry_config_custom_values_flow_to_runtime_options() -> None:
    """验证自定义 retry 配置值正确流入 ProviderRuntimeOptions。"""
    from src.config import MaiDockConfig, build_runtime_options

    raw = {
        "plugin": {},
        "diagnostics": {},
        "openai_responses": {
            "user_agent": "",
            "max_retries": 7,
            "force_max_retries": True,
            "retry_interval": 10.0,
            "force_retry_interval": True,
        },
        "anthropic_messages": {
            "user_agent": "",
            "max_retries": 1,
            "force_max_retries": False,
            "retry_interval": 2.5,
            "force_retry_interval": False,
        },
        "dashscope": {
            "user_agent": "",
            "force_official_endpoint": True,
            "max_retries": 0,
            "force_max_retries": True,
            "retry_interval": 0.0,
            "force_retry_interval": False,
        },
        "siliconflow": {
            "user_agent": "",
            "force_official_endpoint": True,
        },
        "volcengine_ark": {
            "user_agent": "",
            "force_official_endpoint": True,
        },
        "xiaomi_mimo": {
            "user_agent": "",
            "force_disable_thinking": True,
            "audio_transcription_prompt": "旧配置提示词",
            "audio_transcription": {
                "fields": {
                    "body_max_tokens_enabled": False,
                    "body_prompt_enabled": False,
                }
            },
        },
        "compatibility": {},
    }

    config = MaiDockConfig.model_validate(raw)
    opts = build_runtime_options(config)

    assert "audio_transcription_prompt" not in config.xiaomi_mimo.model_dump()

    assert opts.openai_max_retries == 7
    assert opts.openai_force_max_retries is True
    assert opts.openai_retry_interval == 10.0
    assert opts.openai_force_retry_interval is True

    assert opts.anthropic_max_retries == 1
    assert opts.anthropic_force_max_retries is False
    assert opts.anthropic_retry_interval == 2.5
    assert opts.anthropic_force_retry_interval is False

    assert opts.dashscope_max_retries == 0
    assert opts.dashscope_force_max_retries is True
    assert opts.dashscope_retry_interval == 0.0
    assert opts.dashscope_force_retry_interval is False

    assert opts.siliconflow_max_retries == 3
    assert opts.siliconflow_retry_interval == 5.0
