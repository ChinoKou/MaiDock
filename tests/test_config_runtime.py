import pytest
from pydantic import ValidationError

from src.config import (
    MaiDockConfig,
    build_runtime_options,
    normalize_maidock_config_data,
    normalize_user_agent,
)
from src.config_schema import build_maidock_config_schema
from src.core.parameter_catalog import (
    field_enabled_key,
    field_override_enabled_key,
    field_override_value_key,
    get_parameter_catalog,
)
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

    assert config.xiaomi_mimo.force_disable_thinking is True
    assert config.xiaomi_mimo.reasoning_retention_days == 30
    assert config.xiaomi_mimo.audio_transcription_prompt == "请转写这段音频"
    assert config.xiaomi_mimo.audio_transcription_language == "auto"
    assert config.volcengine_ark.audio_transcription_prompt.startswith("请识别")
    assert config.volcengine_ark.audio_transcription.default_params == {}
    assert config.xiaomi_mimo.audio_transcription.default_params == {}
    assert config.openai_responses.response.unknown_extra_params == "forward"
    assert config.openai_responses.audio_transcription.disabled_paths == []
    assert config.anthropic_messages.chat_completion.override_params == {}
    assert config.volcengine_ark.embeddings.accept_model_extra_params is True
    assert config.dashscope.chat_completion.accept_request_extra_params is True
    assert config.siliconflow.embeddings.default_params == {}
    assert config.dashscope.audio_transcription.accept_model_extra_params is True
    assert config.siliconflow.audio_transcription.disabled_paths == []


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

    assert config.xiaomi_mimo.force_disable_thinking is True
    assert config.xiaomi_mimo.reasoning_retention_days == 30
    assert config.xiaomi_mimo.audio_transcription_prompt == "请转写这段音频"
    assert config.xiaomi_mimo.audio_transcription.default_params == {}
    assert config.openai_responses.response.accept_model_extra_params is True
    assert config.dashscope.chat_completion.disabled_paths == []
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


def test_normalize_user_agent_uses_default_for_empty_values() -> None:
    assert normalize_user_agent(None) == DEFAULT_USER_AGENT
    assert normalize_user_agent("") == DEFAULT_USER_AGENT
    assert normalize_user_agent("  ") == DEFAULT_USER_AGENT
    assert normalize_user_agent("  Custom-UA  ") == "Custom-UA"


def test_parameter_policy_config_maps_to_runtime_options() -> None:
    config = MaiDockConfig.model_validate(
        {
            "dashscope": {
                "chat_completion": {
                    "accept_model_extra_params": False,
                    "disabled_paths": [
                        " unknown_field ",
                        "body.parameters.enable_thinking",
                    ],
                    "rejected_paths": ["headers.Authorization"],
                    "default_params": {"top_p": 0.8},
                    "override_params": {"body": {"parameters": {"enable_thinking": False}}},
                    "unknown_extra_params": "drop",
                }
            }
        }
    )

    policy = build_runtime_options(config).parameter_policies.get("dashscope", "chat_completion")

    assert policy.accept_model_extra_params is False
    assert policy.disabled_paths == ("unknown_field", "body.parameters.enable_thinking")
    assert policy.rejected_paths == ("headers.Authorization",)
    assert policy.default_params == {"top_p": 0.8}
    assert policy.override_params == {"body": {"parameters": {"enable_thinking": False}}}
    assert policy.unknown_extra_params == "drop"


def test_generated_field_controls_map_to_runtime_policy() -> None:
    catalog = get_parameter_catalog("dashscope", "chat_completion")
    top_p = catalog.field_by_safe_key("top_p")
    enable_thinking = catalog.field_by_safe_key("enable_thinking")
    assert top_p is not None
    assert enable_thinking is not None
    config = MaiDockConfig.model_validate(
        {
            "dashscope": {
                "chat_completion": {
                    "fields": {
                        field_enabled_key(enable_thinking): False,
                        field_override_enabled_key(top_p): True,
                        field_override_value_key(top_p): "0.6",
                    }
                }
            }
        }
    )

    policy = build_runtime_options(config).parameter_policies.get("dashscope", "chat_completion")

    assert policy.disabled_paths == ("body.parameters.enable_thinking",)
    assert policy.override_params == {"body": {"parameters": {"top_p": 0.6}}}


def test_embedding_catalogs_expose_provider_target_paths() -> None:
    chat_catalog = get_parameter_catalog("dashscope", "chat_completion")
    dashscope_embeddings = get_parameter_catalog("dashscope", "embeddings")
    siliconflow_embeddings = get_parameter_catalog("siliconflow", "embeddings")
    ark_embeddings = get_parameter_catalog("volcengine_ark", "embeddings")

    response_format = chat_catalog.field_by_safe_key("response_format")
    dashscope_dimensions = dashscope_embeddings.field_by_safe_key("dimension")
    siliconflow_dimensions = siliconflow_embeddings.field_by_safe_key("dimensions")
    ark_dimensions = ark_embeddings.field_by_safe_key("dimensions")

    assert response_format is not None
    assert response_format.target_path == ("body", "parameters", "response_format")
    assert response_format.config_key == "body_parameters_response_format"
    assert dashscope_dimensions is not None
    assert dashscope_dimensions.target_path == ("body", "parameters", "dimension")
    assert dashscope_dimensions.source_aliases == ("dimension",)
    assert siliconflow_dimensions is not None
    assert siliconflow_dimensions.target_path == ("body", "dimensions")
    assert ark_dimensions is not None
    assert ark_dimensions.target_path == ("body", "dimensions")


def test_maidock_config_normalization_fills_generated_controls() -> None:
    normalized, changed = normalize_maidock_config_data(MaiDockConfig().model_dump(mode="python"))

    fields = normalized["openai_responses"]["response"]["fields"]

    assert changed is True
    assert fields["body_top_p_enabled"] is True
    assert fields["body_top_p_override_enabled"] is False
    assert fields["body_top_p_override_value"] == ""


def test_maidock_config_normalization_preserves_legacy_policy_paths_without_migration() -> None:
    normalized, changed = normalize_maidock_config_data(
        {
            "plugin": {"enabled": True, "config_version": __version__},
            "dashscope": {
                "chat_completion": {
                    "disabled_paths": [
                        "body.parameters.enable_thinking",
                        "headers.X-Trace",
                    ],
                    "override_params": {
                        "top_p": 0.4,
                        "body": {"parameters": {"top_k": 20}, "custom": True},
                    },
                }
            },
        }
    )

    chat_completion = normalized["dashscope"]["chat_completion"]
    fields = chat_completion["fields"]

    assert changed is True
    assert fields["body_parameters_enable_thinking_enabled"] is True
    assert fields["body_parameters_top_p_override_enabled"] is False
    assert fields["body_parameters_top_k_override_enabled"] is False
    assert chat_completion["disabled_paths"] == [
        "body.parameters.enable_thinking",
        "headers.X-Trace",
    ]
    assert chat_completion["override_params"] == {
        "top_p": 0.4,
        "body": {"parameters": {"top_k": 20}, "custom": True},
    }


def test_maidock_webui_schema_uses_dotted_scalar_sections() -> None:
    schema = build_maidock_config_schema(plugin_id="maidock")
    sections = schema["sections"]
    response_fields = sections["openai_responses_response_fields"]
    dashscope_fields = sections["dashscope_chat_completion_fields"]

    assert response_fields["name"] == "openai_responses.response.fields"
    assert dashscope_fields["name"] == "dashscope.chat_completion.fields"
    dashscope_embedding_fields = sections["dashscope_embeddings_fields"]["fields"]
    siliconflow_embedding_fields = sections["siliconflow_embeddings_fields"]["fields"]
    ark_embedding_fields = sections["volcengine_ark_embeddings_fields"]["fields"]

    assert response_fields["fields"]["body_top_p_enabled"]["ui_type"] == "switch"
    assert response_fields["fields"]["body_top_p_override_value"]["ui_type"] == "textarea"
    assert "body_parameters_dimension_enabled" in dashscope_embedding_fields
    assert "body_dimensions_enabled" not in dashscope_embedding_fields
    assert "body_dimensions_enabled" in siliconflow_embedding_fields
    assert "body_parameters_dimension_enabled" not in siliconflow_embedding_fields
    assert "body_dimensions_enabled" in ark_embedding_fields
    assert "body_parameters_dimension_enabled" not in ark_embedding_fields
    for section in sections.values():
        for field in section["fields"].values():
            assert field["ui_type"] != "json"
            assert field["type"] != "object"


def test_parameter_policy_config_rejects_invalid_unknown_policy() -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"openai_responses": {"response": {"unknown_extra_params": "invalid"}}})


def test_parameter_policy_config_normalizes_nested_param_objects() -> None:
    config = MaiDockConfig.model_validate(
        {
            "openai_responses": {
                "response": {
                    "default_params": {"body": {"temperature": 0.2}},
                    "override_params": {"headers": {"X-Test": "1"}},
                }
            }
        }
    )

    assert config.openai_responses.response.default_params == {"body": {"temperature": 0.2}}
    assert config.openai_responses.response.override_params == {"headers": {"X-Test": "1"}}


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
    ark_max_output_tokens = ark_audio.field_by_safe_key("max_output_tokens")
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


@pytest.mark.parametrize("retention_days", [0, 366])
def test_mimo_reasoning_retention_validation(retention_days: int) -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"xiaomi_mimo": {"reasoning_retention_days": retention_days}})


def test_mimo_audio_language_validation() -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"xiaomi_mimo": {"audio_transcription_language": "ja"}})


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
            "audio_transcription_prompt": "",
        },
        "compatibility": {},
    }

    config = MaiDockConfig.model_validate(raw)
    opts = build_runtime_options(config)

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
