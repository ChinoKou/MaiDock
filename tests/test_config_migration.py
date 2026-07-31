"""1.1.3 → 1.2.0 参数覆写迁移测试。"""

from typing import cast

import pytest
from maibot_sdk.config import rebuild_plugin_config_data

from src.config import MaiDockConfig, normalize_maidock_config_data
from src.config_migration import migrate_legacy_config
from src.plugin import MaiDockPlugin
from src.version import __version__


def _section(config: dict, provider: str, capability: str) -> dict:
    return cast(dict, cast(dict, config[provider])[capability])


def _legacy_113_config() -> dict:
    """构造 Runner 升级重建后的 1.1.3 配置形状（含 fields 桥接段）。"""

    return {
        "plugin": {"enabled": True, "config_version": "1.1.3"},
        "dashscope": {
            "chat_completion": {
                "accept_model_extra_params": True,
                "accept_request_extra_params": True,
                "unknown_extra_params": "forward",
                "disabled_paths": [],
                "rejected_paths": [],
                "default_params": {},
                "override_params": {},
                "fields": {
                    "body_parameters_temperature_enabled": True,
                    "body_parameters_temperature_override_enabled": False,
                    "body_parameters_temperature_override_value": "0.5",
                    "body_parameters_top_p_override_enabled": True,
                    "body_parameters_top_p_override_value": "0.6",
                    "body_parameters_enable_search_override_enabled": False,
                    "body_parameters_enable_search_override_value": "true",
                },
            }
        },
        "volcengine_ark": {
            "audio_transcription_prompt": "自定义提示词",
            "audio_transcription": {},
        },
        "xiaomi_mimo": {
            "force_disable_thinking": True,
            "audio_transcription_language": "zh",
            "chat_completion": {
                "fields": {
                    "body_max_tokens_enabled": True,
                    "body_max_tokens_override_enabled": True,
                    "body_max_tokens_override_value": "88",
                }
            },
            "audio_transcription": {},
        },
    }


def test_migrate_legacy_config_moves_enabled_overrides_only() -> None:
    migrated, changed = migrate_legacy_config(_legacy_113_config())

    assert changed is True
    dashscope_chat = _section(migrated, "dashscope", "chat_completion")
    # 只迁移 override_enabled=true 的覆写；显式 false 与关闭状态遗留值不生效。
    assert dashscope_chat["overrides"] == {"top_p": "0.6"}
    # 旧策略结构与 fields 段全部删除。
    assert "fields" not in dashscope_chat
    for legacy_key in (
        "accept_model_extra_params",
        "accept_request_extra_params",
        "unknown_extra_params",
        "disabled_paths",
        "rejected_paths",
        "default_params",
        "override_params",
    ):
        assert legacy_key not in dashscope_chat
    # 旧顶层字段迁移到覆写目录。
    assert _section(migrated, "volcengine_ark", "audio_transcription")["overrides"]["prompt"] == "自定义提示词"
    assert _section(migrated, "xiaomi_mimo", "audio_transcription")["overrides"]["language"] == "zh"
    assert _section(migrated, "xiaomi_mimo", "chat_completion")["overrides"]["max_tokens"] == "88"
    assert _section(migrated, "xiaomi_mimo", "chat_completion")["overrides"]["thinking"] == '{"type":"disabled"}'
    ark_section = cast(dict, migrated["volcengine_ark"])
    mimo_section = cast(dict, migrated["xiaomi_mimo"])
    assert "audio_transcription_prompt" not in ark_section
    assert "force_disable_thinking" not in mimo_section
    assert "audio_transcription_language" not in mimo_section
    # 版本号同步到当前版本。
    assert cast(dict, migrated["plugin"])["config_version"] == __version__


def test_migrate_legacy_config_boolean_override_values() -> None:
    config = {
        "plugin": {"config_version": "1.1.3"},
        "openai_responses": {
            "response": {
                "fields": {
                    "body_store_override_enabled": True,
                    "body_store_override_value": False,
                }
            }
        },
    }
    migrated, changed = migrate_legacy_config(config)

    assert changed is True
    assert _section(migrated, "openai_responses", "response")["overrides"] == {"store": "false"}


def test_migrate_legacy_config_force_disable_thinking_false_is_not_migrated() -> None:
    config = {
        "plugin": {"config_version": "1.1.3"},
        "xiaomi_mimo": {
            "force_disable_thinking": False,
            "chat_completion": {},
        },
    }
    migrated, changed = migrate_legacy_config(config)

    # 显式 false 不迁移 thinking 覆写，但删除旧键本身必须计入变化。
    assert changed is True
    assert "thinking" not in _section(migrated, "xiaomi_mimo", "chat_completion").get("overrides", {})
    assert "force_disable_thinking" not in cast(dict, migrated["xiaomi_mimo"])


def test_migrate_legacy_config_is_idempotent() -> None:
    migrated, _ = migrate_legacy_config(_legacy_113_config())
    second, changed = migrate_legacy_config(migrated)

    assert changed is False
    assert second == migrated


def test_normalize_maidock_config_data_migrates_and_strips_bridge_keys() -> None:
    normalized, changed = normalize_maidock_config_data(_legacy_113_config())

    assert changed is True
    chat_completion = _section(normalized, "dashscope", "chat_completion")
    assert chat_completion["overrides"]["top_p"] == "0.6"
    assert "fields" not in chat_completion
    assert "temperature" not in chat_completion["overrides"]
    # 默认覆写文本被填充。
    assert chat_completion["overrides"]["result_format"] == "message"
    # 再次归一化稳定（无桥接残留循环）。
    second, changed_again = normalize_maidock_config_data(normalized)
    assert changed_again is False
    assert second == normalized


def test_runner_rebuild_bridge_preserves_legacy_values_before_normalization() -> None:
    """模拟 Core 先按新默认骨架重建旧配置的真实升级顺序。"""

    plugin = MaiDockPlugin()
    rebuilt = rebuild_plugin_config_data(plugin.get_default_config(), _legacy_113_config())
    normalized, changed = plugin.normalize_plugin_config(rebuilt)

    assert changed is True
    assert _section(normalized, "dashscope", "chat_completion")["overrides"]["top_p"] == "0.6"
    assert _section(normalized, "volcengine_ark", "audio_transcription")["overrides"]["prompt"] == "自定义提示词"
    assert _section(normalized, "xiaomi_mimo", "audio_transcription")["overrides"]["language"] == "zh"
    assert _section(normalized, "xiaomi_mimo", "chat_completion")["overrides"]["max_tokens"] == "88"
    assert _section(normalized, "xiaomi_mimo", "chat_completion")["overrides"]["thinking"] == ('{"type":"disabled"}')
    assert "fields" not in _section(normalized, "dashscope", "chat_completion")
    assert "audio_transcription_prompt" not in cast(dict, normalized["volcengine_ark"])
    assert "force_disable_thinking" not in cast(dict, normalized["xiaomi_mimo"])
    assert "audio_transcription_language" not in cast(dict, normalized["xiaomi_mimo"])


def test_runner_bridge_defaults_do_not_become_user_overrides() -> None:
    plugin = MaiDockPlugin()
    rebuilt = rebuild_plugin_config_data(plugin.get_default_config(), _legacy_113_config())
    # 模拟用户从未开启 DashScope temperature 覆写；桥接默认值不得使旧值生效。
    normalized, _ = plugin.normalize_plugin_config(rebuilt)

    assert "temperature" not in _section(normalized, "dashscope", "chat_completion")["overrides"]


def test_normalize_maidock_config_data_new_install_is_stable() -> None:
    first, _ = normalize_maidock_config_data({})
    second, changed = normalize_maidock_config_data(first)

    assert changed is False
    assert second == first
    # 桥接字段不残留。
    assert "fields" not in _section(second, "dashscope", "chat_completion")


def test_normalize_maidock_config_data_rejects_non_dict_bridge_fields() -> None:
    """非 dict 的 fields 桥接输入是异常配置，直接完整暴露而不是静默剥除。"""

    from src.config import normalize_maidock_config_data

    with pytest.raises(TypeError, match="fields"):
        normalize_maidock_config_data(
            {
                "plugin": {"config_version": "1.1.3"},
                "dashscope": {"chat_completion": {"fields": "corrupted"}},
            }
        )


def test_migrated_config_validates_against_new_model() -> None:
    migrated, _ = migrate_legacy_config(_legacy_113_config())
    config = MaiDockConfig.model_validate(migrated)

    assert config.dashscope.chat_completion.overrides == {"top_p": "0.6"}
    assert config.volcengine_ark.audio_transcription.overrides["prompt"] == "自定义提示词"
    assert config.xiaomi_mimo.chat_completion.overrides["thinking"] == '{"type":"disabled"}'
    # 桥接字段只用于升级读取，不进入 model_dump。
    assert "fields" not in config.model_dump(mode="python")["dashscope"]["chat_completion"]
