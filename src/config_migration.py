"""1.1.3 → 1.2.0 参数覆写体系迁移（独立纯函数，幂等）。

正常请求构建与参数转译不得依赖本模块：它只在插件配置归一化入口执行，
把云端 1.1.3 的旧字段控制（fields 三键、extra 策略、独立默认配置项）
迁移到新的 overrides 目录结构，并在同一趟内删除全部旧结构。
"""

from collections.abc import Mapping

from .core.json_types import JsonValue, mapping_to_json_object
from .core.parameter_catalog import (
    CapabilityParameterCatalog,
    ParameterFieldDefinition,
    dotted_path,
    iter_parameter_catalogs,
    safe_parameter_key,
)
from .version import __version__

_LEGACY_PROVIDERS = frozenset(
    {
        "openai_responses",
        "anthropic_messages",
        "dashscope",
        "siliconflow",
        "volcengine_ark",
        "xiaomi_mimo",
    }
)
_OVERRIDE_ENABLED_SUFFIX = "_override_enabled"
_OVERRIDE_VALUE_SUFFIX = "_override_value"
_LEGACY_CAPABILITY_KEYS = (
    "accept_model_extra_params",
    "accept_request_extra_params",
    "disabled_paths",
    "rejected_paths",
    "default_params",
    "override_params",
    "unknown_extra_params",
)
_ARK_AUDIO_TRANSCRIPTION_PROMPT = "请识别音频中的内容，以文字形式返回识别结果。"


def build_legacy_migration_bridge_defaults() -> dict[str, JsonValue]:
    """构造供 Core 版本重建使用的 1.1.3 旧字段骨架。"""

    bridge: dict[str, JsonValue] = {}
    for catalog in iter_parameter_catalogs():
        if catalog.provider not in _LEGACY_PROVIDERS:
            continue
        provider_section = _ensure_object(bridge, catalog.provider)
        capability_section = _ensure_object(provider_section, catalog.capability)
        fields: dict[str, JsonValue] = {}
        for field in catalog.fields:
            # 1.1.3 的 Responses tools 仅来自 extra_params，不属于 fields 控制目录。
            if catalog.capability == "response" and field.key == "tools":
                continue
            legacy_key = _legacy_field_key(catalog, field)
            fields[f"{legacy_key}_enabled"] = True
            fields[f"{legacy_key}{_OVERRIDE_ENABLED_SUFFIX}"] = False
            fields[f"{legacy_key}{_OVERRIDE_VALUE_SUFFIX}"] = False if field.value_kind == "boolean" else ""
        capability_section["fields"] = fields

    ark_section = _ensure_object(bridge, "volcengine_ark")
    ark_section["audio_transcription_prompt"] = _ARK_AUDIO_TRANSCRIPTION_PROMPT
    mimo_section = _ensure_object(bridge, "xiaomi_mimo")
    mimo_section["force_disable_thinking"] = True
    mimo_section["audio_transcription_language"] = "auto"
    return bridge


def inject_legacy_migration_bridge(config_data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """向最终默认配置注入只供 Runner 重建读取的旧字段骨架。"""

    current = mapping_to_json_object(config_data)
    _merge_missing(current, build_legacy_migration_bridge_defaults())
    return current


def migrate_legacy_config(config_data: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], bool]:
    """迁移旧版参数策略配置；未检测到旧结构时原样返回（幂等）。"""

    current = mapping_to_json_object(config_data)
    changed = False
    for catalog in iter_parameter_catalogs():
        capability_section = _capability_section(current, catalog)
        if capability_section is None:
            continue
        if "fields" in capability_section:
            raw_fields = capability_section.pop("fields")
            if not isinstance(raw_fields, dict):
                raise TypeError(
                    f"{catalog.provider}.{catalog.capability}.fields 必须是 object，实际为 {type(raw_fields).__name__}"
                )
            fields = mapping_to_json_object(raw_fields)
            migrated = _migrate_field_overrides(fields, catalog)
            if migrated:
                overrides = capability_section.get("overrides")
                overrides_object = mapping_to_json_object(overrides) if isinstance(overrides, dict) else {}
                for key, value in migrated.items():
                    overrides_object.setdefault(key, value)
                capability_section["overrides"] = overrides_object
            changed = True
        for legacy_key in _LEGACY_CAPABILITY_KEYS:
            if legacy_key in capability_section:
                capability_section.pop(legacy_key)
                changed = True

    ark_section = current.get("volcengine_ark")
    if isinstance(ark_section, dict):
        if "audio_transcription_prompt" in ark_section:
            raw_prompt = ark_section.pop("audio_transcription_prompt")
            if raw_prompt is not None:
                _set_override(current, "volcengine_ark", "audio_transcription", "prompt", str(raw_prompt).strip())
            changed = True

    mimo_section = current.get("xiaomi_mimo")
    if isinstance(mimo_section, dict):
        if "audio_transcription_language" in mimo_section:
            raw_language = mimo_section.pop("audio_transcription_language")
            if raw_language is not None:
                _set_override(current, "xiaomi_mimo", "audio_transcription", "language", str(raw_language).strip())
            changed = True
        if "force_disable_thinking" in mimo_section:
            raw_force_thinking = mimo_section.pop("force_disable_thinking")
            if raw_force_thinking is True:
                _set_override(current, "xiaomi_mimo", "chat_completion", "thinking", '{"type":"disabled"}')
            changed = True

    if changed:
        plugin_section = current.get("plugin")
        if isinstance(plugin_section, dict):
            plugin_section["config_version"] = __version__
        elif plugin_section is None:
            current["plugin"] = {"config_version": __version__}

    return current, changed


def _ensure_object(parent: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    section: dict[str, JsonValue] = {}
    parent[key] = section
    return section


def _merge_missing(target: dict[str, JsonValue], defaults: Mapping[str, JsonValue]) -> None:
    for key, default_value in defaults.items():
        current_value = target.get(key)
        if isinstance(current_value, dict) and isinstance(default_value, dict):
            _merge_missing(current_value, default_value)
            continue
        if key not in target:
            target[key] = mapping_to_json_object(default_value) if isinstance(default_value, dict) else default_value


def _capability_section(
    config_data: dict[str, JsonValue],
    catalog: CapabilityParameterCatalog,
) -> dict[str, JsonValue] | None:
    provider_section = config_data.get(catalog.provider)
    if not isinstance(provider_section, dict):
        return None
    capability_section = provider_section.get(catalog.capability)
    if not isinstance(capability_section, dict):
        return None
    return capability_section


def _migrate_field_overrides(
    fields: Mapping[str, JsonValue],
    catalog: CapabilityParameterCatalog,
) -> dict[str, str]:
    """从旧 fields 三键中提取启用状态的覆写值。

    只迁移 ``<key>_override_enabled = true`` 的条目；显式布尔 ``false``
    与关闭状态下遗留的旧值不得生效。
    """

    migrated: dict[str, str] = {}
    for raw_key, raw_value in fields.items():
        if not isinstance(raw_key, str) or not raw_key.endswith(_OVERRIDE_ENABLED_SUFFIX):
            continue
        if raw_value is not True:
            continue
        field_key = raw_key[: -len(_OVERRIDE_ENABLED_SUFFIX)]
        field = _match_legacy_field(field_key, catalog)
        if field is None:
            continue
        value_key = f"{field_key}{_OVERRIDE_VALUE_SUFFIX}"
        raw_value_text = fields.get(value_key)
        if isinstance(raw_value_text, bool):
            migrated[field.key] = "true" if raw_value_text else "false"
            continue
        if isinstance(raw_value_text, str):
            migrated[field.key] = raw_value_text
            continue
        if raw_value_text is not None:
            migrated[field.key] = str(raw_value_text)
    return migrated


def _match_legacy_field(raw_key: str, catalog: CapabilityParameterCatalog) -> ParameterFieldDefinition | None:
    """按新旧两套配置键匹配目录字段。

    旧版 config_key 由 target_path 生成（例如 ``body_parameters_top_p``），
    新版直接使用规范参数名（例如 ``top_p``）。
    """

    field = catalog.field_by_safe_key(raw_key)
    if field is not None:
        return field
    for candidate in catalog.fields:
        legacy_key = _legacy_field_key(catalog, candidate)
        if legacy_key == safe_parameter_key(raw_key):
            return candidate
    return None


def _legacy_field_key(
    catalog: CapabilityParameterCatalog,
    field: ParameterFieldDefinition,
) -> str:
    """复现 1.1.3 字段控制目录的配置键。"""

    if catalog.provider == "xiaomi_mimo" and catalog.capability == "chat_completion" and field.key == "max_tokens":
        return "body_max_tokens"
    return safe_parameter_key(dotted_path(field.target_path))


def _set_override(
    config_data: dict[str, JsonValue],
    provider: str,
    capability: str,
    key: str,
    value: str,
) -> None:
    provider_section = config_data.get(provider)
    if not isinstance(provider_section, dict):
        provider_section = {}
        config_data[provider] = provider_section
    capability_section = provider_section.get(capability)
    if not isinstance(capability_section, dict):
        capability_section = {}
        provider_section[capability] = capability_section
    overrides = capability_section.get("overrides")
    overrides_object = mapping_to_json_object(overrides) if isinstance(overrides, dict) else {}
    if value:
        overrides_object.setdefault(key, value)
    capability_section["overrides"] = overrides_object
