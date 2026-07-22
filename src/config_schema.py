from typing import cast

from .core.parameter_catalog import (
    CAPABILITY_TITLES,
    PROVIDER_TITLES,
    CapabilityParameterCatalog,
    ParameterFieldDefinition,
    field_enabled_key,
    field_override_enabled_key,
    field_override_value_key,
    get_parameter_catalog,
    provider_catalogs,
)
from .core.parameter_policy import CapabilityKey, ProviderPolicyKey
from .i18n import DEFAULT_LOCALE, Locale, translate, use_locale
from .version import __version__

_PROVIDER_ICONS: dict[ProviderPolicyKey, str] = {
    "openai_responses": "bot",
    "anthropic_messages": "bot-message-square",
    "dashscope": "bot",
    "siliconflow": "bot",
    "volcengine_ark": "bot",
    "xiaomi_mimo": "bot",
}

_PROVIDER_BASE_FIELDS: dict[ProviderPolicyKey, tuple[str, ...]] = {
    "openai_responses": (
        "user_agent",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "anthropic_messages": (
        "user_agent",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "dashscope": (
        "user_agent",
        "force_official_endpoint",
        "auto_detect_endpoint",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "siliconflow": (
        "user_agent",
        "force_official_endpoint",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "volcengine_ark": (
        "user_agent",
        "force_official_endpoint",
        "prefix_cache_enabled",
        "prefix_cache_ttl_seconds",
        "audio_transcription_prompt",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "xiaomi_mimo": (
        "user_agent",
        "force_disable_thinking",
        "reasoning_retention_days",
        "audio_transcription_language",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
}

_PARAMETER_LABEL_KEYS: dict[str, str] = {
    "max_completion_tokens": "ui.parameter.max_completion_tokens",
    "thinking_budget": "ui.parameter.thinking_budget",
    "reasoning_effort": "ui.parameter.reasoning_effort",
    "parallel_tool_calls": "ui.parameter.parallel_tool_calls",
    "store": "ui.parameter.store",
    "stream": "ui.parameter.stream",
    "enable_thinking": "ui.parameter.enable_thinking",
    "enable_search": "ui.parameter.enable_search",
    "search_options": "ui.parameter.search_options",
    "incremental_output": "ui.parameter.incremental_output",
    "tool_stream": "ui.parameter.tool_stream",
    "enable_code_interpreter": "ui.parameter.enable_code_interpreter",
    "vl_high_resolution_images": "ui.parameter.vl_high_resolution_images",
    "auto_truncation": "ui.parameter.auto_truncation",
    "enable_fusion": "ui.parameter.enable_fusion",
    "enable_itn": "ui.parameter.enable_itn",
    "sparse_embedding": "ui.parameter.sparse_embedding",
    "format": "ui.parameter.format",
    "audio_format": "ui.parameter.audio_format",
}

_STATIC_FIELD_KEYS: dict[str, tuple[str, str | None]] = {
    "enabled": ("ui.field.enabled.label", None),
    "locale": ("ui.field.locale.label", "ui.field.locale.hint"),
    "config_version": ("ui.field.config_version.label", None),
    "include_raw_data": ("ui.field.include_raw_data.label", "ui.field.include_raw_data.hint"),
    "log_payload_summary": ("ui.field.log_summary.label", None),
    "log_payload_debug": ("ui.field.log_debug.label", "ui.field.log_debug.hint"),
    "tool_argument_parse_mode": ("ui.field.tool_parse_mode.label", None),
    "reasoning_parse_mode": ("ui.field.reasoning_parse_mode.label", None),
    "invalid_image_policy": ("ui.field.invalid_image_policy.label", None),
    "max_image_bytes_mb": ("ui.field.max_image_bytes.label", None),
    "max_image_pixels": ("ui.field.max_image_pixels.label", None),
    "max_image_dimension": ("ui.field.max_image_dimension.label", None),
    "max_image_frames": ("ui.field.max_image_frames.label", None),
    "force_disable_thinking": ("ui.field.force_thinking.label", "ui.field.force_thinking.hint"),
    "reasoning_retention_days": ("ui.field.reasoning_retention.label", "ui.field.reasoning_retention.hint"),
    "audio_transcription_language": ("ui.field.asr_language.label", "ui.field.asr_language.hint"),
    "auto_detect_endpoint": ("ui.field.auto_endpoint.label", "ui.field.auto_endpoint.hint"),
    "prefix_cache_enabled": ("ui.field.prefix_cache.label", "ui.field.prefix_cache.hint"),
    "prefix_cache_ttl_seconds": ("ui.field.prefix_cache_ttl.label", "ui.field.prefix_cache_ttl.hint"),
    "max_retries": ("ui.field.max_retries.label", "ui.field.max_retries.hint"),
    "force_max_retries": ("ui.field.force_max_retries.label", "ui.field.force_value.hint"),
    "retry_interval": ("ui.field.retry_interval.label", "ui.field.retry_interval.hint"),
    "force_retry_interval": ("ui.field.force_retry_interval.label", "ui.field.force_value.hint"),
    "accept_model_extra_params": ("ui.field.accept_model_extra.label", None),
    "accept_request_extra_params": ("ui.field.accept_request_extra.label", None),
    "unknown_extra_params": ("ui.field.unknown_extra.label", "ui.field.unknown_extra.hint"),
}


def build_maidock_config_schema(
    *,
    plugin_id: str = "",
    plugin_name: str = "",
    plugin_version: str = "",
    plugin_description: str = "",
    plugin_author: str = "",
    locale: Locale = DEFAULT_LOCALE,
) -> dict:
    """使用指定语言构建 WebUI 安全插件配置 schema。"""

    with use_locale(locale):
        schema = _build_maidock_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        _localize_schema(schema)
        return schema


def _build_maidock_config_schema(
    *,
    plugin_id: str = "",
    plugin_name: str = "",
    plugin_version: str = "",
    plugin_description: str = "",
    plugin_author: str = "",
) -> dict:
    """构建不包含顶层 object 字段的 WebUI 安全插件配置 schema。"""

    sections: dict[str, dict] = {
        "plugin": _section(
            name="plugin",
            title="插件",
            description="MaiDock 插件基础设置。",
            icon="package",
            order=0,
            fields={
                "enabled": _field(
                    name="enabled",
                    field_type="boolean",
                    label="启用 MaiDock",
                    default=True,
                    ui_type="switch",
                    order=0,
                ),
                "locale": _select_field(
                    name="locale",
                    label="MaiDock locale",
                    default=DEFAULT_LOCALE,
                    choices=("zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"),
                    hint="MaiDock display, log, and error locale.",
                    order=1,
                ),
                "config_version": _field(
                    name="config_version",
                    field_type="string",
                    label="配置版本",
                    default=__version__,
                    ui_type="text",
                    order=2,
                    disabled=True,
                ),
            },
        ),
        "diagnostics": _section(
            name="diagnostics",
            title="诊断",
            description="请求/响应摘要与调试日志设置。",
            icon="bug",
            order=1,
            fields={
                "include_raw_data": _field(
                    name="include_raw_data",
                    field_type="boolean",
                    label="包含 raw_data",
                    default=False,
                    ui_type="switch",
                    hint="把 Provider API 响应摘要写入 ProviderResponse.raw_data。",
                    order=0,
                ),
                "log_payload_summary": _field(
                    name="log_payload_summary",
                    field_type="boolean",
                    label="记录摘要日志",
                    default=True,
                    ui_type="switch",
                    order=1,
                ),
                "log_payload_debug": _field(
                    name="log_payload_debug",
                    field_type="boolean",
                    label="记录详细载荷日志",
                    default=False,
                    ui_type="switch",
                    hint="仅用于排障；日志会脱敏，但仍建议谨慎开启。",
                    order=2,
                ),
            },
        ),
        "compatibility": _section(
            name="compatibility",
            title="兼容性",
            description="解析模式、未知参数与图片输入保护。",
            icon="settings-2",
            order=90,
            fields={
                "tool_argument_parse_mode": _select_field(
                    name="tool_argument_parse_mode",
                    label="工具参数解析模式",
                    default="auto",
                    choices=("auto", "strict", "repair", "double_decode"),
                    order=0,
                ),
                "reasoning_parse_mode": _select_field(
                    name="reasoning_parse_mode",
                    label="推理内容解析模式",
                    default="auto",
                    choices=("auto", "native", "think_tag", "none"),
                    order=1,
                ),
                "invalid_image_policy": _select_field(
                    name="invalid_image_policy",
                    label="无效图片策略",
                    default="placeholder",
                    choices=("placeholder", "skip", "error"),
                    order=2,
                ),
                "max_image_bytes_mb": _number_field("max_image_bytes_mb", "单图最大解码大小 MB", 30, 3),
                "max_image_pixels": _number_field("max_image_pixels", "单图最大像素", 25_000_000, 4),
                "max_image_dimension": _number_field("max_image_dimension", "单边最大像素", 8192, 5),
                "max_image_frames": _number_field("max_image_frames", "动图最大帧数", 64, 6),
            },
        ),
    }

    tabs: list[dict] = [
        _tab(
            "general",
            "通用",
            ("plugin", "diagnostics", "compatibility"),
            icon="settings",
            order=0,
        )
    ]
    provider_order = 10
    for provider in PROVIDER_TITLES:
        provider_sections = _add_provider_sections(sections, provider, order=provider_order)
        tabs.append(
            _tab(
                provider,
                _provider_tab_title(provider),
                provider_sections,
                icon=_PROVIDER_ICONS[provider],
                order=provider_order,
            )
        )
        provider_order += 10

    return {
        "plugin_id": plugin_id,
        "plugin_info": {
            "name": plugin_name or "MaiDock",
            "version": plugin_version or __version__,
            "description": plugin_description or "MaiDock LLM Provider adapter plugin",
            "author": plugin_author,
        },
        "sections": sections,
        "layout": {"type": "tabs", "tabs": tabs},
    }


def _localize_schema(schema: dict) -> None:
    plugin_info = schema["plugin_info"]
    plugin_info["description"] = translate("ui.plugin.description")

    for tab in schema["layout"]["tabs"]:
        tab_id = str(tab["id"])
        if tab_id == "general":
            tab["title"] = translate("ui.tab.general")
            continue
        provider = cast(ProviderPolicyKey, tab_id)
        provider_title = _localized_provider_title(provider)
        tab["title"] = (
            translate("ui.tab.protocol_client", provider=provider_title)
            if provider in {"openai_responses", "anthropic_messages"}
            else provider_title
        )

    for section_key, section in schema["sections"].items():
        section_name = str(section["name"])
        _localize_section(section_key, section_name, section)
        _localize_section_fields(section_name, section["fields"])


def _localize_section(section_key: str, section_name: str, section: dict) -> None:
    if section_key == "plugin":
        section["title"] = translate("ui.section.plugin.title")
        section["description"] = translate("ui.section.plugin.description")
        return
    if section_key == "diagnostics":
        section["title"] = translate("ui.section.diagnostics.title")
        section["description"] = translate("ui.section.diagnostics.description")
        return
    if section_key == "compatibility":
        section["title"] = translate("ui.section.compatibility.title")
        section["description"] = translate("ui.section.compatibility.description")
        return
    if section_key in _PROVIDER_BASE_FIELDS:
        provider = cast(ProviderPolicyKey, section_key)
        section["title"] = translate(
            "ui.section.provider_base.title",
            provider=_localized_provider_title(provider),
        )
        section["description"] = translate("ui.section.provider_base.description")
        return

    parts = section_name.split(".")
    if len(parts) < 2:
        return
    provider = cast(ProviderPolicyKey, parts[0])
    capability = cast(CapabilityKey, parts[1])
    capability_title = _localized_capability_title(capability)
    if section_name.endswith(".fields"):
        title_key = "ui.section.fields.title_spaced" if capability == "embeddings" else "ui.section.fields.title"
        section["title"] = translate(title_key, capability=capability_title)
        section["description"] = _localized_fields_description(provider, capability)
        return
    title_key = "ui.section.policy.title_spaced" if capability == "embeddings" else "ui.section.policy.title"
    section["title"] = translate(title_key, capability=capability_title)
    section["description"] = _localized_policy_description(provider, capability)


def _localize_section_fields(section_name: str, fields: dict[str, dict]) -> None:
    for field_name, field_schema in fields.items():
        static_keys = _STATIC_FIELD_KEYS.get(field_name)
        if static_keys is not None:
            label_key, hint_key = static_keys
            _set_field_text(field_schema, label_key, hint_key)

    if section_name in _PROVIDER_BASE_FIELDS:
        _localize_provider_base_fields(cast(ProviderPolicyKey, section_name), fields)
        return
    if not section_name.endswith(".fields"):
        return

    provider_name, capability_name, _ = section_name.split(".")
    catalog = get_parameter_catalog(
        cast(ProviderPolicyKey, provider_name),
        cast(CapabilityKey, capability_name),
    )
    for parameter in catalog.fields:
        label = _localized_parameter_label(parameter)
        description = _localized_parameter_description(parameter, label)
        enabled = fields[field_enabled_key(parameter)]
        enabled["label"] = translate("ui.field.parameter.send.label", field=label)
        enabled["description"] = enabled["label"]
        enabled["hint"] = translate("ui.field.parameter.send.hint", description=description)

        override_enabled = fields[field_override_enabled_key(parameter)]
        override_enabled["label"] = translate("ui.field.parameter.override.label", field=label)
        override_enabled["description"] = override_enabled["label"]
        override_enabled["hint"] = translate("ui.field.parameter.override.hint")

        override_value = fields[field_override_value_key(parameter)]
        override_value["label"] = translate("ui.field.parameter.override_value.label", field=label)
        override_value["description"] = override_value["label"]
        override_value["hint"] = description if parameter.value_kind == "boolean" else _localized_value_hint(parameter)
        if parameter.value_kind != "boolean":
            override_value["placeholder"] = _localized_value_placeholder(parameter)


def _set_field_text(field_schema: dict, label_key: str, hint_key: str | None) -> None:
    field_schema["label"] = translate(label_key)
    field_schema["description"] = field_schema["label"]
    if hint_key is not None:
        field_schema["hint"] = translate(hint_key)


def _localize_provider_base_fields(provider: ProviderPolicyKey, fields: dict[str, dict]) -> None:
    if "user_agent" in fields:
        fields["user_agent"]["placeholder"] = translate("ui.field.user_agent.placeholder")
    if "force_official_endpoint" in fields:
        endpoint_key = {
            "volcengine_ark": "ark",
            "dashscope": "dashscope",
            "siliconflow": "siliconflow",
        }.get(provider, "default")
        endpoint = fields["force_official_endpoint"]
        endpoint["label"] = translate(f"ui.endpoint.label.{endpoint_key}")
        endpoint["description"] = endpoint["label"]
        endpoint["hint"] = translate(f"ui.endpoint.hint.{endpoint_key}")
    if "audio_transcription_prompt" in fields:
        prompt = fields["audio_transcription_prompt"]
        prompt["label"] = translate("ui.field.transcription_prompt.label")
        prompt["description"] = prompt["label"]
        prompt["hint"] = translate("ui.field.transcription_prompt.hint.ark")


def _localized_provider_title(provider: ProviderPolicyKey) -> str:
    return translate(f"ui.provider.{provider}")


def _localized_capability_title(capability: CapabilityKey) -> str:
    return translate(f"ui.capability.{capability}")


def _localized_policy_description(provider: ProviderPolicyKey, capability: CapabilityKey) -> str:
    if provider == "xiaomi_mimo" and capability == "audio_transcription":
        return translate("ui.section.policy.description.mimo_audio")
    if provider == "volcengine_ark" and capability == "audio_transcription":
        return translate("ui.section.policy.description.ark_audio")
    return translate("ui.section.policy.description")


def _localized_fields_description(provider: ProviderPolicyKey, capability: CapabilityKey) -> str:
    if provider == "xiaomi_mimo" and capability == "audio_transcription":
        return translate("ui.section.fields.description.mimo_audio")
    if provider == "volcengine_ark" and capability == "audio_transcription":
        return translate("ui.section.fields.description.ark_audio")
    return translate("ui.section.fields.description")


def _localized_parameter_label(field: ParameterFieldDefinition) -> str:
    label_key = _PARAMETER_LABEL_KEYS.get(field.key)
    return translate(label_key) if label_key is not None else field.label


def _localized_parameter_description(field: ParameterFieldDefinition, label: str) -> str:
    if field.key == "response_format" and field.target_path == ("body", "text", "format"):
        return translate("ui.parameter.response_format_description")
    if field.key == "sparse_embedding":
        return translate("ui.parameter.sparse_embedding_description")
    return translate("ui.field.parameter.target_description", field=label)


def _localized_value_placeholder(field: ParameterFieldDefinition) -> str:
    if field.value_kind == "string_list":
        return '["item1","item2"]'
    if field.value_kind == "json":
        return '{"key":"value"}'
    return translate(f"ui.placeholder.{field.value_kind}")


def _localized_value_hint(field: ParameterFieldDefinition) -> str:
    return translate(f"ui.hint.{field.value_kind}")


def _add_provider_sections(sections: dict[str, dict], provider: ProviderPolicyKey, *, order: int) -> tuple[str, ...]:
    section_names: list[str] = []
    base_section_name = provider
    sections[base_section_name] = _provider_base_section(provider, order=order)
    section_names.append(base_section_name)
    capability_order = order + 1
    for catalog in provider_catalogs(provider):
        policy_name = f"{provider}_{catalog.capability}_policy"
        fields_name = f"{provider}_{catalog.capability}_fields"
        sections[policy_name] = _capability_policy_section(catalog, order=capability_order)
        sections[fields_name] = _capability_fields_section(catalog, order=capability_order + 1)
        section_names.extend((policy_name, fields_name))
        capability_order += 2
    return tuple(section_names)


def _provider_base_section(provider: ProviderPolicyKey, *, order: int) -> dict:
    fields: dict[str, dict] = {}
    current_order = 0
    if "user_agent" in _PROVIDER_BASE_FIELDS[provider]:
        fields["user_agent"] = _field(
            name="user_agent",
            field_type="string",
            label="User-Agent",
            default="",
            ui_type="text",
            placeholder="留空使用 MaiDock 默认 UA",
            order=current_order,
        )
        current_order += 1
    if "force_official_endpoint" in _PROVIDER_BASE_FIELDS[provider]:
        fields["force_official_endpoint"] = _field(
            name="force_official_endpoint",
            field_type="boolean",
            label=_force_endpoint_label(provider),
            default=True,
            ui_type="switch",
            hint=_force_endpoint_hint(provider),
            order=current_order,
        )
        current_order += 1
    if "force_disable_thinking" in _PROVIDER_BASE_FIELDS[provider]:
        fields["force_disable_thinking"] = _field(
            name="force_disable_thinking",
            field_type="boolean",
            label="强制关闭 Mimo 深度思考",
            default=True,
            ui_type="switch",
            hint='开启后强制写入 thinking={"type":"disabled"}。关闭后 MaiDock 会通过工具调用 extra_content 和 SQLite 完整回传历史 reasoning_content。',
            order=current_order,
        )
        current_order += 1
    if "reasoning_retention_days" in _PROVIDER_BASE_FIELDS[provider]:
        fields["reasoning_retention_days"] = _field(
            name="reasoning_retention_days",
            field_type="integer",
            label="工具调用思考保留天数",
            default=30,
            ui_type="number",
            min_value=1,
            max_value=365,
            step=1,
            hint="完整 reasoning_content 会以明文保存在插件数据目录的 SQLite 中；成功使用时刷新过期时间。",
            order=current_order,
        )
        current_order += 1
    if "audio_transcription_prompt" in _PROVIDER_BASE_FIELDS[provider]:
        fields["audio_transcription_prompt"] = _field(
            name="audio_transcription_prompt",
            field_type="string",
            label="转录提示词",
            default="请识别音频中的内容，以文字形式返回识别结果。",
            ui_type="text",
            hint="ARK 使用 Responses input_audio + input_text 完成语音转录。",
            order=current_order,
        )
        current_order += 1
    if "audio_transcription_language" in _PROVIDER_BASE_FIELDS[provider]:
        fields["audio_transcription_language"] = _select_field(
            name="audio_transcription_language",
            label="ASR 识别语言",
            default="auto",
            choices=("auto", "zh", "en"),
            hint="mimo-v2.5-asr 使用；auto=自动检测，zh=中文，en=英文。",
            order=current_order,
        )
        current_order += 1
    if "auto_detect_endpoint" in _PROVIDER_BASE_FIELDS[provider]:
        fields["auto_detect_endpoint"] = _field(
            name="auto_detect_endpoint",
            field_type="boolean",
            label="自动探测模型端点",
            default=True,
            ui_type="switch",
            hint="阿里云百炼多模态模型与纯文本模型使用不同 API 端点。开启后，文本端点返回 url error 时自动切换多模态端点并在内存中记录。",
            order=current_order,
        )
        current_order += 1
    if "prefix_cache_enabled" in _PROVIDER_BASE_FIELDS[provider]:
        fields["prefix_cache_enabled"] = _field(
            name="prefix_cache_enabled",
            field_type="boolean",
            label="启用 ARK 显式前缀缓存",
            default=False,
            ui_type="switch",
            hint="需要 Core 1.0.9，并需先开启方舟“推理（缓存）”计价；仅缓存至少 256 tokens 的开头 system 前缀。",
            order=current_order,
        )
        current_order += 1
    if "prefix_cache_ttl_seconds" in _PROVIDER_BASE_FIELDS[provider]:
        fields["prefix_cache_ttl_seconds"] = _field(
            name="prefix_cache_ttl_seconds",
            field_type="integer",
            label="前缀缓存有效期（秒）",
            default=259200,
            ui_type="number",
            min_value=3600,
            max_value=604800,
            step=3600,
            hint="默认 3 天，最短 1 小时，最长 7 天；使用缓存不会延长有效期。",
            order=current_order,
        )
        current_order += 1
    if "max_retries" in _PROVIDER_BASE_FIELDS[provider]:
        fields["max_retries"] = _field(
            name="max_retries",
            field_type="integer",
            label="最大重试次数",
            default=3,
            ui_type="number",
            min_value=0,
            hint="Provider API 调用失败时的最大重试次数。开关关闭时为回退值（Host 未配置时使用），开关开启时强制覆写 Host 值。",
            order=current_order,
        )
        current_order += 1
    if "force_max_retries" in _PROVIDER_BASE_FIELDS[provider]:
        fields["force_max_retries"] = _field(
            name="force_max_retries",
            field_type="boolean",
            label="强制覆写 Host 重试次数",
            default=False,
            ui_type="switch",
            hint="关闭：回退模式，Host 提供值时优先使用 Host 值；开启：始终使用上方配置值。",
            order=current_order,
        )
        current_order += 1
    if "retry_interval" in _PROVIDER_BASE_FIELDS[provider]:
        fields["retry_interval"] = _field(
            name="retry_interval",
            field_type="number",
            label="重试间隔（秒）",
            default=5.0,
            ui_type="number",
            min_value=0,
            step=0.5,
            hint="两次重试之间的等待时间。开关关闭时为回退值，开关开启时强制覆写 Host 值。",
            order=current_order,
        )
        current_order += 1
    if "force_retry_interval" in _PROVIDER_BASE_FIELDS[provider]:
        fields["force_retry_interval"] = _field(
            name="force_retry_interval",
            field_type="boolean",
            label="强制覆写 Host 重试间隔",
            default=False,
            ui_type="switch",
            hint="关闭：回退模式，Host 提供值时优先使用 Host 值；开启：始终使用上方配置值。",
            order=current_order,
        )
        current_order += 1
    return _section(
        name=provider,
        title=f"{PROVIDER_TITLES[provider]} 基础设置",
        description="Provider 通用连接设置。",
        icon=_PROVIDER_ICONS[provider],
        order=order,
        fields=fields,
    )


def _capability_policy_section(catalog: CapabilityParameterCatalog, *, order: int) -> dict:
    path = f"{catalog.provider}.{catalog.capability}"
    return _section(
        name=path,
        title=_capability_policy_title(catalog.capability),
        description=_capability_policy_description(catalog),
        icon="sliders-horizontal",
        order=order,
        fields={
            "accept_model_extra_params": _field(
                name="accept_model_extra_params",
                field_type="boolean",
                label="接收模型配置 extra_params",
                default=True,
                ui_type="switch",
                order=0,
            ),
            "accept_request_extra_params": _field(
                name="accept_request_extra_params",
                field_type="boolean",
                label="接收单次请求 extra_params",
                default=True,
                ui_type="switch",
                order=1,
            ),
            "unknown_extra_params": _select_field(
                name="unknown_extra_params",
                label="未知 extra_params 策略",
                default="forward",
                choices=("forward", "drop", "reject"),
                hint="forward：发送未知字段到 Provider API；drop：静默丢弃未知字段；reject：遇到未知字段时报错。",
                order=2,
            ),
        },
    )


def _capability_policy_description(catalog: CapabilityParameterCatalog) -> str:
    """生成能力参数策略说明。"""
    if catalog.provider == "xiaomi_mimo" and catalog.capability == "audio_transcription":
        return "控制 Mimo ASR 请求的 extra_params。"
    if catalog.provider == "volcengine_ark" and catalog.capability == "audio_transcription":
        return "控制 ARK Responses input_audio 转录请求的 extra_params。"
    return "控制模型配置与单次请求传入的 extra_params 是否被 MaiDock 接收，以及未知字段如何处理。"


def _capability_fields_section(catalog: CapabilityParameterCatalog, *, order: int) -> dict:
    fields: dict[str, dict] = {}
    for field in catalog.fields:
        field_order = field.order * 10
        enabled_key = field_enabled_key(field)
        override_enabled_key = field_override_enabled_key(field)
        override_value_key = field_override_value_key(field)
        fields[enabled_key] = _field(
            name=enabled_key,
            field_type="boolean",
            label=f"发送「{field.label}」到 Provider API",
            default=True,
            ui_type="switch",
            hint=f"{field.description}。关闭后会丢弃 Host 传入的该字段，并从发送给 Provider API 的请求中移除对应参数。",
            order=field_order,
        )
        fields[override_enabled_key] = _field(
            name=override_enabled_key,
            field_type="boolean",
            label=f"覆写「{field.label}」",
            default=False,
            ui_type="switch",
            hint="开启后使用下方值强制覆写发送给 Provider API 的参数值。",
            order=field_order + 1,
        )
        if field.value_kind == "boolean":
            fields[override_value_key] = _field(
                name=override_value_key,
                field_type="boolean",
                label=f"「{field.label}」覆写值",
                default=False,
                ui_type="switch",
                hint=field.description,
                order=field_order + 2,
            )
        else:
            fields[override_value_key] = _field(
                name=override_value_key,
                field_type="string",
                label=f"「{field.label}」覆写值",
                default="",
                ui_type="textarea",
                placeholder=_placeholder_for_value_kind(field),
                hint=_hint_for_value_kind(field),
                order=field_order + 2,
                rows=3,
            )
    return _section(
        name=f"{catalog.provider}.{catalog.capability}.fields",
        title=_capability_fields_title(catalog.capability),
        description=_capability_fields_description(catalog),
        icon="list-checks",
        order=order,
        fields=fields,
        collapsed=True,
    )


def _capability_fields_description(catalog: CapabilityParameterCatalog) -> str:
    """生成能力字段开关说明。"""
    if catalog.provider == "xiaomi_mimo" and catalog.capability == "audio_transcription":
        return "Mimo ASR 使用 input_audio 与 asr_options；格式字段用于校验并构造 data URL，同时作为 input_audio.format 发送。"
    if catalog.provider == "volcengine_ark" and catalog.capability == "audio_transcription":
        return "使用 Responses input_audio + input_text；格式字段只用于内部构造 data URL。"
    return "每个文档字段默认发送到 Provider API；关闭开关会丢弃 Host 传入值，开启覆写会强制替换发送给 Provider API 的参数。"


def _capability_policy_title(capability: CapabilityKey) -> str:
    title = CAPABILITY_TITLES[capability]
    if capability == "embeddings":
        return f"{title} 参数策略"
    return f"{title}参数策略"


def _capability_fields_title(capability: CapabilityKey) -> str:
    title = CAPABILITY_TITLES[capability]
    if capability == "embeddings":
        return f"{title} 字段开关与覆写"
    return f"{title}字段开关与覆写"


def _provider_tab_title(provider: ProviderPolicyKey) -> str:
    if provider in {"openai_responses", "anthropic_messages"}:
        return f"通用协议客户端 · {PROVIDER_TITLES[provider]}"
    return PROVIDER_TITLES[provider]


def _force_endpoint_label(provider: ProviderPolicyKey) -> str:
    match provider:
        case "volcengine_ark":
            return "使用火山方舟原生 endpoint"
        case "dashscope":
            return "使用阿里云百炼 DashScope 原生 endpoint"
        case "siliconflow":
            return "使用 SiliconFlow 官方 endpoint"
        case _:
            return "使用 Provider 原生 endpoint"


def _force_endpoint_hint(provider: ProviderPolicyKey) -> str:
    match provider:
        case "volcengine_ark":
            return "开启后忽略 Host 提供的 base_url，改用火山方舟原生 API endpoint。"
        case "dashscope":
            return "开启后忽略 Host 提供的 base_url，改用阿里云百炼 DashScope 原生 API endpoint。"
        case "siliconflow":
            return "开启后忽略 Host 提供的 base_url，改用 SiliconFlow 官方 API endpoint。"
        case _:
            return "开启后忽略 Host 提供的 base_url，改用 Provider 原生 API endpoint。"


def _placeholder_for_value_kind(field: ParameterFieldDefinition) -> str:
    match field.value_kind:
        case "string":
            return "例如：auto"
        case "integer":
            return "例如：1024"
        case "number":
            return "例如：0.8"
        case "boolean":
            return "true 或 false"
        case "string_list":
            return '["item1","item2"]'
        case "json":
            return '{"key":"value"}'


def _hint_for_value_kind(field: ParameterFieldDefinition) -> str:
    match field.value_kind:
        case "string":
            return "字符串字段直接输入文本，不需要 JSON 引号。"
        case "integer":
            return "整数覆写值，例如 1024。"
        case "number":
            return "数字覆写值，例如 0.8。"
        case "boolean":
            return "布尔覆写值：true 或 false。"
        case "string_list":
            return '字符串数组 JSON，例如 ["a","b"]。'
        case "json":
            return "输入合法 JSON 值；对象、数组、字符串、数字、布尔值均可。"


def _number_field(name: str, label: str, default: int, order: int) -> dict:
    return _field(
        name=name,
        field_type="integer",
        label=label,
        default=default,
        ui_type="number",
        min_value=0,
        order=order,
    )


def _select_field(
    *,
    name: str,
    label: str,
    default: str,
    choices: tuple[str, ...],
    order: int,
    hint: str = "",
) -> dict:
    return _field(
        name=name,
        field_type="string",
        label=label,
        default=default,
        ui_type="select",
        choices=choices,
        hint=hint,
        order=order,
    )


def _field(
    *,
    name: str,
    field_type: str,
    label: str,
    default: object,
    ui_type: str,
    order: int,
    description: str = "",
    placeholder: str = "",
    hint: str = "",
    choices: tuple[str, ...] = (),
    min_value: int | None = None,
    max_value: int | None = None,
    rows: int = 3,
    disabled: bool = False,
    step: float = 1.0,
) -> dict:
    field: dict = {
        "name": name,
        "type": field_type,
        "default": default,
        "description": description or label,
        "required": False,
        "choices": list(choices),
        "min": min_value,
        "max": max_value,
        "step": step,
        "pattern": None,
        "max_length": None,
        "label": label,
        "placeholder": placeholder,
        "hint": hint,
        "hidden": False,
        "disabled": disabled,
        "order": order,
        "input_type": None,
        "ui_type": ui_type,
        "rows": rows,
        "group": None,
        "depends_on": None,
        "depends_value": None,
        "item_type": None,
        "item_fields": None,
        "min_items": None,
        "max_items": None,
    }
    return field


def _section(
    *,
    name: str,
    title: str,
    description: str,
    icon: str,
    order: int,
    fields: dict[str, dict],
    collapsed: bool = False,
) -> dict:
    return {
        "name": name,
        "title": title,
        "description": description,
        "icon": icon,
        "collapsed": collapsed,
        "order": order,
        "fields": fields,
    }


def _tab(tab_id: str, title: str, sections: tuple[str, ...], *, icon: str, order: int) -> dict:
    return {
        "id": tab_id,
        "title": title,
        "sections": list(sections),
        "icon": icon,
        "order": order,
    }
