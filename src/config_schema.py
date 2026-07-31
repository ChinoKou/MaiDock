from typing import cast

from .core.parameter_catalog import (
    CAPABILITY_TITLES,
    PROVIDER_TITLES,
    CapabilityParameterCatalog,
    ParameterFieldDefinition,
    dotted_path,
    get_parameter_catalog,
    provider_catalogs,
)
from .core.parameter_policy import CapabilityKey, ProviderPolicyKey
from .i18n import DEFAULT_LOCALE, Locale, translate, use_locale
from .public_api.catalog import PublicApiWebUiField
from .public_api.providers import PUBLIC_API_CONFIG_CATALOG
from .version import __version__

_PROVIDER_ICONS: dict[ProviderPolicyKey, str] = {
    "openai_responses": "bot",
    "anthropic_messages": "bot-message-square",
    "dashscope": "bot",
    "bailian_responses": "bot",
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
        "builtin_endpoint_mode",
        "prefix_cache_enabled",
        "prefix_cache_ttl_seconds",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "xiaomi_mimo": (
        "user_agent",
        "reasoning_retention_days",
        "max_retries",
        "force_max_retries",
        "retry_interval",
        "force_retry_interval",
    ),
    "bailian_responses": (
        "user_agent",
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
    "builtin_endpoint_mode": ("ui.field.ark_endpoint_mode.label", "ui.field.ark_endpoint_mode.hint"),
    "prefix_cache_enabled": ("ui.field.prefix_cache.label", "ui.field.prefix_cache.hint"),
    "prefix_cache_ttl_seconds": ("ui.field.prefix_cache_ttl.label", "ui.field.prefix_cache_ttl.hint"),
    "max_retries": ("ui.field.max_retries.label", "ui.field.max_retries.hint"),
    "force_max_retries": ("ui.field.force_max_retries.label", "ui.field.force_value.hint"),
    "retry_interval": ("ui.field.retry_interval.label", "ui.field.retry_interval.hint"),
    "force_retry_interval": ("ui.field.force_retry_interval.label", "ui.field.force_value.hint"),
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
    public_api_sections = _add_public_api_sections(sections)
    tabs.append(
        _tab(
            "public_api",
            "跨插件 API",
            public_api_sections,
            icon="share-2",
            order=5,
        )
    )
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
        if tab_id == "public_api":
            tab["title"] = translate("ui.public_api.tab.title")
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
    if section_name.startswith("public_api"):
        return
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
    if section_name.endswith(".overrides"):
        title_key = "ui.section.overrides.title_spaced" if capability == "embeddings" else "ui.section.overrides.title"
        section["title"] = translate(title_key, capability=capability_title)
        section["description"] = _localized_overrides_description(provider, capability)
        return
    section["title"] = translate("ui.section.overrides.title", capability=capability_title)


def _localize_section_fields(section_name: str, fields: dict[str, dict]) -> None:
    if section_name.startswith("public_api"):
        return
    for field_name, field_schema in fields.items():
        static_keys = _STATIC_FIELD_KEYS.get(field_name)
        if static_keys is not None:
            label_key, hint_key = static_keys
            _set_field_text(field_schema, label_key, hint_key)

    if section_name in _PROVIDER_BASE_FIELDS:
        _localize_provider_base_fields(cast(ProviderPolicyKey, section_name), fields)
        return
    if not section_name.endswith(".overrides"):
        return

    provider_name, capability_name, _ = section_name.split(".")
    catalog = get_parameter_catalog(
        cast(ProviderPolicyKey, provider_name),
        cast(CapabilityKey, capability_name),
    )
    for parameter in catalog.fields:
        localized_label = _localized_parameter_label(parameter)
        description = _localized_parameter_description(parameter, localized_label)
        override_value = fields[parameter.config_key]
        override_value["label"] = f"{parameter.key} · {_parameter_type_label(parameter)}"
        override_value["description"] = override_value["label"]
        override_value["hint"] = translate(
            "ui.field.parameter.help",
            description=description,
            target=dotted_path(parameter.target_path),
            format=translate(f"ui.hint.{parameter.value_kind}"),
            default_behavior=_parameter_default_behavior(parameter),
            constraints=parameter.constraints or translate("ui.field.parameter.constraints.provider"),
            documentation_url=catalog.documentation_url,
        )
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
        del fields["audio_transcription_prompt"]


def _localized_provider_title(provider: ProviderPolicyKey) -> str:
    return translate(f"ui.provider.{provider}")


def _localized_capability_title(capability: CapabilityKey) -> str:
    return translate(f"ui.capability.{capability}")


def _localized_overrides_description(provider: ProviderPolicyKey, capability: CapabilityKey) -> str:
    if provider == "xiaomi_mimo" and capability == "audio_transcription":
        return translate("ui.section.overrides.description.mimo_audio")
    if provider == "volcengine_ark" and capability == "audio_transcription":
        return translate("ui.section.overrides.description.ark_audio")
    return translate("ui.section.overrides.description")


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


def _parameter_type_label(field: ParameterFieldDefinition) -> str:
    return {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "json": "JSON",
        "string_list": "string[]",
    }[field.value_kind]


def _parameter_default_behavior(field: ParameterFieldDefinition) -> str:
    if field.default_text.strip():
        return translate("ui.field.parameter.default.configured", value=field.default_text)
    return translate("ui.field.parameter.default.blank")


def _add_provider_sections(sections: dict[str, dict], provider: ProviderPolicyKey, *, order: int) -> tuple[str, ...]:
    section_names: list[str] = []
    base_section_name = provider
    sections[base_section_name] = _provider_base_section(provider, order=order)
    section_names.append(base_section_name)
    capability_order = order + 1
    for catalog in provider_catalogs(provider):
        overrides_name = f"{provider}_{catalog.capability}_overrides"
        sections[overrides_name] = _capability_overrides_section(catalog, order=capability_order)
        section_names.append(overrides_name)
        capability_order += 1
    return tuple(section_names)


def _add_public_api_sections(sections: dict[str, dict]) -> tuple[str, ...]:
    section_names = ["public_api", "public_api_resources"]
    sections["public_api"] = _section(
        name="public_api",
        title=translate("ui.public_api.section.settings.title"),
        description=translate("ui.public_api.section.settings.description"),
        icon="share-2",
        order=5,
        fields={
            "enabled": _field(
                name="enabled",
                field_type="boolean",
                label=translate("ui.public_api.field.enabled"),
                default=False,
                ui_type="switch",
                order=0,
            ),
            "default_image_profile": _field(
                name="default_image_profile",
                field_type="string",
                label=translate("ui.public_api.field.default_image_profile"),
                default="",
                ui_type="text",
                order=1,
            ),
            "default_video_profile": _field(
                name="default_video_profile",
                field_type="string",
                label=translate("ui.public_api.field.default_video_profile"),
                default="",
                ui_type="text",
                order=2,
            ),
        },
    )
    resource_fields = (
        ("max_concurrent_jobs", 2, 1, 32),
        ("max_queued_jobs", 32, 1, 1024),
        ("max_upload_mb", 512, 1, 4096),
        ("max_artifact_mb", 512, 1, 4096),
        ("storage_quota_gb", 10, 1, 1024),
        ("incomplete_upload_ttl_hours", 24, 1, 168),
        ("completed_upload_ttl_days", 7, 1, 90),
        ("artifact_ttl_days", 7, 1, 90),
        ("job_metadata_ttl_days", 30, 1, 365),
        ("max_tracking_hours", 23, 1, 72),
    )
    sections["public_api_resources"] = _section(
        name="public_api.resources",
        title=translate("ui.public_api.section.resources.title"),
        description=translate("ui.public_api.section.resources.description"),
        icon="database",
        order=6,
        fields={
            name: _field(
                name=name,
                field_type="integer",
                label=translate(f"ui.public_api.field.{name}"),
                default=default,
                ui_type="number",
                order=order,
                min_value=minimum,
                max_value=maximum,
            )
            for order, (name, default, minimum, maximum) in enumerate(resource_fields)
        },
    )
    for entry in PUBLIC_API_CONFIG_CATALOG:
        section_key = f"public_api_{entry.provider_key}"
        profiles = _field(
            name="profiles",
            field_type="array",
            label=translate("ui.public_api.field.profiles"),
            default=[],
            ui_type="list",
            order=0,
        )
        profiles["item_type"] = "object"
        profiles["item_fields"] = {field.name: _public_catalog_field(field) for field in entry.build_webui_fields()}
        sections[section_key] = _section(
            name=entry.config_path,
            title=translate(entry.title_key),
            description=translate("ui.public_api.section.profiles.description"),
            icon=entry.icon,
            order=entry.order,
            fields={"profiles": profiles},
        )
        section_names.append(section_key)
    return tuple(section_names)


def _public_catalog_field(definition: PublicApiWebUiField) -> dict:
    field = _field(
        name=definition.name,
        field_type=definition.field_type,
        label=translate(definition.label_key),
        default=definition.default,
        ui_type=definition.ui_type,
        order=definition.order,
        hint=translate(definition.hint_key) if definition.hint_key else "",
        choices=definition.choices,
        rows=definition.rows,
        step=definition.step,
    )
    field["min"] = definition.minimum
    field["max"] = definition.maximum
    field["item_type"] = definition.item_type
    if definition.item_fields:
        field["item_fields"] = {item.name: _public_catalog_field(item) for item in definition.item_fields}
    return field


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
    if "builtin_endpoint_mode" in _PROVIDER_BASE_FIELDS[provider]:
        # depends_on 是 Host ConfigField 定义的条件显示槽位；当前 dashboard 尚未消费，
        # 先按契约填上做前向兼容，hint 里同时写清生效条件。
        fields["builtin_endpoint_mode"] = _select_field(
            name="builtin_endpoint_mode",
            label="内置端点类型",
            default="standard",
            choices=("standard", "agent_plan", "coding_plan"),
            hint=(
                "standard=按量付费 /api/v3；agent_plan=Agent Plan 订阅 /api/plan/v3（需其专属 API Key）；"
                "coding_plan=Coding Plan 订阅 /api/coding/v3。仅在开启原生 endpoint 时生效；"
                "选择订阅端点时前缀缓存自动停用。"
            ),
            order=current_order,
            depends_on=f"{provider}.force_official_endpoint",
            depends_value=True,
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


def _capability_overrides_section(catalog: CapabilityParameterCatalog, *, order: int) -> dict:
    """渲染参数覆写目录：每个参数一个跨双列全宽 textarea。"""

    fields: dict[str, dict] = {}
    for field in catalog.fields:
        fields[field.config_key] = _field(
            name=field.config_key,
            field_type="string",
            label=f"{field.label} · {field.value_kind}",
            default=field.default_text,
            ui_type="textarea",
            hint=field.description,
            order=field.order * 10,
            rows=3,
        )
    return _section(
        name=f"{catalog.provider}.{catalog.capability}.overrides",
        title=_capability_overrides_title(catalog.capability),
        description=_capability_overrides_description(catalog),
        icon="sliders-horizontal",
        order=order,
        fields=fields,
        collapsed=True,
    )


def _capability_overrides_description(catalog: CapabilityParameterCatalog) -> str:
    """生成能力参数覆写说明。"""
    if catalog.provider == "xiaomi_mimo" and catalog.capability == "audio_transcription":
        return "Mimo ASR 使用 input_audio 与 asr_options；格式字段用于校验并构造 data URL，同时作为 input_audio.format 发送。"
    if catalog.provider == "volcengine_ark" and catalog.capability == "audio_transcription":
        return "使用 Responses input_audio + input_text；格式字段只用于内部构造 data URL。"
    return "覆写值拥有最高优先级：空白表示不覆写，布尔使用 true/false，数字使用 JSON 数字，数组和对象使用合法 JSON。"


def _capability_overrides_title(capability: CapabilityKey) -> str:
    title = CAPABILITY_TITLES[capability]
    if capability == "embeddings":
        return f"{title} 参数覆写"
    return f"{title}参数覆写"


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
    depends_on: str | None = None,
    depends_value: object = None,
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
        depends_on=depends_on,
        depends_value=depends_value,
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
    depends_on: str | None = None,
    depends_value: object = None,
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
        "depends_on": depends_on,
        "depends_value": depends_value,
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
