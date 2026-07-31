import json
import math
from collections.abc import Mapping
from typing import Literal

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator

from .config_migration import migrate_legacy_config
from .core.common import (
    ImageProcessingLimits,
    InvalidImagePolicy,
    ProviderRuntimeOptions,
    normalize_ark_builtin_endpoint_mode,
)
from .core.json_types import (
    JsonValue,
    is_json_list,
    json_mapping_or_none,
    mapping_to_json_object,
    normalize_json_value,
)
from .core.parameter_catalog import (
    CapabilityParameterCatalog,
    get_parameter_catalog,
    iter_parameter_catalogs,
)
from .core.parameter_policy import (
    ParameterOverrideRegistry,
    ParameterOverrideSet,
    ProviderCapabilityOverrides,
)
from .core.parsing import (
    normalize_reasoning_parse_mode,
    normalize_tool_argument_parse_mode,
)
from .i18n import (
    DEFAULT_LOCALE,
    Locale,
    normalize_locale,
    runtime_expected,
    translate,
)
from .public_api.config import PublicApiConfig
from .version import DEFAULT_USER_AGENT, __version__


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    locale: Literal["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"] = Field(
        default=DEFAULT_LOCALE,
        description="MaiDock display, log, and error locale",
    )

    @field_validator("locale", mode="before")
    @classmethod
    def validate_locale(cls, value: object) -> Locale:
        return normalize_locale(value)

    enabled: bool = Field(default=True, description="是否启用 MaiDock")
    config_version: str = Field(default=__version__, description="配置版本")


class DiagnosticsConfig(PluginConfigBase):
    """诊断配置。"""

    __ui_label__ = "诊断"
    __ui_icon__ = "bug"
    __ui_order__ = 1

    include_raw_data: bool = Field(default=False, description="是否把 Provider API 响应摘要放入 raw_data")
    log_payload_summary: bool = Field(default=True, description="是否记录脱敏后的请求/响应摘要日志")
    log_payload_debug: bool = Field(default=False, description="是否记录脱敏后的详细请求载荷")


class CapabilityParameterOverridesConfig(PluginConfigBase):
    """Provider 能力级参数覆写配置。"""

    overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "参数覆写值；空白表示不覆写，布尔使用 true/false，整数和浮点数使用 JSON 数字，数组和对象使用合法 JSON"
        ),
    )

    @field_validator("overrides", mode="before")
    @classmethod
    def validate_overrides(cls, value: object) -> dict[str, str]:
        if value is None:
            return {}
        mapping = json_mapping_or_none(value)
        if mapping is None:
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject="overrides",
                    expected=runtime_expected("object"),
                    actual=type(value).__name__,
                )
            )
        normalized: dict[str, str] = {}
        for key, item in mapping.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            if isinstance(item, str):
                normalized[normalized_key] = item
                continue
            if isinstance(item, bool):
                normalized[normalized_key] = "true" if item else "false"
                continue
            if isinstance(item, (int, float)):
                normalized[normalized_key] = str(item)
                continue
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"overrides.{normalized_key}",
                    expected=runtime_expected("string"),
                    actual=type(item).__name__,
                )
            )
        return normalized


class OpenAIResponsesConfig(PluginConfigBase):
    """OpenAI Responses Provider 配置。"""

    __ui_label__ = "OpenAI Responses"
    __ui_icon__ = "bot"
    __ui_order__ = 2

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    response: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    embeddings: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    audio_transcription: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class AnthropicMessagesConfig(PluginConfigBase):
    """Anthropic Messages Provider 配置。"""

    __ui_label__ = "Anthropic Messages"
    __ui_icon__ = "bot-message-square"
    __ui_order__ = 3

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    chat_completion: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class DashScopeConfig(PluginConfigBase):
    """阿里云百炼 DashScope Provider 配置。"""

    __ui_label__ = "阿里云百炼 DashScope"
    __ui_icon__ = "bot"
    __ui_order__ = 4

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    force_official_endpoint: bool = Field(
        default=True,
        description="是否忽略 Host 提供的 base_url，改用阿里云百炼 DashScope 原生 endpoint",
    )
    auto_detect_endpoint: bool = Field(
        default=True,
        description="是否自动探测模型端点：文本端点返回 url error 时自动切换多模态端点",
    )
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    chat_completion: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    embeddings: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    audio_transcription: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class SiliconFlowConfig(PluginConfigBase):
    """SiliconFlow Provider 配置。"""

    __ui_label__ = "SiliconFlow"
    __ui_icon__ = "bot"
    __ui_order__ = 5

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    force_official_endpoint: bool = Field(
        default=True,
        description="是否忽略 Host 提供的 base_url，改用 SiliconFlow 官方 endpoint",
    )
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    chat_completion: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    embeddings: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    audio_transcription: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class VolcengineArkConfig(PluginConfigBase):
    """Volcengine Ark Provider 配置。"""

    __ui_label__ = "Volcengine Ark"
    __ui_icon__ = "bot"
    __ui_order__ = 6

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    force_official_endpoint: bool = Field(
        default=True,
        description="是否忽略 Host 提供的 base_url，改用火山方舟原生 endpoint",
    )
    builtin_endpoint_mode: Literal["standard", "agent_plan", "coding_plan"] = Field(
        default="standard",
        description=(
            "内置端点类型：standard=按量付费 /api/v3，agent_plan=Agent Plan 订阅 /api/plan/v3，"
            "coding_plan=Coding Plan 订阅 /api/coding/v3；仅在开启原生 endpoint 时生效"
        ),
    )

    @field_validator("builtin_endpoint_mode", mode="before")
    @classmethod
    def validate_builtin_endpoint_mode(cls, value: object) -> str:
        return normalize_ark_builtin_endpoint_mode(value)

    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    prefix_cache_enabled: bool = Field(
        default=False,
        description="启用 ARK Responses 显式前缀缓存；缓存存储会产生费用",
    )
    prefix_cache_ttl_seconds: int = Field(
        default=259200,
        ge=3600,
        le=604800,
        description="前缀缓存有效期秒数；范围 3600..604800",
    )
    response: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    embeddings: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    audio_transcription: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class BailianResponsesConfig(PluginConfigBase):
    """阿里云百炼 Responses Provider 配置。"""

    __ui_label__ = "阿里云百炼 Responses"
    __ui_icon__ = "bot"
    __ui_order__ = 7

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    response: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class XiaomiMimoConfig(PluginConfigBase):
    """Xiaomi Mimo Provider 配置。"""

    __ui_label__ = "Xiaomi Mimo"
    __ui_icon__ = "bot"
    __ui_order__ = 8

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    reasoning_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Mimo 工具调用 reasoning_content 的本地保留天数；范围 1..365",
    )
    max_retries: int = Field(
        default=3,
        description="最大重试次数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_max_retries: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    retry_interval: float = Field(
        default=5.0,
        description="重试间隔秒数（关闭下方开关时作为回退值，开启时强制覆写 Host 值）",
    )
    force_retry_interval: bool = Field(default=False, description="关闭=回退模式，开启=强制使用上方配置值")
    chat_completion: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)
    audio_transcription: CapabilityParameterOverridesConfig = Field(default_factory=CapabilityParameterOverridesConfig)


class CompatibilityConfig(PluginConfigBase):
    """兼容性配置。"""

    __ui_label__ = "兼容性"
    __ui_icon__ = "settings-2"
    __ui_order__ = 8

    tool_argument_parse_mode: str = Field(
        default="auto", description="工具参数解析模式：auto/strict/repair/double_decode"
    )
    reasoning_parse_mode: str = Field(default="auto", description="推理内容解析模式：auto/native/think_tag/none")
    invalid_image_policy: str = Field(default="placeholder", description="无效图片处理策略：placeholder/skip/error")
    max_image_bytes_mb: int = Field(default=30, description="单张图片解码后最大字节数（MB）")
    max_image_pixels: int = Field(default=25_000_000, description="单张图片最大像素数量")
    max_image_dimension: int = Field(default=8192, description="单张图片单边最大像素")
    max_image_frames: int = Field(default=64, description="动图最大帧数")


class MaiDockConfig(PluginConfigBase):
    """MaiDock 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    openai_responses: OpenAIResponsesConfig = Field(default_factory=OpenAIResponsesConfig)
    anthropic_messages: AnthropicMessagesConfig = Field(default_factory=AnthropicMessagesConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    bailian_responses: BailianResponsesConfig = Field(default_factory=BailianResponsesConfig)
    siliconflow: SiliconFlowConfig = Field(default_factory=SiliconFlowConfig)
    volcengine_ark: VolcengineArkConfig = Field(default_factory=VolcengineArkConfig)
    xiaomi_mimo: XiaomiMimoConfig = Field(default_factory=XiaomiMimoConfig)
    public_api: PublicApiConfig = Field(default_factory=PublicApiConfig)
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)


def normalize_invalid_image_policy(raw_policy: str) -> InvalidImagePolicy:
    """规范化无效图片处理策略。"""

    if raw_policy == "skip":
        return "skip"
    if raw_policy == "error":
        return "error"
    return "placeholder"


def normalize_user_agent(raw_user_agent: str | None) -> str:
    """规范化 Provider User-Agent，留空时使用默认 UA。"""

    normalized = (raw_user_agent or "").strip()
    return normalized or DEFAULT_USER_AGENT


def positive_int(value: object, default: int) -> int:
    """读取正整数配置，非法时回退到默认值。"""

    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, float) and not isinstance(value, bool) and value > 0 and value == int(value):
        return int(value)
    return default


def build_image_limits(config: CompatibilityConfig) -> ImageProcessingLimits:
    """根据兼容性配置构造图片处理限制。"""

    limits = ImageProcessingLimits()
    max_decoded_bytes = positive_int(config.max_image_bytes_mb, 30) * 1024 * 1024
    return ImageProcessingLimits(
        max_base64_chars=((max_decoded_bytes + 2) // 3) * 4,
        max_decoded_bytes=max_decoded_bytes,
        max_pixels=positive_int(config.max_image_pixels, limits.max_pixels),
        max_dimension=positive_int(config.max_image_dimension, limits.max_dimension),
        max_frames=positive_int(config.max_image_frames, limits.max_frames),
    )


def build_parameter_overrides(
    config: CapabilityParameterOverridesConfig,
    catalog: CapabilityParameterCatalog,
) -> ParameterOverrideSet:
    """把配置中的字符串覆写值解析为类型化的目录覆写集合。"""

    values: dict[str, JsonValue] = {}
    for raw_key, raw_value in config.overrides.items():
        field = catalog.field_by_safe_key(raw_key)
        if field is None:
            raise ValueError(
                translate(
                    "config.error.unknown_override",
                    provider=catalog.provider,
                    capability=catalog.capability,
                    field=raw_key,
                )
            )
        value = parse_override_value(
            raw_value=raw_value,
            value_kind=field.value_kind,
            field_label=f"{catalog.provider}.{catalog.capability}.{field.key}",
        )
        if value is not None:
            values[field.key] = value
    return ParameterOverrideSet(values=values)


def build_parameter_overrides_registry(config: MaiDockConfig) -> ParameterOverrideRegistry:
    """根据插件配置构造 provider/capability 参数覆写注册表。"""

    return ParameterOverrideRegistry(
        openai_responses=ProviderCapabilityOverrides(
            response=build_parameter_overrides(
                config.openai_responses.response,
                get_parameter_catalog("openai_responses", "response"),
            ),
            embeddings=build_parameter_overrides(
                config.openai_responses.embeddings,
                get_parameter_catalog("openai_responses", "embeddings"),
            ),
            audio_transcription=build_parameter_overrides(
                config.openai_responses.audio_transcription,
                get_parameter_catalog("openai_responses", "audio_transcription"),
            ),
        ),
        anthropic_messages=ProviderCapabilityOverrides(
            chat_completion=build_parameter_overrides(
                config.anthropic_messages.chat_completion,
                get_parameter_catalog("anthropic_messages", "chat_completion"),
            ),
        ),
        dashscope=ProviderCapabilityOverrides(
            chat_completion=build_parameter_overrides(
                config.dashscope.chat_completion,
                get_parameter_catalog("dashscope", "chat_completion"),
            ),
            embeddings=build_parameter_overrides(
                config.dashscope.embeddings,
                get_parameter_catalog("dashscope", "embeddings"),
            ),
            audio_transcription=build_parameter_overrides(
                config.dashscope.audio_transcription,
                get_parameter_catalog("dashscope", "audio_transcription"),
            ),
        ),
        bailian_responses=ProviderCapabilityOverrides(
            response=build_parameter_overrides(
                config.bailian_responses.response,
                get_parameter_catalog("bailian_responses", "response"),
            ),
        ),
        siliconflow=ProviderCapabilityOverrides(
            chat_completion=build_parameter_overrides(
                config.siliconflow.chat_completion,
                get_parameter_catalog("siliconflow", "chat_completion"),
            ),
            embeddings=build_parameter_overrides(
                config.siliconflow.embeddings,
                get_parameter_catalog("siliconflow", "embeddings"),
            ),
            audio_transcription=build_parameter_overrides(
                config.siliconflow.audio_transcription,
                get_parameter_catalog("siliconflow", "audio_transcription"),
            ),
        ),
        volcengine_ark=ProviderCapabilityOverrides(
            response=build_parameter_overrides(
                config.volcengine_ark.response,
                get_parameter_catalog("volcengine_ark", "response"),
            ),
            embeddings=build_parameter_overrides(
                config.volcengine_ark.embeddings,
                get_parameter_catalog("volcengine_ark", "embeddings"),
            ),
            audio_transcription=build_parameter_overrides(
                config.volcengine_ark.audio_transcription,
                get_parameter_catalog("volcengine_ark", "audio_transcription"),
            ),
        ),
        xiaomi_mimo=ProviderCapabilityOverrides(
            chat_completion=build_parameter_overrides(
                config.xiaomi_mimo.chat_completion,
                get_parameter_catalog("xiaomi_mimo", "chat_completion"),
            ),
            audio_transcription=build_parameter_overrides(
                config.xiaomi_mimo.audio_transcription,
                get_parameter_catalog("xiaomi_mimo", "audio_transcription"),
            ),
        ),
    )


def normalize_maidock_config_data(
    config_data: Mapping[str, JsonValue],
) -> tuple[dict, bool]:
    """迁移旧配置并填充所有能力参数目录的覆写默认值。"""

    original = mapping_to_json_object(config_data)
    migrated, _ = migrate_legacy_config(original)
    for catalog in iter_parameter_catalogs():
        provider_section, _ = _ensure_object_section(migrated, catalog.provider)
        capability_section, _ = _ensure_object_section(provider_section, catalog.capability)
        overrides_section, _ = _ensure_object_section(capability_section, "overrides")
        _validate_override_keys(overrides_section, catalog)
        _fill_override_defaults(overrides_section, catalog)

    validated = MaiDockConfig.model_validate(migrated)
    normalized = mapping_to_json_object(validated.model_dump(mode="python"))
    return normalized, normalized != original


def build_runtime_options(
    config: MaiDockConfig | None = None,
) -> ProviderRuntimeOptions:
    """根据插件配置构造 Provider 运行时选项。"""

    if config is None:
        return ProviderRuntimeOptions()
    invalid_image_policy = normalize_invalid_image_policy(config.compatibility.invalid_image_policy)
    return ProviderRuntimeOptions(
        locale=config.plugin.locale,
        include_raw_data=bool(config.diagnostics.include_raw_data),
        log_payload_summary=bool(config.diagnostics.log_payload_summary),
        log_payload_debug=bool(config.diagnostics.log_payload_debug),
        tool_argument_parse_mode=normalize_tool_argument_parse_mode(config.compatibility.tool_argument_parse_mode),
        reasoning_parse_mode=normalize_reasoning_parse_mode(config.compatibility.reasoning_parse_mode),
        invalid_image_policy=invalid_image_policy,
        openai_user_agent=normalize_user_agent(config.openai_responses.user_agent),
        anthropic_user_agent=normalize_user_agent(config.anthropic_messages.user_agent),
        volcengine_user_agent=normalize_user_agent(config.volcengine_ark.user_agent),
        dashscope_user_agent=normalize_user_agent(config.dashscope.user_agent),
        bailian_user_agent=normalize_user_agent(config.bailian_responses.user_agent),
        siliconflow_user_agent=normalize_user_agent(config.siliconflow.user_agent),
        mimo_user_agent=normalize_user_agent(config.xiaomi_mimo.user_agent),
        volcengine_force_official_endpoint=bool(config.volcengine_ark.force_official_endpoint),
        volcengine_builtin_endpoint_mode=config.volcengine_ark.builtin_endpoint_mode,
        dashscope_force_official_endpoint=bool(config.dashscope.force_official_endpoint),
        dashscope_auto_detect_endpoint=bool(config.dashscope.auto_detect_endpoint),
        siliconflow_force_official_endpoint=bool(config.siliconflow.force_official_endpoint),
        mimo_reasoning_retention_days=config.xiaomi_mimo.reasoning_retention_days,
        openai_max_retries=max(0, config.openai_responses.max_retries),
        anthropic_max_retries=max(0, config.anthropic_messages.max_retries),
        dashscope_max_retries=max(0, config.dashscope.max_retries),
        bailian_max_retries=max(0, config.bailian_responses.max_retries),
        siliconflow_max_retries=max(0, config.siliconflow.max_retries),
        volcengine_max_retries=max(0, config.volcengine_ark.max_retries),
        mimo_max_retries=max(0, config.xiaomi_mimo.max_retries),
        openai_force_max_retries=bool(config.openai_responses.force_max_retries),
        anthropic_force_max_retries=bool(config.anthropic_messages.force_max_retries),
        dashscope_force_max_retries=bool(config.dashscope.force_max_retries),
        bailian_force_max_retries=bool(config.bailian_responses.force_max_retries),
        siliconflow_force_max_retries=bool(config.siliconflow.force_max_retries),
        volcengine_force_max_retries=bool(config.volcengine_ark.force_max_retries),
        mimo_force_max_retries=bool(config.xiaomi_mimo.force_max_retries),
        openai_retry_interval=max(0.0, float(config.openai_responses.retry_interval)),
        anthropic_retry_interval=max(0.0, float(config.anthropic_messages.retry_interval)),
        dashscope_retry_interval=max(0.0, float(config.dashscope.retry_interval)),
        bailian_retry_interval=max(0.0, float(config.bailian_responses.retry_interval)),
        siliconflow_retry_interval=max(0.0, float(config.siliconflow.retry_interval)),
        volcengine_retry_interval=max(0.0, float(config.volcengine_ark.retry_interval)),
        mimo_retry_interval=max(0.0, float(config.xiaomi_mimo.retry_interval)),
        openai_force_retry_interval=bool(config.openai_responses.force_retry_interval),
        anthropic_force_retry_interval=bool(config.anthropic_messages.force_retry_interval),
        dashscope_force_retry_interval=bool(config.dashscope.force_retry_interval),
        bailian_force_retry_interval=bool(config.bailian_responses.force_retry_interval),
        siliconflow_force_retry_interval=bool(config.siliconflow.force_retry_interval),
        volcengine_force_retry_interval=bool(config.volcengine_ark.force_retry_interval),
        volcengine_prefix_cache_enabled=bool(config.volcengine_ark.prefix_cache_enabled),
        volcengine_prefix_cache_ttl_seconds=config.volcengine_ark.prefix_cache_ttl_seconds,
        mimo_force_retry_interval=bool(config.xiaomi_mimo.force_retry_interval),
        image_limits=build_image_limits(config.compatibility),
        parameter_overrides=build_parameter_overrides_registry(config),
    )


def parse_override_value(*, raw_value: str, value_kind: str, field_label: str) -> JsonValue | None:
    """把覆写框文本解析为类型化的 JSON 值；空白表示不覆写返回 None。"""

    raw = raw_value.strip()
    if not raw:
        return None
    if value_kind == "string":
        return raw
    parsed = _loads_json_value(raw, field_label=field_label)
    if value_kind == "boolean":
        if isinstance(parsed, bool):
            return parsed
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("boolean_override_value"),
                actual=raw,
            )
        )
    if value_kind == "integer":
        if isinstance(parsed, int) and not isinstance(parsed, bool):
            return parsed
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("integer_override_value"),
                actual=raw,
            )
        )
    if value_kind == "number":
        if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
            if isinstance(parsed, float) and not math.isfinite(parsed):
                raise ValueError(
                    translate(
                        "runtime.error.unsupported_value",
                        subject=field_label,
                        allowed="有限数字（不允许 NaN/Infinity）",
                    )
                )
            return parsed
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("numeric_override_value"),
                actual=raw,
            )
        )
    if value_kind == "string_list":
        if is_json_list(parsed) and all(isinstance(item, str) for item in parsed):
            return [str(item) for item in parsed]
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("json_string_array_override_value"),
                actual=raw,
            )
        )
    return _normalize_finite_json(parsed, field_label=field_label)


def _normalize_finite_json(value: object, *, field_label: str) -> JsonValue:
    """递归窄化 JSON 值并拒绝非有限浮点数（如 1e999 解析出的 inf）。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=field_label,
                    allowed="有限数字（不允许 NaN/Infinity）",
                )
            )
        return value
    mapping = json_mapping_or_none(value)
    if mapping is not None:
        return {str(key): _normalize_finite_json(item, field_label=field_label) for key, item in mapping.items()}
    if is_json_list(value):
        return [_normalize_finite_json(item, field_label=field_label) for item in value]
    return normalize_json_value(value)


def _loads_json_value(raw: str, *, field_label: str) -> object:
    try:
        return json.loads(raw, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("valid_json_override_value"),
                actual=exc.msg,
            )
        ) from exc


def _reject_json_constant(value: str) -> object:
    """拒绝 NaN/Infinity/-Infinity：这些不是标准 JSON，会生成非法请求体。"""

    raise ValueError(
        translate(
            "runtime.error.unsupported_value",
            subject="覆写值",
            allowed="标准 JSON（不允许 NaN/Infinity）",
        )
    )


def _ensure_object_section(parent: dict, key: str) -> tuple[dict, bool]:
    value = parent.get(key)
    mapping = json_mapping_or_none(value)
    if mapping is None:
        section: dict = {}
        parent[key] = section
        return section, True
    section = mapping_to_json_object(mapping)
    if value != section:
        parent[key] = section
        return section, True
    parent[key] = section
    return section, False


def _fill_override_defaults(overrides_section: dict, catalog: CapabilityParameterCatalog) -> bool:
    """为声明了可编辑默认文本的目录参数填充覆写默认值。"""

    changed = False
    for field in catalog.fields:
        if field.default_text and field.config_key not in overrides_section:
            overrides_section[field.config_key] = field.default_text
            changed = True
    return changed


def _validate_override_keys(overrides_section: Mapping[str, JsonValue], catalog: CapabilityParameterCatalog) -> None:
    """拒绝目录外覆写键，避免配置拼写错误被静默忽略。"""

    for raw_key in overrides_section:
        if catalog.field_by_safe_key(raw_key) is not None:
            continue
        raise ValueError(
            translate(
                "config.error.unknown_override",
                provider=catalog.provider,
                capability=catalog.capability,
                field=raw_key,
            )
        )
