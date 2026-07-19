import json
from collections.abc import Mapping
from typing import Literal

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator
from pydantic.config import JsonDict

from .core.common import (
    ImageProcessingLimits,
    InvalidImagePolicy,
    ProviderRuntimeOptions,
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
    field_enabled_key,
    field_override_enabled_key,
    field_override_value_key,
    get_parameter_catalog,
    iter_parameter_catalogs,
)
from .core.parameter_policy import (
    ParameterPolicy,
    ParameterPolicyRegistry,
    ProviderCapabilityPolicies,
    UnknownExtraParamsPolicy,
    normalize_policy_params,
    normalize_policy_paths,
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
    runtime_item,
    runtime_subject,
    translate,
)
from .version import DEFAULT_USER_AGENT, __version__

_UNKNOWN_POLICY_CHOICES: tuple[object, ...] = ("forward", "drop", "reject")
_PATH_LIST_UI: JsonDict = {"ui_type": "list", "item_type": "string", "hidden": True}
_JSON_OBJECT_UI: JsonDict = {"ui_type": "json", "rows": 8, "hidden": True}
_UNKNOWN_POLICY_UI: JsonDict = {
    "ui_type": "select",
    "choices": list(_UNKNOWN_POLICY_CHOICES),
}


type FieldControlValue = bool | str
type FieldControlMap = dict[str, FieldControlValue]


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


class CapabilityParameterPolicyConfig(PluginConfigBase):
    """Provider 能力级 extra_params 策略。"""

    accept_model_extra_params: bool = Field(default=True, description="是否接受模型级 extra_params")
    accept_request_extra_params: bool = Field(default=True, description="是否接受请求级 extra_params")
    fields: FieldControlMap = Field(default_factory=dict, description="文档参数字段开关与覆写控制")
    disabled_paths: list[str] = Field(
        default_factory=list,
        description="高级：静默移除的参数路径，例如 temperature、body.temperature、headers.X-Test",
        json_schema_extra=_PATH_LIST_UI,
    )
    rejected_paths: list[str] = Field(
        default_factory=list,
        description="高级：出现时直接拒绝请求的参数路径，例如 headers.Authorization",
        json_schema_extra=_PATH_LIST_UI,
    )
    default_params: dict = Field(
        default_factory=dict,
        description="高级：低优先级默认参数，会被 Host/model extra_params 覆盖",
        json_schema_extra=_JSON_OBJECT_UI,
    )
    override_params: dict = Field(
        default_factory=dict,
        description="高级：最高优先级强制覆写参数，可包含 body/headers/query 子对象",
        json_schema_extra=_JSON_OBJECT_UI,
    )
    unknown_extra_params: UnknownExtraParamsPolicy | str = Field(
        default="forward",
        description="未知 top-level extra_params 处理策略：forward/drop/reject",
        json_schema_extra=_UNKNOWN_POLICY_UI,
    )

    @field_validator("fields", mode="before")
    @classmethod
    def validate_field_controls(cls, value: object) -> FieldControlMap:
        if value is None:
            return {}
        mapping = json_mapping_or_none(value)
        if mapping is None:
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject="fields",
                    expected=runtime_expected("object"),
                    actual=type(value).__name__,
                )
            )
        normalized: FieldControlMap = {}
        for key, item in mapping.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            if isinstance(item, bool):
                normalized[normalized_key] = item
                continue
            if isinstance(item, str):
                normalized[normalized_key] = item
                continue
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"fields.{normalized_key}",
                    expected=runtime_expected("boolean_or_string"),
                    actual=type(item).__name__,
                )
            )
        return normalized

    @field_validator("disabled_paths", "rejected_paths", mode="before")
    @classmethod
    def validate_paths(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not is_json_list(value):
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=runtime_subject("parameter_paths"),
                    expected=runtime_expected("list"),
                    actual=type(value).__name__,
                )
            )
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    translate(
                        "runtime.error.expected_type",
                        subject=runtime_subject("parameter_path"),
                        expected=runtime_expected("string"),
                        actual=type(item).__name__,
                    )
                )
            normalized.append(item.strip())
        return [item for item in normalized if item]

    @field_validator("default_params", "override_params", mode="before")
    @classmethod
    def validate_param_object(cls, value: object) -> dict:
        return normalize_policy_params(value)

    @field_validator("unknown_extra_params")
    @classmethod
    def validate_unknown_extra_params(cls, value: object) -> UnknownExtraParamsPolicy:
        if value in ("forward", "drop", "reject"):
            return value
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="unknown_extra_params",
                allowed="forward/drop/reject",
            )
        )


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
    response: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    embeddings: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    audio_transcription: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    image_generation: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


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
    chat_completion: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    image_generation: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


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
    chat_completion: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    embeddings: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    audio_transcription: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    image_generation: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


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
    chat_completion: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    embeddings: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    audio_transcription: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    image_generation: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


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
    audio_transcription_prompt: str = Field(
        default="请识别音频中的内容，以文字形式返回识别结果。",
        description="ARK Responses 音频转录请求中与 input_audio 一同发送的文本提示词",
    )
    response: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    embeddings: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    audio_transcription: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    image_generation: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


class XiaomiMimoConfig(PluginConfigBase):
    """Xiaomi Mimo Provider 配置。"""

    __ui_label__ = "Xiaomi Mimo"
    __ui_icon__ = "bot"
    __ui_order__ = 7

    user_agent: str = Field(default="", description="自定义 User-Agent；留空时自动使用 MaiDock 默认 UA")
    force_disable_thinking: bool = Field(
        default=True,
        description="是否强制关闭深度思考/推理。关闭后 MaiDock 会通过工具调用元数据和 SQLite 完整回传历史 reasoning_content",
    )
    reasoning_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Mimo 工具调用 reasoning_content 的本地保留天数；范围 1..365",
    )
    audio_transcription_prompt: str = Field(
        default="请转写这段音频",
        description="Mimo 通用音频理解转录请求中与 input_audio 一同发送的文本提示词",
    )
    audio_transcription_language: Literal["auto", "zh", "en"] = Field(
        default="auto",
        description="mimo-v2.5-asr 的识别语言；auto=自动检测，zh=中文，en=英文",
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
    chat_completion: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)
    audio_transcription: CapabilityParameterPolicyConfig = Field(default_factory=CapabilityParameterPolicyConfig)


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
    siliconflow: SiliconFlowConfig = Field(default_factory=SiliconFlowConfig)
    volcengine_ark: VolcengineArkConfig = Field(default_factory=VolcengineArkConfig)
    xiaomi_mimo: XiaomiMimoConfig = Field(default_factory=XiaomiMimoConfig)
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


def build_parameter_policy(
    config: CapabilityParameterPolicyConfig,
    catalog: CapabilityParameterCatalog,
) -> ParameterPolicy:
    """构造能力级参数策略。"""

    advanced_override_params = normalize_policy_params(config.override_params)
    field_override_params = _build_field_override_params(config, catalog)
    _deep_merge_json_object(field_override_params, advanced_override_params)
    return ParameterPolicy(
        accept_model_extra_params=bool(config.accept_model_extra_params),
        accept_request_extra_params=bool(config.accept_request_extra_params),
        disabled_paths=_dedupe_paths(
            (
                *normalize_policy_paths(config.disabled_paths),
                *_disabled_field_paths(config, catalog),
            )
        ),
        rejected_paths=normalize_policy_paths(config.rejected_paths),
        default_params=normalize_policy_params(config.default_params),
        override_params=field_override_params,
        unknown_extra_params=CapabilityParameterPolicyConfig.validate_unknown_extra_params(config.unknown_extra_params),
    )


def build_parameter_policies(config: MaiDockConfig) -> ParameterPolicyRegistry:
    """根据插件配置构造 provider/capability 参数策略。"""

    return ParameterPolicyRegistry(
        openai_responses=ProviderCapabilityPolicies(
            response=build_parameter_policy(
                config.openai_responses.response,
                get_parameter_catalog("openai_responses", "response"),
            ),
            embeddings=build_parameter_policy(
                config.openai_responses.embeddings,
                get_parameter_catalog("openai_responses", "embeddings"),
            ),
            audio_transcription=build_parameter_policy(
                config.openai_responses.audio_transcription,
                get_parameter_catalog("openai_responses", "audio_transcription"),
            ),
            image_generation=build_parameter_policy(
                config.openai_responses.image_generation,
                get_parameter_catalog("openai_responses", "image_generation"),
            ),
        ),
        anthropic_messages=ProviderCapabilityPolicies(
            chat_completion=build_parameter_policy(
                config.anthropic_messages.chat_completion,
                get_parameter_catalog("anthropic_messages", "chat_completion"),
            ),
            image_generation=build_parameter_policy(
                config.anthropic_messages.image_generation,
                get_parameter_catalog("anthropic_messages", "image_generation"),
            ),
        ),
        dashscope=ProviderCapabilityPolicies(
            chat_completion=build_parameter_policy(
                config.dashscope.chat_completion,
                get_parameter_catalog("dashscope", "chat_completion"),
            ),
            embeddings=build_parameter_policy(
                config.dashscope.embeddings,
                get_parameter_catalog("dashscope", "embeddings"),
            ),
            audio_transcription=build_parameter_policy(
                config.dashscope.audio_transcription,
                get_parameter_catalog("dashscope", "audio_transcription"),
            ),
            image_generation=build_parameter_policy(
                config.dashscope.image_generation,
                get_parameter_catalog("dashscope", "image_generation"),
            ),
        ),
        siliconflow=ProviderCapabilityPolicies(
            chat_completion=build_parameter_policy(
                config.siliconflow.chat_completion,
                get_parameter_catalog("siliconflow", "chat_completion"),
            ),
            embeddings=build_parameter_policy(
                config.siliconflow.embeddings,
                get_parameter_catalog("siliconflow", "embeddings"),
            ),
            audio_transcription=build_parameter_policy(
                config.siliconflow.audio_transcription,
                get_parameter_catalog("siliconflow", "audio_transcription"),
            ),
            image_generation=build_parameter_policy(
                config.siliconflow.image_generation,
                get_parameter_catalog("siliconflow", "image_generation"),
            ),
        ),
        volcengine_ark=ProviderCapabilityPolicies(
            response=build_parameter_policy(
                config.volcengine_ark.response,
                get_parameter_catalog("volcengine_ark", "response"),
            ),
            embeddings=build_parameter_policy(
                config.volcengine_ark.embeddings,
                get_parameter_catalog("volcengine_ark", "embeddings"),
            ),
            audio_transcription=build_parameter_policy(
                config.volcengine_ark.audio_transcription,
                get_parameter_catalog("volcengine_ark", "audio_transcription"),
            ),
            image_generation=build_parameter_policy(
                config.volcengine_ark.image_generation,
                get_parameter_catalog("volcengine_ark", "image_generation"),
            ),
        ),
        xiaomi_mimo=ProviderCapabilityPolicies(
            chat_completion=build_parameter_policy(
                config.xiaomi_mimo.chat_completion,
                get_parameter_catalog("xiaomi_mimo", "chat_completion"),
            ),
            audio_transcription=build_parameter_policy(
                config.xiaomi_mimo.audio_transcription,
                get_parameter_catalog("xiaomi_mimo", "audio_transcription"),
            ),
        ),
    )


def normalize_maidock_config_data(
    config_data: Mapping[str, JsonValue],
) -> tuple[dict, bool]:
    """为所有能力参数目录填充生成的目标字段控制默认值。"""

    current = mapping_to_json_object(config_data)
    changed = False
    for catalog in iter_parameter_catalogs():
        provider_section, provider_changed = _ensure_object_section(current, catalog.provider)
        capability_section, capability_changed = _ensure_object_section(provider_section, catalog.capability)
        fields_section, fields_changed = _ensure_object_section(capability_section, "fields")
        changed = changed or provider_changed or capability_changed or fields_changed
        changed = _fill_field_control_defaults(fields_section, catalog) or changed

    validated = MaiDockConfig.model_validate(current)
    normalized = mapping_to_json_object(validated.model_dump(mode="python"))
    return normalized, changed or normalized != current


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
        siliconflow_user_agent=normalize_user_agent(config.siliconflow.user_agent),
        mimo_user_agent=normalize_user_agent(config.xiaomi_mimo.user_agent),
        volcengine_force_official_endpoint=bool(config.volcengine_ark.force_official_endpoint),
        dashscope_force_official_endpoint=bool(config.dashscope.force_official_endpoint),
        dashscope_auto_detect_endpoint=bool(config.dashscope.auto_detect_endpoint),
        siliconflow_force_official_endpoint=bool(config.siliconflow.force_official_endpoint),
        mimo_force_disable_thinking=bool(config.xiaomi_mimo.force_disable_thinking),
        mimo_reasoning_retention_days=config.xiaomi_mimo.reasoning_retention_days,
        volcengine_audio_transcription_prompt=config.volcengine_ark.audio_transcription_prompt.strip(),
        mimo_audio_transcription_prompt=config.xiaomi_mimo.audio_transcription_prompt.strip(),
        mimo_audio_transcription_language=config.xiaomi_mimo.audio_transcription_language,
        openai_max_retries=max(0, config.openai_responses.max_retries),
        anthropic_max_retries=max(0, config.anthropic_messages.max_retries),
        dashscope_max_retries=max(0, config.dashscope.max_retries),
        siliconflow_max_retries=max(0, config.siliconflow.max_retries),
        volcengine_max_retries=max(0, config.volcengine_ark.max_retries),
        mimo_max_retries=max(0, config.xiaomi_mimo.max_retries),
        openai_force_max_retries=bool(config.openai_responses.force_max_retries),
        anthropic_force_max_retries=bool(config.anthropic_messages.force_max_retries),
        dashscope_force_max_retries=bool(config.dashscope.force_max_retries),
        siliconflow_force_max_retries=bool(config.siliconflow.force_max_retries),
        volcengine_force_max_retries=bool(config.volcengine_ark.force_max_retries),
        mimo_force_max_retries=bool(config.xiaomi_mimo.force_max_retries),
        openai_retry_interval=max(0.0, float(config.openai_responses.retry_interval)),
        anthropic_retry_interval=max(0.0, float(config.anthropic_messages.retry_interval)),
        dashscope_retry_interval=max(0.0, float(config.dashscope.retry_interval)),
        siliconflow_retry_interval=max(0.0, float(config.siliconflow.retry_interval)),
        volcengine_retry_interval=max(0.0, float(config.volcengine_ark.retry_interval)),
        mimo_retry_interval=max(0.0, float(config.xiaomi_mimo.retry_interval)),
        openai_force_retry_interval=bool(config.openai_responses.force_retry_interval),
        anthropic_force_retry_interval=bool(config.anthropic_messages.force_retry_interval),
        dashscope_force_retry_interval=bool(config.dashscope.force_retry_interval),
        siliconflow_force_retry_interval=bool(config.siliconflow.force_retry_interval),
        volcengine_force_retry_interval=bool(config.volcengine_ark.force_retry_interval),
        volcengine_prefix_cache_enabled=bool(config.volcengine_ark.prefix_cache_enabled),
        volcengine_prefix_cache_ttl_seconds=config.volcengine_ark.prefix_cache_ttl_seconds,
        mimo_force_retry_interval=bool(config.xiaomi_mimo.force_retry_interval),
        image_limits=build_image_limits(config.compatibility),
        parameter_policies=build_parameter_policies(config),
    )


def _disabled_field_paths(
    config: CapabilityParameterPolicyConfig,
    catalog: CapabilityParameterCatalog,
) -> tuple[str, ...]:
    paths: list[str] = []
    for field in catalog.fields:
        if not _field_control_bool(config.fields, field_enabled_key(field), default=True):
            paths.extend(field.disable_paths)
    return tuple(paths)


def _build_field_override_params(
    config: CapabilityParameterPolicyConfig,
    catalog: CapabilityParameterCatalog,
) -> dict:
    override_params: dict = {}
    for field in catalog.fields:
        if not _field_control_bool(config.fields, field_override_enabled_key(field), default=False):
            continue
        if field.value_kind == "boolean":
            raw_value = _field_control_bool(config.fields, field_override_value_key(field), default=False)
        else:
            raw_value = _field_control_str(config.fields, field_override_value_key(field))
        value = parse_field_override_value(
            raw_value=raw_value,
            value_kind=field.value_kind,
            field_label=f"{catalog.provider}.{catalog.capability}.{field.key}",
        )
        _set_path_value(override_params, field.override_path, value)
    return override_params


def parse_field_override_value(*, raw_value: str | bool, value_kind: str, field_label: str) -> object:
    """将覆写值（来自开关、文本框等）解析为 JSON 兼容的值。"""

    if isinstance(raw_value, bool):
        if value_kind == "boolean":
            return raw_value
        raw = "true" if raw_value else "false"
    else:
        raw = raw_value.strip()
        if not raw:
            raise ValueError(
                translate("runtime.error.required", subject=field_label, field=runtime_item("override_value"))
            )
    if value_kind == "string":
        return raw
    if value_kind == "boolean":
        if isinstance(raw_value, bool):
            return raw_value
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("boolean_override_value"),
                actual=type(raw_value).__name__,
            )
        )
    parsed = _loads_json_value(raw, field_label=field_label)
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
    return normalize_json_value(parsed)


def _loads_json_value(raw: str, *, field_label: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            translate(
                "runtime.error.expected_type",
                subject=field_label,
                expected=runtime_expected("valid_json_override_value"),
                actual=exc.msg,
            )
        ) from exc


def _field_control_bool(fields: Mapping[str, FieldControlValue], key: str, *, default: bool) -> bool:
    value = fields.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return default


def _field_control_str(fields: Mapping[str, FieldControlValue], key: str) -> str:
    value = fields.get(key)
    return value if isinstance(value, str) else ""


def _dedupe_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = path.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


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


def _fill_field_control_defaults(fields_section: dict, catalog: CapabilityParameterCatalog) -> bool:
    changed = False
    for field in catalog.fields:
        override_default: FieldControlValue = False if field.value_kind == "boolean" else ""
        defaults: tuple[tuple[str, FieldControlValue], ...] = (
            (field_enabled_key(field), True),
            (field_override_enabled_key(field), False),
            (field_override_value_key(field), override_default),
        )
        for key, default in defaults:
            if key not in fields_section:
                fields_section[key] = default
                changed = True
    return changed


def _set_path_value(target: dict, path: tuple[str, ...], value: object) -> None:
    if not path:
        return
    current = target
    for part in path[:-1]:
        child = json_mapping_or_none(current.get(part))
        child_object = mapping_to_json_object(child) if child is not None else {}
        current[part] = child_object
        current = child_object
    current[path[-1]] = normalize_json_value(value)


def _deep_merge_json_object(target: dict, source: Mapping[str, JsonValue]) -> None:
    for key, value in source.items():
        normalized_key = str(key)
        source_mapping = json_mapping_or_none(value)
        target_mapping = json_mapping_or_none(target.get(normalized_key))
        if source_mapping is not None and target_mapping is not None:
            merged = mapping_to_json_object(target_mapping)
            _deep_merge_json_object(merged, mapping_to_json_object(source_mapping))
            target[normalized_key] = merged
            continue
        target[normalized_key] = normalize_json_value(value)
