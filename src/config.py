from maibot_sdk import Field, PluginConfigBase

from .core.common import ImageProcessingLimits, InvalidImagePolicy, ProviderRuntimeOptions
from .core.parsing import normalize_reasoning_parse_mode, normalize_tool_argument_parse_mode


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用 MaiDock")
    config_version: str = Field(default="1.0.0", description="配置版本")


class DiagnosticsConfig(PluginConfigBase):
    """诊断配置。"""

    __ui_label__ = "诊断"
    __ui_icon__ = "bug"
    __ui_order__ = 1

    include_raw_data: bool = Field(default=False, description="是否把上游响应摘要放入 raw_data")
    log_payload_summary: bool = Field(default=True, description="是否记录脱敏后的请求/响应摘要日志")
    log_payload_debug: bool = Field(default=False, description="是否记录脱敏后的详细请求载荷")
    anthropic_sdk_log_level: str = Field(
        default="INFO", description="Anthropic SDK 日志级别：inherit/DEBUG/INFO/WARNING/ERROR/CRITICAL"
    )


class CompatibilityConfig(PluginConfigBase):
    """兼容性配置。"""

    __ui_label__ = "兼容性"
    __ui_icon__ = "settings-2"
    __ui_order__ = 2

    tool_argument_parse_mode: str = Field(
        default="auto", description="工具参数解析模式：auto/strict/repair/double_decode"
    )
    reasoning_parse_mode: str = Field(default="auto", description="推理内容解析模式：auto/native/think_tag/none")
    strict_extra_params: bool = Field(default=False, description="是否拒绝未知 extra_params 字段")
    invalid_image_policy: str = Field(default="placeholder", description="无效图片处理策略：placeholder/skip/error")
    max_image_bytes_mb: int = Field(default=30, description="单张图片解码后最大字节数（MB）")
    max_image_pixels: int = Field(default=25_000_000, description="单张图片最大像素数量")
    max_image_dimension: int = Field(default=8192, description="单张图片单边最大像素")
    max_image_frames: int = Field(default=64, description="动图最大帧数")


class MaiDockConfig(PluginConfigBase):
    """MaiDock 插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)


def normalize_invalid_image_policy(raw_policy: str) -> InvalidImagePolicy:
    """规范化无效图片处理策略。"""

    if raw_policy == "skip":
        return "skip"
    if raw_policy == "error":
        return "error"
    return "placeholder"


def normalize_anthropic_sdk_log_level(raw_level: str | None) -> str | None:
    """规范化 Anthropic SDK 日志级别。"""

    normalized = (raw_level or "INFO").strip().upper()
    if normalized in {"", "INHERIT"}:
        return "inherit"
    if normalized in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        return normalized
    return "INFO"


def positive_int(value: object, default: int) -> int:
    """读取正整数配置，非法时回退到默认值。"""

    if isinstance(value, int) and value > 0:
        return value
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


def build_runtime_options(config: MaiDockConfig | None = None) -> ProviderRuntimeOptions:
    """根据插件配置构造 Provider 运行时选项。"""

    if config is None:
        return ProviderRuntimeOptions()
    invalid_image_policy = normalize_invalid_image_policy(config.compatibility.invalid_image_policy)
    return ProviderRuntimeOptions(
        include_raw_data=bool(config.diagnostics.include_raw_data),
        log_payload_summary=bool(config.diagnostics.log_payload_summary),
        log_payload_debug=bool(config.diagnostics.log_payload_debug),
        anthropic_sdk_log_level=normalize_anthropic_sdk_log_level(config.diagnostics.anthropic_sdk_log_level),
        tool_argument_parse_mode=normalize_tool_argument_parse_mode(config.compatibility.tool_argument_parse_mode),
        reasoning_parse_mode=normalize_reasoning_parse_mode(config.compatibility.reasoning_parse_mode),
        strict_extra_params=bool(config.compatibility.strict_extra_params),
        invalid_image_policy=invalid_image_policy,
        image_limits=build_image_limits(config.compatibility),
    )
