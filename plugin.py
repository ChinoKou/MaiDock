from typing import cast

from maibot_sdk import Field, LLMProvider, LLMProviderBase, MaiBotPlugin, PluginConfigBase

from .anthropic_messages import AnthropicMessagesProvider
from .common import ImageProcessingLimits, InvalidImagePolicy, ProviderRuntimeOptions
from .openai_responses import OpenAIResponsesProvider
from .schemas import JsonObject
from .parsing import normalize_reasoning_parse_mode, normalize_tool_argument_parse_mode


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


class MaiDockPlugin(MaiBotPlugin):
    """提供 OpenAI Responses 与 Anthropic Messages 端点的 Provider 插件。"""

    config_model = MaiDockConfig

    def __init__(self) -> None:
        super().__init__()
        self._openai_provider: LLMProviderBase | None = None
        self._anthropic_provider: LLMProviderBase | None = None

    async def on_load(self) -> None:
        self._refresh_providers()

    async def on_unload(self) -> None:
        self._openai_provider = None
        self._anthropic_provider = None

    async def on_config_update(self, scope: str, config_data: JsonObject, version: str) -> None:
        del scope, config_data, version
        self._refresh_providers()

    def _build_options(self) -> ProviderRuntimeOptions:
        try:
            config = cast(MaiDockConfig, self.config)
        except RuntimeError:
            return ProviderRuntimeOptions()

        invalid_image_policy = self._normalize_invalid_image_policy(config.compatibility.invalid_image_policy)
        return ProviderRuntimeOptions(
            include_raw_data=bool(config.diagnostics.include_raw_data),
            log_payload_summary=bool(config.diagnostics.log_payload_summary),
            log_payload_debug=bool(config.diagnostics.log_payload_debug),
            anthropic_sdk_log_level=self._normalize_anthropic_sdk_log_level(config.diagnostics.anthropic_sdk_log_level),
            tool_argument_parse_mode=normalize_tool_argument_parse_mode(config.compatibility.tool_argument_parse_mode),
            reasoning_parse_mode=normalize_reasoning_parse_mode(config.compatibility.reasoning_parse_mode),
            strict_extra_params=bool(config.compatibility.strict_extra_params),
            invalid_image_policy=invalid_image_policy,
            image_limits=self._build_image_limits(config.compatibility),
        )

    @staticmethod
    def _normalize_invalid_image_policy(raw_policy: str) -> InvalidImagePolicy:
        if raw_policy == "skip":
            return "skip"
        if raw_policy == "error":
            return "error"
        return "placeholder"

    @staticmethod
    def _normalize_anthropic_sdk_log_level(raw_level: str | None) -> str | None:
        normalized = (raw_level or "INFO").strip().upper()
        if normalized in {"", "INHERIT"}:
            return "inherit"
        if normalized in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return normalized
        return "INFO"

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        if isinstance(value, int) and value > 0:
            return value
        return default

    @classmethod
    def _build_image_limits(cls, config: CompatibilityConfig) -> ImageProcessingLimits:
        limits = ImageProcessingLimits()
        max_decoded_bytes = cls._positive_int(config.max_image_bytes_mb, 30) * 1024 * 1024
        return ImageProcessingLimits(
            max_base64_chars=((max_decoded_bytes + 2) // 3) * 4,
            max_decoded_bytes=max_decoded_bytes,
            max_pixels=cls._positive_int(config.max_image_pixels, limits.max_pixels),
            max_dimension=cls._positive_int(config.max_image_dimension, limits.max_dimension),
            max_frames=cls._positive_int(config.max_image_frames, limits.max_frames),
        )

    def _refresh_providers(self) -> None:
        options = self._build_options()
        self._openai_provider = OpenAIResponsesProvider(options=options)
        self._anthropic_provider = AnthropicMessagesProvider(options=options)

    def _require_openai_provider(self) -> LLMProviderBase:
        if self._openai_provider is None:
            self._refresh_providers()
        if self._openai_provider is None:
            raise RuntimeError("MaiDock OpenAI Responses Provider 尚未初始化")
        return self._openai_provider

    def _require_anthropic_provider(self) -> LLMProviderBase:
        if self._anthropic_provider is None:
            self._refresh_providers()
        if self._anthropic_provider is None:
            raise RuntimeError("MaiDock Anthropic Messages Provider 尚未初始化")
        return self._anthropic_provider

    def _ensure_enabled(self) -> None:
        try:
            config = cast(MaiDockConfig, self.config)
        except RuntimeError:
            return
        if not config.plugin.enabled:
            raise RuntimeError("MaiDock 当前已在插件配置中禁用")

    @LLMProvider(
        client_type="maidock-openai-responses",
        name="MaiDock OpenAI Responses",
        description="基于 OpenAI Responses API 的 LLM Provider。",
        version="1.0.0",
    )
    async def openai_responses_provider(self, operation: str, request: JsonObject) -> JsonObject:
        self._ensure_enabled()
        return await self._require_openai_provider().dispatch(operation=operation, request=request)

    @LLMProvider(
        client_type="maidock-anthropic",
        name="MaiDock Anthropic Messages",
        description="基于 Anthropic Messages API 的 LLM Provider。",
        version="1.0.0",
    )
    async def anthropic_provider(self, operation: str, request: JsonObject) -> JsonObject:
        self._ensure_enabled()
        return await self._require_anthropic_provider().dispatch(operation=operation, request=request)


def create_plugin() -> MaiDockPlugin:
    return MaiDockPlugin()
