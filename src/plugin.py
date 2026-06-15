from collections.abc import Callable, Mapping
from typing import Literal

from maibot_sdk import LLMProvider, LLMProviderBase, MaiBotPlugin

from .config import MaiDockConfig, build_runtime_options, normalize_maidock_config_data
from .config_schema import build_maidock_config_schema
from .core.common import ProviderRuntimeOptions
from .core.json_types import value_to_json_object
from .version import __version__


type _ProviderKey = Literal["openai", "anthropic", "volcengine", "dashscope", "siliconflow", "xiaomi_mimo"]
type _ProviderFactory = Callable[[ProviderRuntimeOptions], LLMProviderBase]


def _create_openai_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.openai_responses_provider.provider import OpenAIResponsesProvider

    return OpenAIResponsesProvider(options=options)


def _create_anthropic_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.anthropic_messages_provider.provider import AnthropicMessagesProvider

    return AnthropicMessagesProvider(options=options)


def _create_volcengine_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.volcengine_ark_provider.provider import VolcengineArkResponsesProvider

    return VolcengineArkResponsesProvider(options=options)


def _create_dashscope_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.dashscope_provider.provider import DashScopeProvider

    return DashScopeProvider(options=options)


def _create_siliconflow_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.siliconflow_provider.provider import SiliconFlowProvider

    return SiliconFlowProvider(options=options)


def _create_mimo_provider(options: ProviderRuntimeOptions) -> LLMProviderBase:
    from .providers.xiaomi_mimo_provider.provider import XiaomiMimoProvider

    return XiaomiMimoProvider(options=options)


class MaiDockPlugin(MaiBotPlugin):
    """提供额外端点支持的插件。"""

    config_model = MaiDockConfig

    def __init__(self) -> None:
        super().__init__()
        self._runtime_options: ProviderRuntimeOptions | None = None
        self._openai_provider: LLMProviderBase | None = None
        self._anthropic_provider: LLMProviderBase | None = None
        self._volcengine_provider: LLMProviderBase | None = None
        self._dashscope_provider: LLMProviderBase | None = None
        self._siliconflow_provider: LLMProviderBase | None = None
        self._mimo_provider: LLMProviderBase | None = None

    async def on_load(self) -> None:
        self._invalidate_runtime_state()

    async def on_unload(self) -> None:
        self._invalidate_runtime_state()

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        del scope, config_data, version
        self._invalidate_runtime_state()

    def normalize_plugin_config(self, config_data: Mapping[str, object] | None) -> tuple[dict[str, object], bool]:
        normalized_config, changed = super().normalize_plugin_config(config_data)
        maidock_config, maidock_changed = normalize_maidock_config_data(normalized_config)
        return maidock_config, changed or maidock_changed

    def get_webui_config_schema(
        self,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, object]:
        return build_maidock_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )

    def _read_config(self) -> MaiDockConfig | None:
        try:
            config = self.config
        except RuntimeError:
            return None
        if isinstance(config, MaiDockConfig):
            return config
        return MaiDockConfig.model_validate(config)

    def _get_runtime_options(self) -> ProviderRuntimeOptions:
        if self._runtime_options is None:
            self._runtime_options = build_runtime_options(self._read_config())
        return self._runtime_options

    def _clear_provider_instances(self) -> None:
        self._openai_provider = None
        self._anthropic_provider = None
        self._volcengine_provider = None
        self._dashscope_provider = None
        self._siliconflow_provider = None
        self._mimo_provider = None

    def _invalidate_runtime_state(self) -> None:
        self._runtime_options = None
        self._clear_provider_instances()

    def _get_provider_slot(self, key: _ProviderKey) -> LLMProviderBase | None:
        match key:
            case "openai":
                return self._openai_provider
            case "anthropic":
                return self._anthropic_provider
            case "volcengine":
                return self._volcengine_provider
            case "dashscope":
                return self._dashscope_provider
            case "siliconflow":
                return self._siliconflow_provider
            case "xiaomi_mimo":
                return self._mimo_provider

    def _set_provider_slot(self, key: _ProviderKey, provider: LLMProviderBase) -> None:
        match key:
            case "openai":
                self._openai_provider = provider
            case "anthropic":
                self._anthropic_provider = provider
            case "volcengine":
                self._volcengine_provider = provider
            case "dashscope":
                self._dashscope_provider = provider
            case "siliconflow":
                self._siliconflow_provider = provider
            case "xiaomi_mimo":
                self._mimo_provider = provider

    def _get_or_create_provider(self, key: _ProviderKey, factory: _ProviderFactory) -> LLMProviderBase:
        provider = self._get_provider_slot(key)
        if provider is not None:
            return provider
        provider = factory(self._get_runtime_options())
        self._set_provider_slot(key, provider)
        return provider

    def _require_openai_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("openai", _create_openai_provider)

    def _require_anthropic_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("anthropic", _create_anthropic_provider)

    def _require_volcengine_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("volcengine", _create_volcengine_provider)

    def _require_dashscope_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("dashscope", _create_dashscope_provider)

    def _require_siliconflow_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("siliconflow", _create_siliconflow_provider)

    def _require_mimo_provider(self) -> LLMProviderBase:
        return self._get_or_create_provider("xiaomi_mimo", _create_mimo_provider)

    def _ensure_enabled(self) -> None:
        config = self._read_config()
        if config is not None and not config.plugin.enabled:
            raise RuntimeError("MaiDock 当前已在插件配置中禁用")

    @LLMProvider(
        client_type="maidock-openai-responses",
        name="MaiDock OpenAI Responses",
        description="基于 OpenAI Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def openai_responses_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(
            await self._require_openai_provider().dispatch(operation=operation, request=request)
        )

    @LLMProvider(
        client_type="maidock-anthropic-messages",
        name="MaiDock Anthropic Messages",
        description="基于 Anthropic Messages API 的 LLM Provider。",
        version=__version__,
    )
    async def anthropic_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(
            await self._require_anthropic_provider().dispatch(operation=operation, request=request)
        )

    @LLMProvider(
        client_type="maidock-volcengine-ark-responses",
        name="MaiDock Volcengine Ark Responses",
        description="基于火山方舟 Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def volcengine_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(
            await self._require_volcengine_provider().dispatch(operation=operation, request=request)
        )

    @LLMProvider(
        client_type="maidock-dashscope",
        name="MaiDock DashScope",
        description="基于 DashScope 原生 HTTP API 的 LLM Provider。",
        version=__version__,
    )
    async def dashscope_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(
            await self._require_dashscope_provider().dispatch(operation=operation, request=request)
        )

    @LLMProvider(
        client_type="maidock-siliconflow",
        name="MaiDock SiliconFlow",
        description="基于 SiliconFlow 原生 HTTP API 的 LLM Provider。",
        version=__version__,
    )
    async def siliconflow_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(
            await self._require_siliconflow_provider().dispatch(operation=operation, request=request)
        )

    @LLMProvider(
        client_type="maidock-xiaomi-mimo",
        name="MaiDock Xiaomi Mimo",
        description="基于小米 Mimo Chat Completions API 的 LLM Provider。",
        version=__version__,
    )
    async def xiaomi_mimo_provider(self, operation: str, request: dict) -> dict:
        self._ensure_enabled()
        return value_to_json_object(await self._require_mimo_provider().dispatch(operation=operation, request=request))


def create_plugin() -> MaiDockPlugin:
    return MaiDockPlugin()
