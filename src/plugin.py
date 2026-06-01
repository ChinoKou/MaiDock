from typing import cast

from maibot_sdk import LLMProvider, LLMProviderBase, MaiBotPlugin

from .config import MaiDockConfig, build_runtime_options
from .core.schemas import JsonObject
from .providers.anthropic_messages import AnthropicMessagesProvider
from .providers.openai_responses import OpenAIResponsesProvider
from .version import __version__


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

    def _read_config(self) -> MaiDockConfig | None:
        try:
            return cast(MaiDockConfig, self.config)
        except RuntimeError:
            return None

    def _refresh_providers(self) -> None:
        options = build_runtime_options(self._read_config())
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
        config = self._read_config()
        if config is not None and not config.plugin.enabled:
            raise RuntimeError("MaiDock 当前已在插件配置中禁用")

    @LLMProvider(
        client_type="maidock-openai-responses",
        name="MaiDock OpenAI Responses",
        description="基于 OpenAI Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def openai_responses_provider(self, operation: str, request: JsonObject) -> JsonObject:
        self._ensure_enabled()
        return cast(JsonObject, await self._require_openai_provider().dispatch(operation=operation, request=request))

    @LLMProvider(
        client_type="maidock-anthropic-messages",
        name="MaiDock Anthropic Messages",
        description="基于 Anthropic Messages API 的 LLM Provider。",
        version=__version__,
    )
    async def anthropic_provider(self, operation: str, request: JsonObject) -> JsonObject:
        self._ensure_enabled()
        return cast(JsonObject, await self._require_anthropic_provider().dispatch(operation=operation, request=request))


def create_plugin() -> MaiDockPlugin:
    return MaiDockPlugin()
