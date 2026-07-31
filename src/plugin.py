from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Protocol, cast

from maibot_sdk import LLMProvider, MaiBotPlugin
from pydantic import ValidationError

from .config import MaiDockConfig, build_runtime_options, normalize_maidock_config_data
from .config_migration import inject_legacy_migration_bridge
from .config_schema import build_maidock_config_schema
from .core.common import ProviderRuntimeOptions
from .core.json_types import JsonValue, value_to_json_object
from .core.state_store import PluginStateStore
from .i18n import (
    DEFAULT_LOCALE,
    Locale,
    format_validation_error,
    normalize_locale,
    translate,
    use_locale,
    validate_catalogs,
)
from .public_api import PublicApiConfig, PublicApiRuntime
from .public_api.api.definitions import PublicApiHandler
from .public_api.domain import PublicRpcObject
from .runtime import (
    RuntimeKey,
    RuntimeContainer,
    VendorClientContainer,
    create_vendor_client,
    create_vendor_runtime,
)
from .version import __version__


class _PluginPaths(Protocol):
    data_dir: Path
    runtime_dir: Path


class _PluginContextWithPaths(Protocol):
    paths: _PluginPaths


class MaiDockPlugin(MaiBotPlugin):
    """提供额外端点支持的插件。"""

    config_model = MaiDockConfig

    def __init__(self) -> None:
        super().__init__()
        self._runtime_options: ProviderRuntimeOptions | None = None
        self._runtime_container: RuntimeContainer | None = None
        self._client_container: VendorClientContainer | None = None
        self._public_api_runtime: PublicApiRuntime | None = None
        self._public_apis_online = False
        self._state_store: PluginStateStore | None = None

    async def on_load(self) -> None:
        validate_catalogs()
        await self._close_all_runtimes()
        self._runtime_options = None
        config = self._read_config() or MaiDockConfig()
        with use_locale(config.plugin.locale):
            self._initialize_state_store()
            self._client_container = VendorClientContainer(factory=create_vendor_client)
            paths = cast(_PluginContextWithPaths, self.ctx).paths
            public_runtime = PublicApiRuntime(
                data_dir=paths.data_dir,
                config=self._effective_public_config(config),
                clients=self._client_container,
            )
            await public_runtime.start()
            self._public_api_runtime = public_runtime
            await self._sync_public_api_registration(config)

    async def on_unload(self) -> None:
        await self._sync_public_api_registration(None)
        await self._close_all_runtimes()
        self._runtime_options = None
        if self._state_store is not None:
            await self._state_store.close()
            self._state_store = None

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        del scope, config_data, version
        await self._close_runtime_container()
        self._runtime_options = None
        config = self._read_config() or MaiDockConfig()
        with use_locale(config.plugin.locale):
            public_runtime = self._public_api_runtime
            if public_runtime is not None:
                await public_runtime.update_config(self._effective_public_config(config))
            await self._sync_public_api_registration(config)

    async def _close_all_runtimes(self) -> None:
        public_runtime = self._public_api_runtime
        self._public_api_runtime = None
        if public_runtime is not None:
            await public_runtime.stop()
        await self._close_runtime_container()
        clients = self._client_container
        self._client_container = None
        if clients is not None:
            await clients.aclose()

    async def _close_runtime_container(self) -> None:
        container = self._runtime_container
        self._runtime_container = None
        if container is not None:
            await container.aclose()

    def _initialize_state_store(self) -> None:
        if self._state_store is not None:
            return
        try:
            paths = cast(_PluginContextWithPaths, self.ctx).paths
        except AttributeError as exc:
            raise RuntimeError(translate("runtime.plugin.paths_missing")) from exc
        self._state_store = PluginStateStore(paths.data_dir / "maidock_state.sqlite3")

    @staticmethod
    def _effective_public_config(config: MaiDockConfig) -> PublicApiConfig:
        enabled = config.plugin.enabled and config.public_api.enabled
        return config.public_api.model_copy(update={"enabled": enabled})

    async def _sync_public_api_registration(self, config: MaiDockConfig | None) -> None:
        should_enable = bool(config and config.plugin.enabled and config.public_api.enabled)
        runtime = self._public_api_runtime
        if runtime is not None:
            runtime.require_engine().set_accepting(should_enable)
        if should_enable:
            if runtime is None:
                raise RuntimeError("Public API Runtime 尚未初始化")
            self.clear_dynamic_apis()
            for definition in runtime.require_facade().definitions():
                self.register_dynamic_api(
                    definition.name,
                    self._dynamic_api_handler(definition.handler),
                    description=translate(definition.description_key),
                    version="1",
                    public=True,
                    timeout_ms=25_000,
                )
            await self.sync_dynamic_apis(offline_reason=translate("public_api.offline"))
            self._public_apis_online = True
            return
        if not self._public_apis_online:
            return
        self.clear_dynamic_apis()
        await self.sync_dynamic_apis(offline_reason=translate("public_api.offline"))
        self._public_apis_online = False

    @staticmethod
    def _dynamic_api_handler(
        handler: PublicApiHandler,
    ) -> Callable[..., Awaitable[PublicRpcObject]]:
        async def invoke(request: object, **_metadata: object) -> PublicRpcObject:
            return await handler(request)

        return invoke

    def normalize_plugin_config(self, config_data: Mapping[str, object] | None) -> tuple[dict[str, object], bool]:
        locale = self._locale_from_config_data(config_data)
        with use_locale(locale):
            raw_config = value_to_json_object(dict(config_data or {}))
            normalized_config, changed = normalize_maidock_config_data(raw_config)
            return normalized_config, changed

    def get_default_config(self) -> dict[str, JsonValue]:
        """返回含 1.1.3 迁移桥接键的 Runner 默认配置骨架。"""

        default_config = value_to_json_object(super().get_default_config())
        return inject_legacy_migration_bridge(default_config)

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
            locale=self._get_locale(),
        )

    @staticmethod
    def _locale_from_config_data(config_data: Mapping[str, object] | None) -> Locale:
        if config_data is None:
            return DEFAULT_LOCALE
        plugin_section = config_data.get("plugin")
        if not isinstance(plugin_section, Mapping):
            return DEFAULT_LOCALE
        try:
            return normalize_locale(plugin_section.get("locale", DEFAULT_LOCALE))
        except ValueError:
            return DEFAULT_LOCALE

    def _get_locale(self) -> Locale:
        config = self._read_config()
        return config.plugin.locale if config is not None else DEFAULT_LOCALE

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

    def _get_runtime_container(self) -> RuntimeContainer:
        if self._runtime_container is None:
            clients = self._client_container
            if clients is None:
                clients = VendorClientContainer(factory=create_vendor_client)
                self._client_container = clients
            self._runtime_container = RuntimeContainer(
                options=self._get_runtime_options(),
                state_store=self._state_store,
                factory=create_vendor_runtime,
                clients=clients,
            )
        return self._runtime_container

    def _ensure_enabled(self) -> None:
        config = self._read_config()
        if config is not None and not config.plugin.enabled:
            raise RuntimeError(translate("runtime.plugin.disabled"))

    # 这一层是 SDK 直接调用的入口：SDK 基类 dispatch 的形参是 dict[str, Any]，
    # 所以请求只能收窄成 dict[str, JsonValue] 而不能像 host_adapters 那样用 Mapping，
    # 否则原样转发就不再类型兼容。
    async def _dispatch_provider(
        self, key: RuntimeKey, operation: str, request: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        options = self._get_runtime_options()
        with use_locale(options.locale):
            self._ensure_enabled()
            if operation not in {"response", "embedding", "audio_transcription"}:
                raise ValueError(translate("runtime.error.operation_unsupported", operation=operation))
            runtime = await self._get_runtime_container().get(key)
            try:
                result = await runtime.ingress.dispatch(operation=operation, request=request)
            except ValidationError as exc:
                raise ValueError(format_validation_error(exc)) from exc
            return value_to_json_object(result)

    @LLMProvider(
        client_type="maidock-openai-responses",
        name="MaiDock OpenAI Responses",
        description="基于 OpenAI Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def openai_responses_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("openai", operation, request)

    @LLMProvider(
        client_type="maidock-anthropic-messages",
        name="MaiDock Anthropic Messages",
        description="基于 Anthropic Messages API 的 LLM Provider。",
        version=__version__,
    )
    async def anthropic_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("anthropic", operation, request)

    @LLMProvider(
        client_type="maidock-dashscope",
        name="MaiDock 阿里云百炼 DashScope",
        description="基于阿里云百炼 DashScope 原生 HTTP API 的 LLM Provider。",
        version=__version__,
    )
    async def dashscope_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("dashscope", operation, request)

    @LLMProvider(
        client_type="maidock-bailian-responses",
        name="MaiDock 阿里云百炼 Responses",
        description="基于阿里云百炼 OpenAI Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def bailian_responses_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("bailian_responses", operation, request)

    @LLMProvider(
        client_type="maidock-siliconflow",
        name="MaiDock SiliconFlow",
        description="基于 SiliconFlow 原生 HTTP API 的 LLM Provider。",
        version=__version__,
    )
    async def siliconflow_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("siliconflow", operation, request)

    @LLMProvider(
        client_type="maidock-volcengine-ark-responses",
        name="MaiDock Volcengine Ark Responses",
        description="基于火山方舟 Responses API 的 LLM Provider。",
        version=__version__,
    )
    async def volcengine_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("volcengine", operation, request)

    @LLMProvider(
        client_type="maidock-xiaomi-mimo",
        name="MaiDock Xiaomi Mimo",
        description="基于小米 Mimo Chat Completions API 的 LLM Provider。",
        version=__version__,
    )
    async def xiaomi_mimo_provider(self, operation: str, request: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return await self._dispatch_provider("xiaomi_mimo", operation, request)


def create_plugin() -> MaiDockPlugin:
    return MaiDockPlugin()
