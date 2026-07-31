import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from maibot_sdk import PluginContext

from src import plugin as plugin_module
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.host_adapters.common.rpc import HostRpcRequest, HostRpcResponse
from src.i18n import translate
from src.plugin import MaiDockPlugin
from src.runtime import LLMProviderIngress, VendorClient, VendorRuntime

type ProviderName = Literal[
    "openai",
    "anthropic",
    "volcengine",
    "dashscope",
    "siliconflow",
    "xiaomi_mimo",
]
type ProviderEntry = Callable[[MaiDockPlugin, str, dict[str, Any]], Awaitable[dict[str, Any]]]

PROVIDER_NAMES: tuple[ProviderName, ...] = (
    "openai",
    "anthropic",
    "volcengine",
    "dashscope",
    "siliconflow",
    "xiaomi_mimo",
)
PROVIDER_ENTRIES: dict[ProviderName, ProviderEntry] = {
    "openai": MaiDockPlugin.openai_responses_provider,
    "anthropic": MaiDockPlugin.anthropic_provider,
    "volcengine": MaiDockPlugin.volcengine_provider,
    "dashscope": MaiDockPlugin.dashscope_provider,
    "siliconflow": MaiDockPlugin.siliconflow_provider,
    "xiaomi_mimo": MaiDockPlugin.xiaomi_mimo_provider,
}


class FakePluginPaths:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class FakeContext:
    def __init__(self, data_dir: Path) -> None:
        self.paths = FakePluginPaths(data_dir)


class ContextWithoutPaths:
    pass


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.close_count = 0

    async def aclose(self) -> None:
        self.closed = True
        self.close_count += 1


class FakeProvider:
    def __init__(
        self,
        provider_name: ProviderName,
        generation: int,
        options: ProviderRuntimeOptions,
        state_store: PluginStateStore | None,
    ) -> None:
        self.provider_name = provider_name
        self.generation = generation
        self.options = options
        self.state_store = state_store
        self.operations: list[str] = []
        self.requests: list[HostRpcRequest] = []

    async def _record(self, operation: str, request: HostRpcRequest) -> HostRpcResponse:
        self.operations.append(operation)
        self.requests.append(request)
        await asyncio.sleep(0)
        return {
            "provider": self.provider_name,
            "operation": operation,
            "generation": self.generation,
        }

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        return await self._record("response", request)

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        return await self._record("embedding", request)

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        return await self._record("audio_transcription", request)


class FactoryRecorder:
    def __init__(self) -> None:
        self.providers: dict[ProviderName, list[FakeProvider]] = {name: [] for name in PROVIDER_NAMES}
        self.state_stores: dict[ProviderName, list[PluginStateStore | None]] = {name: [] for name in PROVIDER_NAMES}
        self.clients: dict[ProviderName, list[FakeClient]] = {name: [] for name in PROVIDER_NAMES}

    def factory(
        self,
        raw_provider_name: str,
        options: ProviderRuntimeOptions,
        state_store: PluginStateStore | None,
        raw_client: VendorClient,
    ) -> VendorRuntime:
        provider_name = cast(ProviderName, raw_provider_name)
        if provider_name == "volcengine" and options.volcengine_prefix_cache_enabled and state_store is None:
            raise RuntimeError(translate("runtime.plugin.cache_store_missing"))
        if provider_name == "xiaomi_mimo" and state_store is None:
            raise RuntimeError(translate("runtime.plugin.store_missing"))
        providers = self.providers[provider_name]
        provider = FakeProvider(
            provider_name,
            len(providers) + 1,
            options,
            state_store,
        )
        client = cast(FakeClient, raw_client)
        providers.append(provider)
        if client not in self.clients[provider_name]:
            self.clients[provider_name].append(client)
        self.state_stores[provider_name].append(state_store)
        ingress = LLMProviderIngress(
            adapter=provider,
            capabilities=frozenset({"response", "embedding", "audio_transcription"}),
            provider_name=provider_name,
        )
        return VendorRuntime(client=client, host_adapter=provider, ingress=ingress)


def install_factories(
    monkeypatch: pytest.MonkeyPatch,
    recorder: FactoryRecorder,
) -> None:
    monkeypatch.setattr(plugin_module, "create_vendor_runtime", recorder.factory)

    def create_client(raw_provider_name: str) -> FakeClient:
        del raw_provider_name
        return FakeClient()

    monkeypatch.setattr(plugin_module, "create_vendor_client", create_client)


def create_plugin(
    tmp_path: Path,
    *,
    config: dict[str, Any] | None = None,
    with_paths: bool = True,
) -> MaiDockPlugin:
    plugin = MaiDockPlugin()
    plugin.set_plugin_config(config if config is not None else {})
    context = FakeContext(tmp_path / "data") if with_paths else ContextWithoutPaths()
    plugin._set_context(cast(PluginContext, context))
    return plugin
