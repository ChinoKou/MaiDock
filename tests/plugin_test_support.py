import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from maibot_sdk import LLMProviderBase, PluginContext

from src import plugin as plugin_module
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.plugin import MaiDockPlugin

type ProviderName = Literal[
    "openai",
    "anthropic",
    "volcengine",
    "dashscope",
    "siliconflow",
    "xiaomi_mimo",
]
type ProviderEntry = Callable[[MaiDockPlugin, str, dict[str, Any]], Awaitable[dict[str, Any]]]
type ProviderFactory = Callable[[ProviderRuntimeOptions, PluginStateStore | None], LLMProviderBase]

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
FACTORY_NAMES: dict[ProviderName, str] = {
    "openai": "_create_openai_provider",
    "anthropic": "_create_anthropic_provider",
    "volcengine": "_create_volcengine_provider",
    "dashscope": "_create_dashscope_provider",
    "siliconflow": "_create_siliconflow_provider",
    "xiaomi_mimo": "_create_mimo_provider",
}


class FakePluginPaths:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class FakeContext:
    def __init__(self, data_dir: Path) -> None:
        self.paths = FakePluginPaths(data_dir)


class ContextWithoutPaths:
    pass


class FakeProvider(LLMProviderBase):
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
        self.requests: list[dict[str, Any]] = []

    async def dispatch(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        self.operations.append(operation)
        self.requests.append(request)
        await asyncio.sleep(0)
        return {
            "provider": self.provider_name,
            "operation": operation,
            "generation": self.generation,
        }

    async def get_response(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.dispatch("response", request)


class FactoryRecorder:
    def __init__(self) -> None:
        self.providers: dict[ProviderName, list[FakeProvider]] = {name: [] for name in PROVIDER_NAMES}
        self.state_stores: dict[ProviderName, list[PluginStateStore | None]] = {name: [] for name in PROVIDER_NAMES}

    def factory_for(self, provider_name: ProviderName) -> ProviderFactory:
        def factory(
            options: ProviderRuntimeOptions,
            state_store: PluginStateStore | None = None,
        ) -> LLMProviderBase:
            providers = self.providers[provider_name]
            provider = FakeProvider(
                provider_name,
                len(providers) + 1,
                options,
                state_store,
            )
            providers.append(provider)
            self.state_stores[provider_name].append(state_store)
            return provider

        return factory


def install_factories(
    monkeypatch: pytest.MonkeyPatch,
    recorder: FactoryRecorder,
) -> None:
    for provider_name, factory_name in FACTORY_NAMES.items():
        monkeypatch.setattr(plugin_module, factory_name, recorder.factory_for(provider_name))


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
