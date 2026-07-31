from pathlib import Path
from typing import Any, cast

import importlib.util
import json
import sys

from maibot_sdk import LLMProviderBase
import pytest

from src import plugin as plugin_module
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.host_adapters.common.options import (
    AnthropicHostOptions,
    ArkHostOptions,
    BailianHostOptions,
    DashScopeHostOptions,
    HostCommonOptions,
    MimoHostOptions,
    OpenAIHostOptions,
    SiliconFlowHostOptions,
)
from src.runtime import CLIENT_KEY_BY_RUNTIME, RuntimeKey, create_vendor_client, create_vendor_runtime

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "_manifest.json"
PLUGIN_ENTRY_PATH = Path(__file__).resolve().parents[1] / "plugin.py"
VENDOR_OPTION_TYPES: dict[RuntimeKey, type[object]] = {
    "openai": OpenAIHostOptions,
    "anthropic": AnthropicHostOptions,
    "volcengine": ArkHostOptions,
    "dashscope": DashScopeHostOptions,
    "bailian_responses": BailianHostOptions,
    "siliconflow": SiliconFlowHostOptions,
    "xiaomi_mimo": MimoHostOptions,
}


def _manifest_client_types() -> set[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    providers = manifest.get("llm_providers")
    assert isinstance(providers, list)
    client_types: set[str] = set()
    for provider in providers:
        assert isinstance(provider, dict)
        client_type = provider.get("client_type")
        assert isinstance(client_type, str)
        client_types.add(client_type)
    return client_types


def test_root_plugin_entry_exports_create_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "maidock_test_entry",
        PLUGIN_ENTRY_PATH,
        submodule_search_locations=[str(PLUGIN_ENTRY_PATH.parent)],
    )
    assert spec is not None
    assert spec.loader is not None
    entry_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, entry_module)
    monkeypatch.setitem(sys.modules, f"{spec.name}.src", sys.modules["src"])
    monkeypatch.setitem(sys.modules, f"{spec.name}.src.plugin", plugin_module)

    spec.loader.exec_module(entry_module)

    assert entry_module.create_plugin is plugin_module.create_plugin
    assert entry_module.__all__ == ["create_plugin"]


def test_create_plugin_returns_fresh_plugin_instances() -> None:
    first = plugin_module.create_plugin()
    second = plugin_module.create_plugin()

    assert isinstance(first, plugin_module.MaiDockPlugin)
    assert isinstance(second, plugin_module.MaiDockPlugin)
    assert first is not second


def test_llm_provider_metadata_matches_manifest() -> None:
    plugin = plugin_module.MaiDockPlugin()

    registered_client_types = {str(provider["client_type"]) for provider in plugin.get_llm_providers()}

    assert registered_client_types == _manifest_client_types()
    assert registered_client_types == {
        "maidock-openai-responses",
        "maidock-anthropic-messages",
        "maidock-volcengine-ark-responses",
        "maidock-dashscope",
        "maidock-bailian-responses",
        "maidock-siliconflow",
        "maidock-xiaomi-mimo",
    }


@pytest.mark.parametrize(
    "provider_key",
    [
        pytest.param("openai", id="openai"),
        pytest.param("anthropic", id="anthropic"),
        pytest.param("volcengine", id="volcengine"),
        pytest.param("dashscope", id="dashscope"),
        pytest.param("bailian_responses", id="bailian-responses"),
        pytest.param("siliconflow", id="siliconflow"),
        pytest.param("xiaomi_mimo", id="xiaomi-mimo"),
    ],
)
@pytest.mark.asyncio
async def test_factory_returns_vendor_runtime_with_sdk_ingress(
    provider_key: RuntimeKey,
    tmp_path: Path,
) -> None:
    options = ProviderRuntimeOptions()
    state_store = PluginStateStore(tmp_path / "state.sqlite3")

    client = create_vendor_client(CLIENT_KEY_BY_RUNTIME[provider_key])
    runtime = create_vendor_runtime(provider_key, options, state_store, client)

    assert isinstance(runtime.ingress, LLMProviderBase)
    host_adapter = cast(Any, runtime.host_adapter)
    assert isinstance(host_adapter.options, HostCommonOptions)
    assert isinstance(host_adapter.vendor_options, VENDOR_OPTION_TYPES[provider_key])
    assert runtime.client is client
    await client.aclose()
    await state_store.close()
