import importlib.util
import json
import sys
from pathlib import Path

import pytest
from maibot_sdk import LLMProviderBase

from src import plugin as plugin_module
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "_manifest.json"
PLUGIN_ENTRY_PATH = Path(__file__).resolve().parents[1] / "plugin.py"


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
        "maidock-siliconflow",
        "maidock-xiaomi-mimo",
    }


@pytest.mark.parametrize(
    ("factory", "needs_store"),
    [
        pytest.param(plugin_module._create_openai_provider, False, id="openai"),
        pytest.param(plugin_module._create_anthropic_provider, False, id="anthropic"),
        pytest.param(
            plugin_module._create_volcengine_provider,
            True,
            id="volcengine",
        ),
        pytest.param(plugin_module._create_dashscope_provider, False, id="dashscope"),
        pytest.param(
            plugin_module._create_siliconflow_provider,
            False,
            id="siliconflow",
        ),
        pytest.param(plugin_module._create_mimo_provider, True, id="xiaomi-mimo"),
    ],
)
def test_factory_local_imports_return_provider_base_instances(
    factory: object,
    needs_store: bool,
    tmp_path: Path,
) -> None:
    options = ProviderRuntimeOptions()
    state_store = PluginStateStore(tmp_path / "state.sqlite3")

    if needs_store:
        assert callable(factory)
        provider = factory(options, state_store)
    else:
        assert callable(factory)
        provider = factory(options)

    assert isinstance(provider, LLMProviderBase)
