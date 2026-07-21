from pathlib import Path

import pytest

from src.config import MaiDockConfig

from .plugin_test_support import (
    PROVIDER_ENTRIES,
    PROVIDER_NAMES,
    FactoryRecorder,
    create_plugin,
    install_factories,
)


@pytest.fixture
def factory_recorder(monkeypatch: pytest.MonkeyPatch) -> FactoryRecorder:
    recorder = FactoryRecorder()
    install_factories(monkeypatch, recorder)
    return recorder


@pytest.mark.asyncio
async def test_load_initializes_lazy_store_without_creating_providers(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)

    await plugin.on_load()

    assert all(not providers for providers in factory_recorder.providers.values())
    assert not (tmp_path / "data" / "maidock_state.sqlite3").exists()

    await plugin.xiaomi_mimo_provider("response", {})
    state_store = factory_recorder.state_stores["xiaomi_mimo"][0]
    assert state_store is not None
    assert state_store.database_path == tmp_path / "data" / "maidock_state.sqlite3"
    assert not state_store.database_path.exists()
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_load_requires_core_standard_paths(tmp_path: Path) -> None:
    plugin = create_plugin(tmp_path, with_paths=False)

    with pytest.raises(
        RuntimeError,
        match="当前 Core 未提供 MaiDock 所需的插件标准持久化路径",
    ):
        await plugin.on_load()


@pytest.mark.asyncio
async def test_repeated_load_invalidates_all_providers_but_reuses_store(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)
    await plugin.on_load()
    for provider_name in PROVIDER_NAMES:
        await PROVIDER_ENTRIES[provider_name](plugin, "response", {})
    first_store = factory_recorder.state_stores["xiaomi_mimo"][0]

    await plugin.on_load()

    assert all(len(factory_recorder.providers[name]) == 1 for name in PROVIDER_NAMES)
    for provider_name in PROVIDER_NAMES:
        result = await PROVIDER_ENTRIES[provider_name](plugin, "response", {})
        assert result["generation"] == 2

    assert all(len(factory_recorder.providers[name]) == 2 for name in PROVIDER_NAMES)
    assert factory_recorder.state_stores["xiaomi_mimo"][1] is first_store
    assert factory_recorder.state_stores["volcengine"][1] is first_store
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_config_update_rebuilds_all_providers_and_keeps_store(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    before_config = MaiDockConfig()
    before_config.openai_responses.user_agent = "Before-UA/1"
    plugin = create_plugin(
        tmp_path,
        config=before_config.model_dump(mode="python"),
    )
    await plugin.on_load()
    for provider_name in PROVIDER_NAMES:
        await PROVIDER_ENTRIES[provider_name](plugin, "response", {})
    first_store = factory_recorder.state_stores["xiaomi_mimo"][0]
    assert factory_recorder.providers["openai"][0].options.openai_user_agent == ("Before-UA/1")

    after_config = MaiDockConfig()
    after_config.openai_responses.user_agent = "After-UA/1"
    plugin.set_plugin_config(after_config.model_dump(mode="python"))
    await plugin.on_config_update("self", {}, "2")

    assert all(len(factory_recorder.providers[name]) == 1 for name in PROVIDER_NAMES)
    for provider_name in PROVIDER_NAMES:
        result = await PROVIDER_ENTRIES[provider_name](plugin, "response", {})
        assert result["generation"] == 2

    assert factory_recorder.providers["openai"][1].options.openai_user_agent == ("After-UA/1")
    assert factory_recorder.state_stores["xiaomi_mimo"][1] is first_store
    assert factory_recorder.state_stores["volcengine"][1] is first_store
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_unload_is_idempotent_and_reload_uses_new_store(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    unloaded_plugin = create_plugin(tmp_path)
    await unloaded_plugin.on_unload()
    await unloaded_plugin.on_unload()

    plugin = create_plugin(tmp_path)
    await plugin.on_load()
    await plugin.xiaomi_mimo_provider("response", {})
    first_store = factory_recorder.state_stores["xiaomi_mimo"][0]
    assert first_store is not None
    await first_store.set("lifecycle", "key", "value")

    await plugin.on_unload()
    await plugin.on_unload()

    with pytest.raises(RuntimeError, match="关闭"):
        await first_store.get("lifecycle", "key")

    await plugin.on_load()
    result = await plugin.xiaomi_mimo_provider("response", {})
    second_store = factory_recorder.state_stores["xiaomi_mimo"][1]
    assert result["generation"] == 2
    assert second_store is not None
    assert second_store is not first_store
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_volcengine_requires_store_only_when_prefix_cache_is_enabled(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    config = MaiDockConfig()
    config.volcengine_ark.prefix_cache_enabled = True
    cached_plugin = create_plugin(
        tmp_path,
        config=config.model_dump(mode="python"),
    )

    with pytest.raises(RuntimeError, match="持久化存储"):
        await cached_plugin.volcengine_provider("response", {})

    uncached_plugin = create_plugin(tmp_path)
    result = await uncached_plugin.volcengine_provider("response", {})

    assert result["provider"] == "volcengine"
    assert factory_recorder.state_stores["volcengine"] == [None]


@pytest.mark.asyncio
async def test_mimo_requires_load_to_initialize_store(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)

    with pytest.raises(RuntimeError, match="持久化存储"):
        await plugin.xiaomi_mimo_provider("response", {})

    assert factory_recorder.providers["xiaomi_mimo"] == []


@pytest.mark.asyncio
async def test_loaded_volcengine_and_mimo_share_plugin_store(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    config = MaiDockConfig()
    config.volcengine_ark.prefix_cache_enabled = True
    plugin = create_plugin(tmp_path, config=config.model_dump(mode="python"))
    await plugin.on_load()

    await plugin.volcengine_provider("response", {})
    await plugin.xiaomi_mimo_provider("response", {})

    volcengine_store = factory_recorder.state_stores["volcengine"][0]
    mimo_store = factory_recorder.state_stores["xiaomi_mimo"][0]
    assert volcengine_store is not None
    assert mimo_store is volcengine_store
    await plugin.on_unload()
