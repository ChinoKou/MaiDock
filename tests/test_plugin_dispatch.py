import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from src import plugin as plugin_module
from src.config import MaiDockConfig
from src.core.common import ProviderRuntimeOptions
from src.i18n import translate
from src.runtime import HostAdapter, LLMProviderIngress, ProviderCapability, VendorRuntime

from .plugin_test_support import (
    PROVIDER_ENTRIES,
    PROVIDER_NAMES,
    FactoryRecorder,
    FakeClient,
    ProviderName,
    create_plugin,
    install_factories,
)


def _runtime(
    adapter: object,
    *,
    capabilities: frozenset[ProviderCapability] = frozenset({"response", "embedding", "audio_transcription"}),
    provider_name: str = "Test Adapter",
) -> VendorRuntime:
    host_adapter = cast(HostAdapter, adapter)
    client = FakeClient()
    ingress = LLMProviderIngress(
        adapter=host_adapter,
        capabilities=capabilities,
        provider_name=provider_name,
    )
    return VendorRuntime(client=client, host_adapter=host_adapter, ingress=ingress)


@pytest.fixture
def factory_recorder(monkeypatch: pytest.MonkeyPatch) -> FactoryRecorder:
    recorder = FactoryRecorder()
    install_factories(monkeypatch, recorder)
    return recorder


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_provider_entry_forwards_all_operations_and_reuses_instance(
    provider_name: ProviderName,
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)
    await plugin.on_load()
    entry = PROVIDER_ENTRIES[provider_name]
    requests = [
        {"request": "response"},
        {"request": "embedding"},
        {"request": "audio"},
    ]

    results = [
        await entry(plugin, "response", requests[0]),
        await entry(plugin, "embedding", requests[1]),
        await entry(plugin, "audio_transcription", requests[2]),
    ]

    assert results == [
        {"provider": provider_name, "operation": "response", "generation": 1},
        {"provider": provider_name, "operation": "embedding", "generation": 1},
        {
            "provider": provider_name,
            "operation": "audio_transcription",
            "generation": 1,
        },
    ]
    assert len(factory_recorder.providers[provider_name]) == 1
    provider = factory_recorder.providers[provider_name][0]
    assert provider.operations == ["response", "embedding", "audio_transcription"]
    assert provider.requests == requests
    await plugin.on_unload()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_concurrent_first_calls_share_one_provider_instance(
    provider_name: ProviderName,
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)
    await plugin.on_load()
    entry = PROVIDER_ENTRIES[provider_name]

    results = await asyncio.gather(
        entry(plugin, "response", {"request": 1}),
        entry(plugin, "response", {"request": 2}),
    )

    assert [result["generation"] for result in results] == [1, 1]
    assert len(factory_recorder.providers[provider_name]) == 1
    await plugin.on_unload()


@pytest.mark.asyncio
async def test_concurrent_provider_entries_keep_independent_instances(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)
    await plugin.on_load()

    results = await asyncio.gather(
        *(PROVIDER_ENTRIES[name](plugin, "response", {"provider": name}) for name in PROVIDER_NAMES)
    )

    assert {result["provider"] for result in results} == set(PROVIDER_NAMES)
    assert all(len(factory_recorder.providers[name]) == 1 for name in PROVIDER_NAMES)
    await plugin.on_unload()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", PROVIDER_NAMES)
async def test_invalid_operation_is_rejected_before_provider_creation(
    provider_name: ProviderName,
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    plugin = create_plugin(tmp_path)

    with pytest.raises(ValueError, match="invalid_operation"):
        await PROVIDER_ENTRIES[provider_name](plugin, "invalid_operation", {})

    assert all(not providers for providers in factory_recorder.providers.values())


@pytest.mark.asyncio
async def test_disabled_plugin_rejects_request_before_provider_creation(
    factory_recorder: FactoryRecorder,
    tmp_path: Path,
) -> None:
    config = MaiDockConfig()
    config.plugin.enabled = False
    plugin = create_plugin(tmp_path, config=config.model_dump(mode="python"))

    with pytest.raises(RuntimeError, match="禁用"):
        await plugin.openai_responses_provider("response", {})

    assert all(not providers for providers in factory_recorder.providers.values())


@pytest.mark.asyncio
async def test_sdk_unsupported_capability_is_localized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ResponseOnlyAdapter:
        async def get_response(self, request: dict[str, Any]) -> dict[str, Any]:
            return request

    monkeypatch.setattr(
        plugin_module,
        "create_vendor_runtime",
        lambda key, options, store, client: _runtime(
            ResponseOnlyAdapter(),
            capabilities=frozenset({"response"}),
            provider_name="ResponseOnlyAdapter",
        ),
    )
    plugin = create_plugin(tmp_path)

    with pytest.raises(NotImplementedError, match="ResponseOnlyAdapter"):
        await plugin.openai_responses_provider("embedding", {})


@pytest.mark.asyncio
async def test_provider_boundary_rewrites_pydantic_error_in_runtime_locale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RequestModel(BaseModel):
        count: int

    class ValidationAdapter:
        async def get_response(self, request: dict[str, Any]) -> dict[str, Any]:
            RequestModel.model_validate(request)
            return {}

    monkeypatch.setattr(
        plugin_module,
        "create_vendor_runtime",
        lambda key, options, store, client: _runtime(ValidationAdapter()),
    )
    config = MaiDockConfig.model_validate({"plugin": {"locale": "ja-JP"}})
    plugin = create_plugin(tmp_path, config=config.model_dump(mode="python"))

    with pytest.raises(ValueError) as captured:
        await plugin.openai_responses_provider("response", {"count": "invalid"})

    message = str(captured.value)
    assert "count" in message
    assert "int_parsing" in message
    assert "整数として解析できません" in message
    assert "invalid" not in message


@pytest.mark.asyncio
async def test_provider_boundary_rejects_non_object_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InvalidResultAdapter:
        async def get_response(self, request: dict[str, Any]) -> Any:
            del request
            return ["not", "an", "object"]

    monkeypatch.setattr(
        plugin_module,
        "create_vendor_runtime",
        lambda key, options, store, client: _runtime(InvalidResultAdapter()),
    )
    plugin = create_plugin(tmp_path)

    with pytest.raises(TypeError, match="映射对象"):
        await plugin.openai_responses_provider("response", {})


@pytest.mark.asyncio
async def test_config_update_keeps_inflight_locale_and_new_requests_use_new_locale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release_old = asyncio.Event()

    class LocaleAdapter:
        def __init__(self, options: ProviderRuntimeOptions) -> None:
            self.options = options

        async def get_response(self, request: dict[str, Any]) -> dict[str, Any]:
            del request
            before = translate("ui.tab.general")
            if self.options.locale == "zh-CN":
                started.set()
                await release_old.wait()
            after = translate("ui.tab.general")
            return {"before": before, "after": after, "locale": self.options.locale}

    monkeypatch.setattr(
        plugin_module,
        "create_vendor_runtime",
        lambda key, options, store, client: _runtime(LocaleAdapter(options)),
    )
    before_config = MaiDockConfig.model_validate({"plugin": {"locale": "zh-CN"}})
    plugin = create_plugin(tmp_path, config=before_config.model_dump(mode="python"))

    old_request = asyncio.create_task(plugin.openai_responses_provider("response", {}))
    await started.wait()

    after_config = MaiDockConfig.model_validate({"plugin": {"locale": "en-US"}})
    plugin.set_plugin_config(after_config.model_dump(mode="python"))
    await plugin.on_config_update("self", {}, "2")
    new_result = await plugin.openai_responses_provider("response", {})

    release_old.set()
    old_result = await old_request

    assert old_result == {"before": "通用", "after": "通用", "locale": "zh-CN"}
    assert new_result == {"before": "General", "after": "General", "locale": "en-US"}
