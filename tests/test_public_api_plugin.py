from pathlib import Path
from typing import Any, cast

import hashlib

import pytest
from maibot_sdk import PluginContext

from src.plugin import MaiDockPlugin
from src.public_api.api.responses import ErrorEnvelope
from src.version import __version__

API_NAMES = {
    "media.capabilities",
    "media.jobs.create",
    "media.jobs.get",
    "media.jobs.cancel",
    "media.jobs.delete",
    "media.uploads.create",
    "media.uploads.upload",
    "media.uploads.get",
    "media.uploads.write_chunk",
    "media.uploads.complete",
    "media.uploads.delete",
    "media.artifacts.read",
}


class FakePaths:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.runtime_dir = data_dir / "runtime"


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], str]] = []

    async def replace_dynamic_apis(
        self,
        components: list[dict[str, Any]],
        *,
        offline_reason: str,
    ) -> bool:
        self.calls.append((components, offline_reason))
        return True


class FakeContext:
    def __init__(self, data_dir: Path) -> None:
        self.paths = FakePaths(data_dir)
        self.api = FakeApi()


def _config(*, enabled: bool = True, plugin_enabled: bool = True, locale: str = "zh-CN") -> dict[str, Any]:
    return {
        "plugin": {"enabled": plugin_enabled, "config_version": __version__, "locale": locale},
        "public_api": {
            "enabled": enabled,
            "default_image_profile": "main",
            "default_video_profile": "main",
            "dashscope": {
                "profiles": [
                    {
                        "name": "main",
                        "api_key": "dashscope-secret",
                        "default_image_model": "wan2.7-image",
                        "default_video_model": "wan2.7-t2v",
                        "retry_interval_seconds": 0,
                    }
                ]
            },
        },
    }


def _plugin(tmp_path: Path, config: dict[str, Any]) -> tuple[MaiDockPlugin, FakeContext]:
    plugin = MaiDockPlugin()
    plugin.set_plugin_config(config)
    context = FakeContext(tmp_path / "data")
    plugin._set_context(cast(PluginContext, context))
    return plugin, context


async def _invoke(plugin: MaiDockPlugin, name: str, **kwargs: object) -> dict[str, Any]:
    return await plugin.invoke_component(f"dynamic_api__{name}__1", **kwargs)


@pytest.mark.asyncio
async def test_dynamic_apis_are_public_versioned_and_capabilities_are_redacted(tmp_path: Path) -> None:
    plugin, context = _plugin(tmp_path, _config())
    await plugin.on_load()
    try:
        components = plugin.get_dynamic_api_components()
        assert {component["name"] for component in components} == API_NAMES
        assert all(component["metadata"]["version"] == "1" for component in components)
        assert all(component["metadata"]["public"] is True for component in components)
        assert all(component["metadata"]["timeout_ms"] == 25_000 for component in components)
        assert context.api.calls[-1][0] == components

        envelope = await _invoke(plugin, "media.capabilities", request={})
        assert envelope["ok"] is True
        assert envelope["error"] is None
        assert envelope["data"]["profiles"] == [
            {"name": "main", "provider": "dashscope", "driver": "dashscope.media.v1"}
        ]
        serialized = repr(envelope["data"]).lower()
        assert "dashscope-secret" not in serialized
        assert "api_key" not in serialized
        assert "base_url" not in serialized
        assert "workspace" not in serialized
    finally:
        await plugin.on_unload()


@pytest.mark.asyncio
async def test_upload_bytes_and_strict_unified_submit_envelope(tmp_path: Path) -> None:
    plugin, _context = _plugin(tmp_path, _config())
    await plugin.on_load()
    try:
        data = b"abcdef"
        uploaded = await _invoke(
            plugin,
            "media.uploads.upload",
            request={
                "media_type": "image/png",
                "data": data,
                "sha256": hashlib.sha256(data).hexdigest(),
                "file_name": "source.png",
            },
        )
        assert uploaded["ok"] is True
        assert uploaded["data"]["status"] == "complete"

        created = await _invoke(
            plugin,
            "media.uploads.create",
            request={
                "media_type": "video/mp4",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        upload_id = created["data"]["upload_id"]
        wrong_type = await _invoke(
            plugin,
            "media.uploads.write_chunk",
            request={"upload_id": upload_id, "offset": 0, "data": bytearray(data)},
        )
        assert wrong_type["error"]["code"] == "INVALID_REQUEST"
        written = await _invoke(
            plugin,
            "media.uploads.write_chunk",
            request={"upload_id": upload_id, "offset": 0, "data": data},
        )
        assert written["ok"] is True
        completed = await _invoke(
            plugin,
            "media.uploads.complete",
            request={"upload_id": upload_id},
        )
        assert completed["data"]["status"] == "complete"

        invalid = await _invoke(
            plugin,
            "media.jobs.create",
            request={
                "capability": "image_generation",
                "mode": "text_to_image",
                "prompt": "生成",
                "unexpected": True,
            },
        )
        assert invalid["ok"] is False
        assert invalid["data"] is None
        assert invalid["error"]["code"] == "INVALID_REQUEST"
        assert invalid["error"]["retryable"] is False
        assert invalid["error"]["uncertain"] is False
        assert invalid["error"]["provider_request_id"] is None
    finally:
        await plugin.on_unload()


@pytest.mark.asyncio
async def test_disabling_unpublishes_all_handlers_and_stops_accepting(tmp_path: Path) -> None:
    plugin, context = _plugin(tmp_path, _config())
    await plugin.on_load()
    try:
        plugin.set_plugin_config(_config(enabled=False))
        await plugin.on_config_update("plugin", {}, "1")
        assert plugin.get_dynamic_api_components() == []
        assert context.api.calls[-1][0] == []
        assert context.api.calls[-1][1] == "MaiDock 跨插件 API 已关闭"
        with pytest.raises(AttributeError, match="未注册动态组件"):
            await _invoke(plugin, "media.capabilities", request={})
        runtime = plugin._public_api_runtime
        assert runtime is not None
        disabled = await runtime.require_facade().create_job(
            {"capability": "image_generation", "mode": "text_to_image", "prompt": "生成"}
        )
        assert ErrorEnvelope.model_validate(disabled).error.code == "MEDIA_API_DISABLED"
    finally:
        await plugin.on_unload()
