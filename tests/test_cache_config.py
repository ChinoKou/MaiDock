import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import MaiDockConfig, build_runtime_options
from src.config_schema import build_maidock_config_schema
from src.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_ark_prefix_cache_defaults_and_runtime_options() -> None:
    config = MaiDockConfig()

    assert config.volcengine_ark.prefix_cache_enabled is False
    assert config.volcengine_ark.prefix_cache_ttl_seconds == 259200
    options = build_runtime_options(config)
    assert options.volcengine_prefix_cache_enabled is False
    assert options.volcengine_prefix_cache_ttl_seconds == 259200


@pytest.mark.parametrize("ttl_seconds", [3599, 604801])
def test_ark_prefix_cache_ttl_validation(ttl_seconds: int) -> None:
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"volcengine_ark": {"prefix_cache_ttl_seconds": ttl_seconds}})


def test_ark_prefix_cache_webui_fields_follow_scalar_schema_style() -> None:
    schema = build_maidock_config_schema(plugin_id="chinokou.maidock")
    fields = schema["sections"]["volcengine_ark"]["fields"]

    enabled = fields["prefix_cache_enabled"]
    ttl = fields["prefix_cache_ttl_seconds"]
    assert enabled["ui_type"] == "switch"
    assert enabled["default"] is False
    assert ttl["ui_type"] == "number"
    assert ttl["default"] == 259200
    assert ttl["min"] == 3600
    assert ttl["max"] == 604800
    assert ttl["step"] == 3600


def test_version_and_config_template_metadata_are_synchronized() -> None:
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))

    assert config["plugin"]["config_version"] == __version__
    assert config["volcengine_ark"]["prefix_cache_enabled"] is False
    assert config["volcengine_ark"]["prefix_cache_ttl_seconds"] == 259200
    assert pyproject["project"]["version"] == __version__
    assert manifest["version"] == __version__
    assert manifest["host_application"]["min_version"] == "1.0.9"
    assert manifest["sdk"]["min_version"] == "2.7.0"
