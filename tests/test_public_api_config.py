import tomllib

from pydantic import ValidationError
import pytest

from src.config import MaiDockConfig
from src.config_schema import build_maidock_config_schema
from src.core.parameter_catalog import provider_catalogs
from src.i18n import Locale
from src.public_api.config import (
    DashScopePublicProfileConfig,
    PublicApiConfig,
    parameter_entries_to_object,
)
from src.public_api.providers.dashscope.contribution import _build_profiles as build_dashscope_profiles


def test_public_api_defaults_are_closed_and_bounded() -> None:
    config = PublicApiConfig()

    assert config.enabled is False
    assert config.default_image_profile == ""
    assert config.default_video_profile == ""
    assert config.resources.max_concurrent_jobs == 2
    assert config.resources.max_queued_jobs == 32
    assert config.resources.max_upload_mb == 512
    assert config.resources.max_artifact_mb == 512
    assert config.resources.storage_quota_gb == 10
    assert config.resources.max_tracking_hours == 23
    assert config.dashscope.profiles == []


def test_dashscope_public_profile_parses_typed_parameter_entries_and_strict_routes() -> None:
    profile = DashScopePublicProfileConfig.model_validate(
        {
            "name": " primary ",
            "api_key": " sk-test ",
            "base_url": "https://dashscope.aliyuncs.com/api/v1/",
            "image_default_parameters": [
                {"name": "size", "value_type": "string", "value": "1024*1024"},
                {"name": "steps", "value_type": "integer", "value": "20"},
                {"name": "cfg", "value_type": "number", "value": "7.5"},
                {"name": "watermark", "value_type": "boolean", "value": "false"},
                {"name": "metadata", "value_type": "json", "value": '{"tags":["a",null]}'},
                {"name": "seed", "value_type": "null", "value": ""},
            ],
            "video_override_parameters": [{"name": "duration", "value_type": "integer", "value": "5"}],
            "protocol_routes": [
                {
                    "capability": "image_generation",
                    "model": " custom-model ",
                    "mode": "text_to_image",
                    "protocol_family": "dashscope_text2image_synthesis",
                }
            ],
        }
    )

    assert profile.name == "primary"
    assert profile.api_key == "sk-test"
    assert profile.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert profile.protocol_routes[0].model == "custom-model"
    assert parameter_entries_to_object(profile.image_default_parameters) == {
        "size": "1024*1024",
        "steps": 20,
        "cfg": 7.5,
        "watermark": False,
        "metadata": {"tags": ["a", None]},
        "seed": None,
    }
    assert parameter_entries_to_object(profile.video_override_parameters) == {"duration": 5}
    assert "sk-test" not in repr(profile)


@pytest.mark.parametrize(
    "update",
    [
        {"base_url": "http://dashscope.aliyuncs.com/api/v1"},
        {"base_url": "https://user:secret@dashscope.aliyuncs.com/api/v1"},
        {"image_default_parameters": "[]"},
        {"image_default_parameters": {"size": "1024*1024"}},
        {"image_default_parameters": [{"name": " ", "value_type": "string", "value": "x"}]},
        {
            "image_default_parameters": [
                {"name": "size", "value_type": "string", "value": "a"},
                {"name": " size ", "value_type": "string", "value": "b"},
            ]
        },
        {"image_default_parameters": [{"name": "x", "value_type": "integer", "value": "true"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "number", "value": "NaN"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "number", "value": "1e999"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "boolean", "value": "True"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "json", "value": "1"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "null", "value": "null"}]},
        {"image_default_parameters": [{"name": "x", "value_type": "string", "value": 1}]},
    ],
)
def test_dashscope_public_profile_rejects_invalid_connection_or_parameter_list(
    update: dict[str, object],
) -> None:
    raw: dict[str, object] = {"name": "main", "api_key": "sk-test"}
    raw.update(update)

    with pytest.raises(ValidationError):
        DashScopePublicProfileConfig.model_validate(raw)


def test_public_api_rejects_duplicate_profile_names() -> None:
    profile = {"name": "same", "api_key": "sk-test"}

    with pytest.raises(ValidationError):
        PublicApiConfig.model_validate({"dashscope": {"profiles": [profile, profile]}})


def test_parameter_entries_round_trip_from_toml_array_tables() -> None:
    raw = tomllib.loads(
        """
        [[dashscope.profiles]]
        name = "main"
        api_key = "sk-test"

        [[dashscope.profiles.image_default_parameters]]
        name = "size"
        value_type = "string"
        value = "1024*1024"

        [[dashscope.profiles.image_default_parameters]]
        name = "n"
        value_type = "integer"
        value = "2"
        """
    )

    config = PublicApiConfig.model_validate(raw)

    assert parameter_entries_to_object(config.dashscope.profiles[0].image_default_parameters) == {
        "size": "1024*1024",
        "n": 2,
    }


def test_dashscope_contribution_converts_parameter_entries_to_runtime_objects() -> None:
    config = PublicApiConfig.model_validate(
        {
            "dashscope": {
                "profiles": [
                    {
                        "name": "main",
                        "api_key": "sk-test",
                        "image_default_parameters": [{"name": "size", "value_type": "string", "value": "1024*1024"}],
                        "image_override_parameters": [{"name": "watermark", "value_type": "boolean", "value": "false"}],
                    }
                ]
            }
        }
    )

    profiles, _fingerprints = build_dashscope_profiles(config)

    assert profiles[0].image_default_parameters == {"size": "1024*1024"}
    assert profiles[0].image_override_parameters == {"watermark": False}


def test_public_api_webui_catalog_uses_plain_key_and_structured_routes() -> None:
    schema = build_maidock_config_schema(locale="zh-CN")
    public_tab = next(tab for tab in schema["layout"]["tabs"] if tab["id"] == "public_api")
    sections = schema["sections"]
    profile_fields = sections["public_api_dashscope"]["fields"]["profiles"]["item_fields"]

    assert public_tab["title"] == "跨插件 API"
    assert public_tab["sections"] == [
        "public_api",
        "public_api_resources",
        "public_api_dashscope",
        "public_api_volcengine_ark",
    ]
    assert profile_fields["api_key"]["ui_type"] == "text"
    assert profile_fields["api_key"]["input_type"] is None
    parameter_field = profile_fields["image_default_parameters"]
    assert parameter_field["type"] == "array"
    assert parameter_field["default"] == []
    assert parameter_field["item_type"] == "object"
    assert set(parameter_field["item_fields"]) == {"name", "value_type", "value"}
    assert parameter_field["item_fields"]["value_type"]["type"] == "select"
    assert parameter_field["item_fields"]["value_type"]["choices"] == [
        "string",
        "integer",
        "number",
        "boolean",
        "json",
        "null",
    ]
    assert profile_fields["protocol_routes"]["item_type"] == "object"
    assert set(profile_fields["protocol_routes"]["item_fields"]) == {
        "capability",
        "model",
        "mode",
        "protocol_family",
    }
    assert profile_fields["protocol_routes"]["item_fields"]["capability"]["type"] == "select"
    assert profile_fields["protocol_routes"]["item_fields"]["protocol_family"]["type"] == "select"


def test_public_api_webui_catalog_exposes_ark_section() -> None:
    schema = build_maidock_config_schema(locale="zh-CN")
    fields = schema["sections"]["public_api_volcengine_ark"]["fields"]["profiles"]["item_fields"]

    assert fields["base_url"]["default"] == "https://ark.cn-beijing.volces.com/api/v3"
    # ARK 靠 API Key 本身区分租户，没有 DashScope 的 workspace_id。
    assert "workspace_id" not in fields
    assert fields["video_override_parameters"]["item_fields"]["value_type"]["type"] == "select"
    assert set(fields["protocol_routes"]["item_fields"]) == {
        "capability",
        "model",
        "mode",
        "protocol_family",
    }


@pytest.mark.parametrize(
    ("locale", "tab_title", "api_key_label", "parameter_name_label"),
    [
        ("zh-CN", "跨插件 API", "DashScope API Key（明文）", "参数名"),
        ("zh-TW", "跨外掛 API", "DashScope API Key（明文）", "參數名稱"),
        ("en-US", "Cross-plugin API", "DashScope API Key (plaintext)", "Parameter name"),
        ("ja-JP", "プラグイン間 API", "DashScope API Key（平文）", "パラメータ名"),
        ("ko-KR", "플러그인 간 API", "DashScope API Key(평문)", "매개변수 이름"),
    ],
)
def test_public_api_webui_catalog_is_localized(
    locale: Locale,
    tab_title: str,
    api_key_label: str,
    parameter_name_label: str,
) -> None:
    schema = build_maidock_config_schema(locale=locale)
    public_tab = next(tab for tab in schema["layout"]["tabs"] if tab["id"] == "public_api")
    profile_fields = schema["sections"]["public_api_dashscope"]["fields"]["profiles"]["item_fields"]

    assert public_tab["title"] == tab_title
    assert profile_fields["api_key"]["label"] == api_key_label
    parameter_fields = profile_fields["image_default_parameters"]["item_fields"]
    assert parameter_fields["name"]["label"] == parameter_name_label


def test_host_dashscope_config_and_parameter_catalog_have_no_image_generation_policy() -> None:
    host_config = MaiDockConfig().dashscope.model_dump()

    assert "image_generation" not in host_config
    assert all(catalog.capability != "image_generation" for catalog in provider_catalogs("dashscope"))
