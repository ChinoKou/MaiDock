import asyncio
import json
import logging
import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from src.config import (
    MaiDockConfig,
    build_runtime_options,
    normalize_maidock_config_data,
)
from src.config_schema import build_maidock_config_schema
from src.core.common import (
    ProviderRuntimeOptions,
    log_request_summary,
    normalize_auth_type,
)
from src.core.diagnostics import (
    build_parse_error_message,
    build_status_error_message,
    sanitize_upstream_detail,
)
from src.core.parameter_catalog import dotted_path, iter_parameter_catalogs
from src.core.state_store import PluginStateStore
from src.i18n import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    Locale,
    TranslationCatalogError,
    _load_catalogs_from_directory,
    format_validation_error,
    normalize_locale,
    translate,
    use_locale,
    validate_catalogs,
)
from src.plugin import MaiDockPlugin
from src.host_adapters.common.embeddings import coerce_embedding_vector
from src.host_adapters.volcengine_ark_provider.prefix_cache import PrefixCacheManager
from src.host_adapters.xiaomi_mimo_provider.reasoning import MimoReasoningManager

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class _ValidationTarget(BaseModel):
    count: int


class _UnknownValidationTarget(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def reject_value(cls, value: str) -> str:
        raise PydanticCustomError("custom_code", "framework detail: {value}", {"value": value})


def test_catalogs_are_complete_and_valid() -> None:
    validate_catalogs()


def _copy_catalogs(target: Path) -> None:
    target.mkdir()
    for locale in SUPPORTED_LOCALES:
        source = PLUGIN_ROOT / "locales" / f"{locale}.json"
        (target / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


@pytest.mark.parametrize(
    "failure",
    [
        "missing_file",
        "invalid_json",
        "duplicate_key",
        "empty_text",
        "missing_key",
        "extra_key",
        "placeholder",
    ],
)
def test_catalog_validation_rejects_all_invalid_shapes(tmp_path: Path, failure: str) -> None:
    catalog_dir = tmp_path / "locales"
    _copy_catalogs(catalog_dir)
    target = catalog_dir / "en-US.json"

    if failure == "missing_file":
        target.unlink()
    elif failure == "invalid_json":
        target.write_text("{", encoding="utf-8")
    elif failure == "duplicate_key":
        target.write_text('{"duplicate":"one","duplicate":"two"}', encoding="utf-8")
    else:
        catalog = json.loads(target.read_text(encoding="utf-8"))
        if failure == "empty_text":
            catalog["ui.tab.general"] = ""
        elif failure == "missing_key":
            del catalog["ui.tab.general"]
        elif failure == "extra_key":
            catalog["unexpected.key"] = "unexpected"
        else:
            catalog["runtime.log.retry_status"] = "bad {unexpected}"
        target.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(TranslationCatalogError):
        _load_catalogs_from_directory(catalog_dir)


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_supported_locales_are_accepted(locale: str) -> None:
    assert normalize_locale(locale) == locale


def test_unknown_locale_and_message_key_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="xx-XX"):
        normalize_locale("xx-XX")
    with pytest.raises(TranslationCatalogError, match="missing.message"):
        translate("missing.message")


@pytest.mark.parametrize(
    ("locale", "tab_title", "locale_label"),
    [
        (
            "zh-CN",
            "通用",
            "MaiDock 显示与日志语言",
        ),
        (
            "zh-TW",
            "通用",
            "MaiDock 顯示與日誌語言",
        ),
        (
            "en-US",
            "General",
            "MaiDock display and log language",
        ),
        (
            "ja-JP",
            "全般",
            "MaiDock の表示・ログ言語",
        ),
        (
            "ko-KR",
            "일반",
            "MaiDock 표시 및 로그 언어",
        ),
    ],
)
def test_schema_localizes_static_and_dynamic_fields(
    locale: str,
    tab_title: str,
    locale_label: str,
) -> None:
    schema = build_maidock_config_schema(locale=normalize_locale(locale))

    assert schema["layout"]["tabs"][0]["title"] == tab_title
    assert schema["sections"]["plugin"]["fields"]["locale"]["label"] == locale_label
    dynamic_field = schema["sections"]["dashscope_chat_completion_overrides"]["fields"]
    assert dynamic_field["enable_thinking"]["label"] == "enable_thinking · boolean"


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_all_dynamic_provider_fields_are_localized(
    locale: str,
) -> None:
    schema = build_maidock_config_schema(locale=normalize_locale(locale))

    for catalog in iter_parameter_catalogs():
        fields = schema["sections"][f"{catalog.provider}_{catalog.capability}_overrides"]["fields"]
        for parameter in catalog.fields:
            override_value = fields[parameter.config_key]
            assert override_value["label"].startswith(f"{parameter.key} · ")
            assert dotted_path(parameter.target_path) in override_value["hint"]
            assert catalog.documentation_url in override_value["hint"]
            assert override_value["hint"].count("\n") == 5


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_all_schema_sections_and_visible_field_labels_are_populated(
    locale: Locale,
) -> None:
    schema = build_maidock_config_schema(locale=locale)

    for tab in schema["layout"]["tabs"]:
        assert tab["title"].strip()
    for section in schema["sections"].values():
        assert section["title"].strip()
        assert section["description"].strip()
        for field in section["fields"].values():
            assert field["label"].strip()
            assert field["description"].strip()


def test_config_default_normalization_and_runtime_locale() -> None:
    default_config = MaiDockConfig.model_validate({"plugin": {"enabled": True}})
    english_config = MaiDockConfig.model_validate({"plugin": {"locale": "en-US"}})
    normalized, changed = normalize_maidock_config_data({"plugin": {"enabled": True}})

    assert default_config.plugin.locale == DEFAULT_LOCALE
    assert build_runtime_options(english_config).locale == "en-US"
    assert changed is True
    assert normalized["plugin"]["locale"] == DEFAULT_LOCALE
    with pytest.raises(ValidationError):
        MaiDockConfig.model_validate({"plugin": {"locale": "xx-XX"}})


def test_plugin_schema_uses_updated_config_locale() -> None:
    plugin = MaiDockPlugin()
    plugin.set_plugin_config(MaiDockConfig().model_dump(mode="python"))
    assert _first_schema_tab_title(plugin.get_webui_config_schema()) == "通用"

    updated = MaiDockConfig.model_validate({"plugin": {"locale": "en-US"}})
    plugin.set_plugin_config(updated.model_dump(mode="python"))
    assert _first_schema_tab_title(plugin.get_webui_config_schema()) == "General"


def _first_schema_tab_title(schema: dict[str, object]) -> str:
    layout = schema["layout"]
    assert isinstance(layout, dict)
    tabs = layout["tabs"]
    assert isinstance(tabs, list)
    first_tab = tabs[0]
    assert isinstance(first_tab, dict)
    title = first_tab["title"]
    assert isinstance(title, str)
    return title


@pytest.mark.asyncio
async def test_locale_context_is_isolated_between_tasks() -> None:
    async def render(locale: str) -> str:
        with use_locale(normalize_locale(locale)):
            await asyncio.sleep(0)
            return translate("ui.tab.general")

    assert await asyncio.gather(render("en-US"), render("ja-JP"), render("ko-KR")) == [
        "General",
        "全般",
        "일반",
    ]


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("zh-CN", "请求"),
        ("zh-TW", "請求"),
        ("en-US", "request"),
        ("ja-JP", "リクエスト"),
        ("ko-KR", "요청"),
    ],
)
def test_request_log_uses_bound_locale(caplog: pytest.LogCaptureFixture, locale: str, expected: str) -> None:
    logger = logging.getLogger(f"maidock.i18n.{locale}")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        use_locale(normalize_locale(locale)),
    ):
        log_request_summary(
            logger,
            provider_label="Test Provider",
            model="test-model",
            options=ProviderRuntimeOptions(locale=normalize_locale(locale)),
        )

    assert expected in caplog.text
    assert "Test Provider" in caplog.text
    assert "test-model" in caplog.text


@pytest.mark.parametrize(
    ("locale", "description"),
    [
        ("zh-CN", "无法解析为整数"),
        ("zh-TW", "無法解析為整數"),
        ("en-US", "cannot be parsed as an integer"),
        ("ja-JP", "整数として解析できません"),
        ("ko-KR", "정수로 파싱할 수 없습니다"),
    ],
)
def test_pydantic_error_is_rewritten_with_localized_description(locale: str, description: str) -> None:
    with pytest.raises(ValidationError) as captured:
        _ValidationTarget.model_validate({"count": "not-an-int"})

    with use_locale(normalize_locale(locale)):
        message = format_validation_error(captured.value)

    assert "count" in message
    assert "int_parsing" in message
    assert description in message
    assert "not-an-int" not in message


def test_pydantic_missing_and_unknown_codes_have_safe_descriptions() -> None:
    with pytest.raises(ValidationError) as missing_error:
        _ValidationTarget.model_validate({})
    with pytest.raises(ValidationError) as unknown_error:
        _UnknownValidationTarget.model_validate({"value": "sensitive-input"})

    with use_locale("en-US"):
        missing_message = format_validation_error(missing_error.value)
        unknown_message = format_validation_error(unknown_error.value)

    assert "missing" in missing_message
    assert "required field is missing" in missing_message
    assert "custom_code" in unknown_message
    assert "value is invalid" in unknown_message
    assert "sensitive-input" not in unknown_message
    assert "framework detail" not in unknown_message


@pytest.mark.parametrize(
    ("locale", "finite_number"),
    [
        ("zh-CN", "有限数值"),
        ("zh-TW", "有限數值"),
        ("en-US", "finite number"),
        ("ja-JP", "有限数"),
        ("ko-KR", "유한한 수"),
    ],
)
def test_embedding_error_localizes_maidock_semantics(locale: str, finite_number: str) -> None:
    with use_locale(normalize_locale(locale)), pytest.raises(ValueError) as captured:
        coerce_embedding_vector([float("nan")], provider_label="Test Provider")

    message = str(captured.value)
    assert finite_number in message
    assert "embedding[0]" in message
    if locale != "en-US":
        assert "finite number" not in message


def test_upstream_details_are_redacted_and_truncated() -> None:
    long_base64 = "A" * 512
    detail = {
        "Authorization": "Bearer structured-secret",
        "api_key": "structured-key",
        "prompt": "P" * 600,
        "audio_base64": long_base64,
        "image": f"data:image/png;base64,{long_base64}",
    }
    sanitized = sanitize_upstream_detail(detail)
    raw_text = sanitize_upstream_detail('Authorization: Bearer raw-secret api_key="raw-key" token=raw-token')

    for secret in (
        "structured-secret",
        "structured-key",
        "raw-secret",
        "raw-key",
        "raw-token",
        long_base64,
    ):
        assert secret not in sanitized
        assert secret not in raw_text
    assert "***" in sanitized
    assert "***" in raw_text
    assert "<truncated:" in sanitized
    assert "<base64:" in sanitized


def test_status_error_uses_sanitized_message_and_body() -> None:
    class UpstreamError(RuntimeError):
        status_code = 400
        message = "Authorization: Bearer message-secret"
        body = {"api_key": "body-secret", "code": "UPSTREAM_BAD_REQUEST"}

    message = build_status_error_message("Test Provider", UpstreamError("token=exception-secret"))

    assert "UPSTREAM_BAD_REQUEST" in message
    assert "***" in message
    assert "message-secret" not in message
    assert "body-secret" not in message
    assert "exception-secret" not in message


@pytest.mark.parametrize("auth_type", ("none", "bearer", "header", "query"))
def test_auth_type_normalization_accepts_all_supported_values(auth_type: str) -> None:
    assert normalize_auth_type(auth_type) == auth_type


def test_auth_type_error_lists_the_actual_supported_values() -> None:
    with pytest.raises(ValueError, match="none/bearer/header/query"):
        normalize_auth_type("api_key")


def test_parse_error_adds_provider_context_once() -> None:
    with use_locale("en-US"):
        reason = translate("runtime.error.output_missing", item="text")
        message = build_parse_error_message("Test Provider", reason)

    assert message.count("Test Provider") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "marker"),
    [
        ("zh-CN", "禁用"),
        ("zh-TW", "停用"),
        ("en-US", "disabled"),
        ("ja-JP", "無効"),
        ("ko-KR", "비활성화"),
    ],
)
async def test_disabled_plugin_error_uses_runtime_locale(locale: str, marker: str) -> None:
    config = MaiDockConfig.model_validate({"plugin": {"locale": locale, "enabled": False}})
    plugin = MaiDockPlugin()
    plugin.set_plugin_config(config.model_dump(mode="python"))

    with pytest.raises(RuntimeError) as captured:
        await plugin.openai_responses_provider("response", {})

    assert marker in str(captured.value)


@pytest.mark.parametrize(
    ("locale", "cache_marker", "mimo_marker"),
    [
        ("zh-CN", "仅支持", "必须是"),
        ("zh-TW", "僅支援", "必須是"),
        ("en-US", "only supports", "must be"),
        ("ja-JP", "対応する値", "である必要"),
        ("ko-KR", "지원합니다", "이어야"),
    ],
)
def test_ark_cache_and_mimo_reasoning_errors_use_bound_locale(
    tmp_path: Path,
    locale: str,
    cache_marker: str,
    mimo_marker: str,
) -> None:
    with use_locale(normalize_locale(locale)):
        with pytest.raises(ValueError) as cache_error:
            PrefixCacheManager(PluginStateStore(tmp_path / "state.sqlite3"), ttl_seconds=1)
        with pytest.raises(TypeError) as mimo_error:
            MimoReasoningManager._assistant_messages_by_call_id({"messages": "invalid"})

    assert cache_marker in str(cache_error.value)
    assert mimo_marker in str(mimo_error.value)


def test_manifest_declares_all_locales() -> None:
    manifest = json.loads((PLUGIN_ROOT / "_manifest.json").read_text(encoding="utf-8"))
    config_template = tomllib.loads((PLUGIN_ROOT / "config.toml").read_text(encoding="utf-8"))

    assert manifest["i18n"] == {
        "default_locale": DEFAULT_LOCALE,
        "locales_path": "locales",
        "supported_locales": list(SUPPORTED_LOCALES),
    }
    assert config_template["plugin"]["locale"] == DEFAULT_LOCALE
