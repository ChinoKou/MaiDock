import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Final, Literal, cast

from pydantic import ValidationError

type Locale = Literal["zh-CN", "zh-TW", "en-US", "ja-JP", "ko-KR"]
type RuntimeSubject = Literal[
    "ark_prefix_cache_entry",
    "ark_prefix_cache_response",
    "ark_responses_input",
    "ark_responses_request",
    "ark_settings",
    "ark_tokenization_data_item",
    "anthropic_image_data",
    "audio_file",
    "audio_format",
    "audio_transcription_request",
    "base64_data",
    "detected_file_signature",
    "explicit_audio_format",
    "historical_tool_call",
    "image_data",
    "image_dimension",
    "image_dimensions",
    "image_frames",
    "image_pixels",
    "json_response",
    "json_value",
    "llm_provider_request",
    "mimo_assistant_tool_call_item",
    "mimo_historical_tool_call",
    "mimo_historical_tool_calls",
    "mimo_native_reasoning",
    "mimo_outbound_tool_call",
    "mimo_reasoning_tool_call_response",
    "mimo_request_messages",
    "mimo_settings",
    "mimo_tool_call",
    "mimo_tool_call_extra_content",
    "mimo_tool_call_metadata",
    "mimo_thinking",
    "parameter_path",
    "parameter_paths",
    "parsed_tool_call_arguments",
    "response",
    "sanitized_value",
    "sse_data",
    "sse_json_data",
    "target_path",
    "tool_call_arguments",
    "unrecognized_audio",
    "value",
]
type RuntimeExpected = Literal[
    "array",
    "boolean",
    "boolean_or_string",
    "boolean_override_value",
    "finite_number",
    "float_compatible_value",
    "integer_override_value",
    "json_string_array_override_value",
    "list",
    "list_of_objects",
    "mapping",
    "non_empty_string",
    "non_negative_integer",
    "number",
    "numeric_override_value",
    "object",
    "one_field",
    "positive_dimensions",
    "string",
    "valid_base64_data",
    "valid_image",
    "valid_json",
    "valid_json_override_value",
]
type RuntimeActual = Literal["invalid_base64_data", "invalid_image", "invalid_json", "non_ascii_data"]
type RuntimeItem = Literal[
    "audio_transcription_text",
    "decoded_audio_data",
    "embedding_array",
    "matching_outbound_assistant_message",
    "name_and_schema",
    "format_or_audio_format",
    "non_empty_name",
    "non_empty_path",
    "non_empty_value",
    "output_stream_text_or_tools",
    "output_text_or_tools",
    "output_text_reasoning_or_tools",
    "override_value",
    "path_segment",
    "duplicate_call_id",
    "reasoning_manager",
    "response_reasoning_content",
    "restorable_reasoning_content",
    "stored_reasoning_content",
    "text_content",
    "valid_id",
]

DEFAULT_LOCALE: Final[Locale] = "zh-CN"
SUPPORTED_LOCALES: Final[tuple[Locale, ...]] = (
    "zh-CN",
    "zh-TW",
    "en-US",
    "ja-JP",
    "ko-KR",
)

_LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"
_CURRENT_LOCALE: ContextVar[Locale] = ContextVar("maidock_locale", default=DEFAULT_LOCALE)
_FORMATTER: Final = Formatter()
_VALIDATION_DESCRIPTION_BY_CODE: Final[dict[str, str]] = {
    "missing": "missing",
    "int_type": "integer",
    "int_parsing": "integer",
    "int_from_float": "integer",
    "float_type": "number",
    "float_parsing": "number",
    "finite_number": "finite_number",
    "bool_type": "boolean",
    "bool_parsing": "boolean",
    "string_type": "string",
    "string_too_short": "string_length",
    "string_too_long": "string_length",
    "list_type": "list",
    "tuple_type": "list",
    "dict_type": "object",
    "mapping_type": "object",
    "model_type": "object",
    "literal_error": "allowed_value",
    "enum": "allowed_value",
    "greater_than": "number_range",
    "greater_than_equal": "number_range",
    "less_than": "number_range",
    "less_than_equal": "number_range",
    "multiple_of": "number_range",
    "extra_forbidden": "extra_forbidden",
    "json_invalid": "json",
    "json_type": "json",
}


class TranslationCatalogError(RuntimeError):
    """语言目录不完整或格式不合法。"""


def normalize_locale(value: object) -> Locale:
    """严格校验 MaiDock 支持的语言代码。"""

    if isinstance(value, str) and value in SUPPORTED_LOCALES:
        return cast(Locale, value)
    supported = ", ".join(SUPPORTED_LOCALES)
    raise ValueError(translate("runtime.locale.unsupported", value=repr(value), supported=supported))


def get_locale() -> Locale:
    """返回当前请求绑定的语言。"""

    return _CURRENT_LOCALE.get()


@contextmanager
def use_locale(locale: Locale) -> Iterator[None]:
    """在当前同步或异步上下文中绑定语言。"""

    token = _CURRENT_LOCALE.set(locale)
    try:
        yield
    finally:
        _CURRENT_LOCALE.reset(token)


def translate(key: str, /, *, locale: Locale | None = None, **values: object) -> str:
    """读取并格式化指定消息；缺失键或参数时直接报错。"""

    selected_locale = locale or get_locale()
    catalogs = _load_catalogs()
    try:
        template = catalogs[selected_locale][key]
    except KeyError as exc:
        raise TranslationCatalogError(f"语言 {selected_locale} 缺少消息键 {key}") from exc
    try:
        return template.format_map(values)
    except (KeyError, ValueError) as exc:
        raise TranslationCatalogError(f"消息键 {key} 格式化失败: {exc}") from exc


def runtime_subject(key: RuntimeSubject) -> str:
    """返回本地化的运行时错误主体。"""

    return translate(f"runtime.subject.{key}")


def runtime_expected(key: RuntimeExpected) -> str:
    """返回本地化的运行时期望类型或约束。"""

    return translate(f"runtime.expected.{key}")


def runtime_actual(key: RuntimeActual) -> str:
    """返回本地化的运行时实际值说明。"""

    return translate(f"runtime.actual.{key}")


def runtime_item(key: RuntimeItem) -> str:
    """返回本地化的运行时字段或输出项说明。"""

    return translate(f"runtime.item.{key}")


def validate_catalogs() -> None:
    """预加载并严格校验所有语言目录。"""

    _load_catalogs()


def format_validation_error(error: ValidationError) -> str:
    """把 Pydantic 校验错误整理为稳定、可本地化的外显文本。"""

    issues: list[str] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        raw_location = issue.get("loc", ())
        if isinstance(raw_location, (list, tuple)):
            location = ".".join(str(part) for part in raw_location) or "$"
        else:
            location = str(raw_location) or "$"
        error_code = str(issue.get("type", "validation_error"))
        description_key = _VALIDATION_DESCRIPTION_BY_CODE.get(error_code, "invalid_value")
        description = translate(f"runtime.validation.description.{description_key}")
        issues.append(
            translate(
                "runtime.validation.issue",
                path=location,
                code=error_code,
                description=description,
            )
        )
    return translate("runtime.validation.failed", issues="; ".join(issues) or type(error).__name__)


@lru_cache(maxsize=1)
def _load_catalogs() -> dict[Locale, dict[str, str]]:
    return _load_catalogs_from_directory(_LOCALES_DIR)


def _load_catalogs_from_directory(locales_dir: Path) -> dict[Locale, dict[str, str]]:
    """从指定目录加载语言资源，供启动校验和隔离测试复用。"""

    catalogs: dict[Locale, dict[str, str]] = {}
    for locale in SUPPORTED_LOCALES:
        path = locales_dir / f"{locale}.json"
        try:
            raw_catalog = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_duplicate_rejecting_object_hook(path),
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TranslationCatalogError(f"无法读取语言目录 {path}: {exc}") from exc
        if not isinstance(raw_catalog, dict):
            raise TranslationCatalogError(f"语言目录 {path} 的顶层必须是 object")

        catalog: dict[str, str] = {}
        for raw_key, raw_text in raw_catalog.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise TranslationCatalogError(f"语言目录 {path} 包含空消息键")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise TranslationCatalogError(f"语言目录 {path} 的消息 {raw_key} 必须是非空字符串")
            catalog[raw_key] = raw_text
        catalogs[locale] = catalog

    reference = catalogs[DEFAULT_LOCALE]
    reference_keys = set(reference)
    for locale, catalog in catalogs.items():
        catalog_keys = set(catalog)
        if catalog_keys != reference_keys:
            missing = sorted(reference_keys - catalog_keys)
            extra = sorted(catalog_keys - reference_keys)
            raise TranslationCatalogError(f"语言 {locale} 的消息键不一致: missing={missing}, extra={extra}")
        for key, reference_text in reference.items():
            expected_fields = _format_fields(reference_text)
            actual_fields = _format_fields(catalog[key])
            if actual_fields != expected_fields:
                raise TranslationCatalogError(
                    f"语言 {locale} 的消息 {key} 占位符不一致: expected={sorted(expected_fields)}, "
                    f"actual={sorted(actual_fields)}"
                )
    return catalogs


def _duplicate_rejecting_object_hook(path: Path):
    def build_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TranslationCatalogError(f"语言目录 {path} 包含重复消息键 {key}")
            result[key] = value
        return result

    return build_object


def _format_fields(template: str) -> frozenset[str]:
    try:
        return frozenset(field_name for _, field_name, _, _ in _FORMATTER.parse(template) if field_name)
    except ValueError as exc:
        raise TranslationCatalogError(f"消息模板格式不合法: {template!r}") from exc
