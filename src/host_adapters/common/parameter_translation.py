import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ...core.json_types import JsonValue, json_mapping_or_none, mapping_to_json_object, normalize_json_value
from ...core.parameter_catalog import CapabilityParameterCatalog, dotted_path
from ...core.parameter_policy import CapabilityKey, ParameterOverrideSet, ProviderPolicyKey
from ...i18n import runtime_expected, runtime_item, runtime_subject, translate
from ...schemas.host_snapshots import (
    AudioTranscriptionRequestSnapshot,
    BaseProviderRequestSnapshot,
    EmbeddingRequestSnapshot,
    ResponseFormatSnapshot,
    ResponseRequestSnapshot,
)

# normalized 字段包几乎全是 JSON 值，唯独 response_format 原样携带 Host 快照对象：
# 各 Provider 的转译器要靠 ResponseFormatSnapshot 上的 schema 别名做结构化输出，
# 提前拍平成 JSON 会丢掉别名信息，所以这里用联合类型如实表达而不是硬套 JsonValue。
type HostParameterValue = JsonValue | ResponseFormatSnapshot


@dataclass(slots=True)
class NormalizedHostParameters:
    """Provider 转译前的统一 Host 参数包。"""

    fields: dict[str, HostParameterValue] = field(default_factory=dict[str, HostParameterValue])
    sources: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(slots=True)
class TranslationEnvelope:
    """Provider 转译过程中构造的最终 transport envelope。"""

    body: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    headers: dict[str, str] = field(default_factory=dict[str, str])
    query: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass(slots=True)
class TranslationContext:
    """单次 Provider 参数转译上下文。"""

    request: BaseProviderRequestSnapshot | None
    provider_label: str
    provider: ProviderPolicyKey
    capability: CapabilityKey
    catalog: CapabilityParameterCatalog
    overrides: ParameterOverrideSet
    normalized: NormalizedHostParameters
    model: str


FieldTranslator = Callable[[TranslationContext, TranslationEnvelope, HostParameterValue], None]


def build_normalized_host_parameters(
    request: BaseProviderRequestSnapshot,
    *,
    overrides: ParameterOverrideSet,
    catalog: CapabilityParameterCatalog,
    provider_label: str,
    capability: CapabilityKey,
) -> NormalizedHostParameters:
    """把 Core 已解析的类型字段与 MaiDock 覆写值规整成内部字段包。

    优先级固定为：MaiDock 非空覆写值 > Core 请求级类型字段。覆写值后插入
    normalized.fields，转译时按序执行、后者覆盖前者，同一对象路径由此实现
    叶级合并（覆写对象只替换同名叶子）。
    """

    del provider_label, capability
    fields: dict[str, HostParameterValue] = {}
    sources: dict[str, str] = {}
    _merge_host_typed_fields(fields, request, catalog=catalog, sources=sources)
    for key, value in overrides.values.items():
        field = catalog.field_by_key(key)
        if field is None:
            continue
        fields[field.key] = value
        sources[field.key] = f"overrides.{field.key}"
    return NormalizedHostParameters(fields=fields, sources=sources)


def build_translation_context(
    request: BaseProviderRequestSnapshot,
    *,
    overrides: ParameterOverrideSet,
    catalog: CapabilityParameterCatalog,
    provider_label: str,
    provider: ProviderPolicyKey,
    capability: CapabilityKey,
    model: str,
) -> TranslationContext:
    """构建 normalized Host 参数与转译上下文。"""

    normalized = build_normalized_host_parameters(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=provider_label,
        capability=capability,
    )
    return TranslationContext(
        request=request,
        provider_label=provider_label,
        provider=provider,
        capability=capability,
        catalog=catalog,
        overrides=overrides,
        normalized=normalized,
        model=model,
    )


def run_translators(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    translators: Mapping[str, FieldTranslator],
) -> None:
    """按 normalized.fields 执行 Provider 字段转译。"""

    for key, value in context.normalized.fields.items():
        translator = translators.get(key)
        if translator is not None:
            translator(context, envelope, value)
            continue
        _write_catalog_target(context, envelope, key, value)


def _write_catalog_target(context: TranslationContext, envelope: TranslationEnvelope, key: str, value: object) -> None:
    """目录字段没有专用 translator 时按目标路径直写覆写值。"""

    field = context.catalog.field_by_key(key)
    if field is None:
        return
    set_target_value(envelope, field.target_path, value)


def set_target_value(envelope: TranslationEnvelope, path: tuple[str, ...], value: object) -> None:
    """在 body/headers/query 目标路径上写入值。"""

    root, tail = _split_target_path(path)
    if root == "headers":
        if len(tail) != 1:
            raise ValueError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"headers {runtime_subject('target_path')} {dotted_path(path)}",
                    expected=runtime_expected("one_field"),
                    actual=len(tail),
                )
            )
        if not isinstance(value, str):
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=dotted_path(path),
                    expected=runtime_expected("string"),
                    actual=type(value).__name__,
                )
            )
        envelope.headers[tail[0]] = value
        return
    target = envelope.body if root == "body" else envelope.query
    _set_nested_value(target, tail, normalize_json_value(value))


def get_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> JsonValue | None:
    root, tail = _split_target_path(path)
    if root == "headers":
        return envelope.headers.get(tail[0]) if len(tail) == 1 else None
    target: JsonValue = envelope.body if root == "body" else envelope.query
    return _get_nested_value(target, tail)


def has_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> bool:
    return get_target_value(envelope, path) is not None


def pop_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> JsonValue | None:
    root, tail = _split_target_path(path)
    if root == "headers":
        return envelope.headers.pop(tail[0], None) if len(tail) == 1 else None
    target = envelope.body if root == "body" else envelope.query
    return _pop_nested_value(target, tail)


def normalize_temperature(value: object, *, field_name: str = "temperature") -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(
        translate(
            "runtime.error.expected_type",
            subject=field_name,
            expected=runtime_expected("number"),
            actual=type(value).__name__,
        )
    )


def normalize_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(translate("runtime.error.positive_integer", subject=field_name))


def normalize_dimensions(value: object, *, field_name: str = "dimensions") -> int:
    return normalize_positive_int(value, field_name=field_name)


def normalize_json_object_value(value: object, *, field_name: str) -> dict[str, JsonValue]:
    mapping = json_mapping_or_none(value)
    if mapping is None:
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=field_name,
                expected=runtime_expected("object"),
                actual=type(value).__name__,
            )
        )
    return mapping_to_json_object(mapping)


def merge_body_object(envelope: TranslationEnvelope, key: str, value: object) -> dict[str, JsonValue]:
    existing = json_mapping_or_none(envelope.body.get(key))
    result = mapping_to_json_object(existing) if existing is not None else {}
    incoming = json_mapping_or_none(value)
    if incoming is not None:
        result.update(mapping_to_json_object(incoming))
    envelope.body[key] = result
    return result


def plugin_header_value(value: object) -> str:
    normalized = normalize_json_value(value)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _merge_host_typed_fields(
    fields: dict[str, HostParameterValue],
    request: BaseProviderRequestSnapshot,
    *,
    catalog: CapabilityParameterCatalog,
    sources: dict[str, str],
) -> None:
    """只合并 Core 已解析的请求级类型字段；请求字段为 None 时不注入。"""

    catalog_keys = {field.key for field in catalog.fields}
    if isinstance(request, ResponseRequestSnapshot):
        if "temperature" in catalog_keys and request.temperature is not None:
            fields["temperature"] = request.temperature
            sources["temperature"] = "request.temperature"
        if "max_tokens" in catalog_keys and request.max_tokens is not None:
            fields["max_tokens"] = request.max_tokens
            sources["max_tokens"] = "request.max_tokens"
        if "response_format" in catalog_keys and request.response_format is not None:
            fields["response_format"] = request.response_format
            sources["response_format"] = "request.response_format"
    elif isinstance(request, EmbeddingRequestSnapshot):
        if "dimensions" in catalog_keys and request.dimensions is not None:
            fields["dimensions"] = request.dimensions
            sources["dimensions"] = "request.dimensions"
    elif isinstance(request, AudioTranscriptionRequestSnapshot):
        if "max_tokens" in catalog_keys and request.max_tokens is not None:
            fields["max_tokens"] = request.max_tokens
            sources["max_tokens"] = "request.max_tokens"


def _split_target_path(path: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if len(path) < 2 or path[0] not in {"body", "headers", "query"}:
        raise ValueError(
            translate(
                "runtime.error.target_root_invalid",
                path=dotted_path(path),
            )
        )
    return path[0], path[1:]


def _set_nested_value(target: dict[str, JsonValue], parts: tuple[str, ...], value: object) -> None:
    if not parts:
        raise ValueError(
            translate(
                "runtime.error.required",
                subject=runtime_subject("target_path"),
                field=runtime_item("path_segment"),
            )
        )
    current = target
    for part in parts[:-1]:
        child = json_mapping_or_none(current.get(part))
        child_object = mapping_to_json_object(child) if child is not None else {}
        current[part] = child_object
        current = child_object
    current[parts[-1]] = normalize_json_value(value)


def _get_nested_value(target: JsonValue, parts: tuple[str, ...]) -> JsonValue | None:
    current = target
    for part in parts:
        mapping = json_mapping_or_none(current)
        if mapping is None or part not in mapping:
            return None
        current = mapping[part]
    return current


def _pop_nested_value(target: dict[str, JsonValue], parts: tuple[str, ...]) -> JsonValue | None:
    if not parts:
        return None
    current = target
    for part in parts[:-1]:
        child = json_mapping_or_none(current.get(part))
        if child is None:
            return None
        child_object = mapping_to_json_object(child)
        current[part] = child_object
        current = child_object
    return current.pop(parts[-1], None)
