import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ...core.json_types import (
    mapping_to_json_object,
    normalize_json_value,
    json_mapping_or_none,
)
from ...core.parameter_catalog import CapabilityParameterCatalog, dotted_path
from ...core.parameter_policy import (
    CapabilityKey,
    ParameterPolicy,
    ProviderPolicyKey,
    UnknownExtraParamsPolicy,
)
from ...schemas.host_snapshots import (
    AudioTranscriptionRequestSnapshot,
    BaseProviderRequestSnapshot,
    EmbeddingRequestSnapshot,
    ResponseRequestSnapshot,
)


@dataclass(slots=True)
class NormalizedHostParameters:
    """Provider 转译前的统一 Host 参数包。"""

    fields: dict = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(slots=True)
class TranslationEnvelope:
    """Provider 转译过程中构造的最终 transport envelope。"""

    body: dict = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict[str, str])
    query: dict = field(default_factory=dict)


@dataclass(slots=True)
class TranslationContext:
    """单次 Provider 参数转译上下文。"""

    request: BaseProviderRequestSnapshot | None
    provider_label: str
    provider: ProviderPolicyKey
    capability: CapabilityKey
    catalog: CapabilityParameterCatalog
    policy: ParameterPolicy
    normalized: NormalizedHostParameters
    model: str


FieldTranslator = Callable[[TranslationContext, TranslationEnvelope, object], None]
_TRANSPORT_ROOTS = {"body", "headers", "query"}


def build_normalized_host_parameters(
    request: BaseProviderRequestSnapshot,
    *,
    policy: ParameterPolicy,
    catalog: CapabilityParameterCatalog,
    provider_label: str,
    capability: CapabilityKey,
) -> NormalizedHostParameters:
    """
    把 typed Host 字段、model extra 与 request extra 规整成内部字段包。

    优先级从低到高固定为：policy defaults、model extra、model typed fallback、
    request extra、request typed fields。最终 target-path override/disable/reject
    仍由 transport policy 统一处理。
    """

    aliases = catalog.source_alias_map()
    catalog_keys = set(aliases.values())
    reserved_keys = set(catalog.reserved_body_keys)
    disabled_paths = _normalized_paths(policy.disabled_paths)
    rejected_paths = _normalized_paths(policy.rejected_paths)
    fields: dict = {}
    sources: dict[str, str] = {}

    _merge_source(
        fields,
        policy.default_params,
        aliases=aliases,
        catalog_keys=catalog_keys,
        reserved_keys=reserved_keys,
        disabled_paths=disabled_paths,
        sources=sources,
        source_label="default_params",
    )
    if policy.accept_model_extra_params:
        _merge_source(
            fields,
            _accepted_extra_params(request.model_info.extra_params.fields),
            aliases=aliases,
            catalog_keys=catalog_keys,
            reserved_keys=reserved_keys,
            disabled_paths=disabled_paths,
            sources=sources,
            source_label="model_info.extra_params",
        )
    _merge_typed_model_fallbacks(
        fields,
        request,
        catalog_keys=catalog_keys,
        disabled_paths=disabled_paths,
        sources=sources,
    )
    if policy.accept_request_extra_params:
        _merge_source(
            fields,
            _accepted_extra_params(request.extra_params.fields),
            aliases=aliases,
            catalog_keys=catalog_keys,
            reserved_keys=reserved_keys,
            disabled_paths=disabled_paths,
            sources=sources,
            source_label="request.extra_params",
        )
    _merge_typed_request_fields(
        fields,
        request,
        catalog_keys=catalog_keys,
        disabled_paths=disabled_paths,
        sources=sources,
    )

    _raise_for_rejected_normalized_fields(
        fields,
        catalog=catalog,
        rejected_paths=rejected_paths,
        provider_label=provider_label,
        capability=capability,
    )
    _drop_disabled_normalized_fields(fields, sources=sources, catalog=catalog, disabled_paths=disabled_paths)
    return NormalizedHostParameters(fields=fields, sources=sources)


def build_translation_context(
    request: BaseProviderRequestSnapshot,
    *,
    policy: ParameterPolicy,
    catalog: CapabilityParameterCatalog,
    provider_label: str,
    provider: ProviderPolicyKey,
    capability: CapabilityKey,
    model: str,
) -> TranslationContext:
    """构建 normalized Host 参数与转译上下文。"""

    normalized = build_normalized_host_parameters(
        request,
        policy=policy,
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
        policy=policy,
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
        if key in _TRANSPORT_ROOTS:
            _merge_transport_root(envelope, key, value)
            continue
        translator = translators.get(key)
        if translator is not None:
            translator(context, envelope, value)
            continue
        _handle_unknown_field(context, envelope, key, value)


def set_target_value(envelope: TranslationEnvelope, path: tuple[str, ...], value: object) -> None:
    """在 body/headers/query 目标路径上写入值。"""

    root, tail = _split_target_path(path)
    if root == "headers":
        if len(tail) != 1:
            raise ValueError(f"headers 目标路径必须只有一个字段: {dotted_path(path)}")
        if not isinstance(value, str):
            raise TypeError(f"{dotted_path(path)} 必须是字符串，实际为 {type(value).__name__}")
        envelope.headers[tail[0]] = value
        return
    target = envelope.body if root == "body" else envelope.query
    _set_nested_value(target, tail, normalize_json_value(value))


def get_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> object | None:
    root, tail = _split_target_path(path)
    if root == "headers":
        return envelope.headers.get(tail[0]) if len(tail) == 1 else None
    target: object = envelope.body if root == "body" else envelope.query
    return _get_nested_value(target, tail)


def has_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> bool:
    return get_target_value(envelope, path) is not None


def pop_target_value(envelope: TranslationEnvelope, path: tuple[str, ...]) -> object | None:
    root, tail = _split_target_path(path)
    if root == "headers":
        return envelope.headers.pop(tail[0], None) if len(tail) == 1 else None
    target = envelope.body if root == "body" else envelope.query
    return _pop_nested_value(target, tail)


def normalize_temperature(value: object, *, field_name: str = "temperature") -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"{field_name} 必须是数字，实际为 {type(value).__name__}")


def normalize_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError(f"{field_name} 必须是正整数")


def normalize_dimensions(value: object, *, field_name: str = "dimensions") -> int:
    return normalize_positive_int(value, field_name=field_name)


def normalize_json_object_value(value: object, *, field_name: str) -> dict:
    mapping = json_mapping_or_none(value)
    if mapping is None:
        raise TypeError(f"{field_name} 必须是 object，实际为 {type(value).__name__}")
    return mapping_to_json_object(mapping)


def merge_body_object(envelope: TranslationEnvelope, key: str, value: object) -> dict:
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


def _merge_typed_model_fallbacks(
    fields: dict,
    request: BaseProviderRequestSnapshot,
    *,
    catalog_keys: set[str],
    disabled_paths: set[str],
    sources: dict[str, str],
) -> None:
    if isinstance(request, ResponseRequestSnapshot):
        if "temperature" in catalog_keys and request.model_info.temperature is not None:
            _set_known_field(
                fields,
                "temperature",
                request.model_info.temperature,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="model_info.temperature",
            )
        if "max_tokens" in catalog_keys and request.model_info.max_tokens is not None:
            _set_known_field(
                fields,
                "max_tokens",
                request.model_info.max_tokens,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="model_info.max_tokens",
            )
    elif isinstance(request, AudioTranscriptionRequestSnapshot):
        if "max_tokens" in catalog_keys and request.model_info.max_tokens is not None:
            _set_known_field(
                fields,
                "max_tokens",
                request.model_info.max_tokens,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="model_info.max_tokens",
            )


def _merge_typed_request_fields(
    fields: dict,
    request: BaseProviderRequestSnapshot,
    *,
    catalog_keys: set[str],
    disabled_paths: set[str],
    sources: dict[str, str],
) -> None:
    if isinstance(request, ResponseRequestSnapshot):
        if "temperature" in catalog_keys and request.temperature is not None:
            _set_known_field(
                fields,
                "temperature",
                request.temperature,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="request.temperature",
            )
        if "max_tokens" in catalog_keys and request.max_tokens is not None:
            _set_known_field(
                fields,
                "max_tokens",
                request.max_tokens,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="request.max_tokens",
            )
        if "response_format" in catalog_keys and request.response_format is not None:
            _set_known_field(
                fields,
                "response_format",
                request.response_format,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="request.response_format",
                reject_conflict=True,
            )
    elif isinstance(request, EmbeddingRequestSnapshot):
        dimensions = request.dimensions
        if "dimensions" in catalog_keys and dimensions is not None:
            _set_known_field(
                fields,
                "dimensions",
                dimensions,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="request.dimensions",
                reject_conflict=True,
            )
    elif isinstance(request, AudioTranscriptionRequestSnapshot):
        if "max_tokens" in catalog_keys and request.max_tokens is not None:
            _set_known_field(
                fields,
                "max_tokens",
                request.max_tokens,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label="request.max_tokens",
            )


def _merge_source(
    fields: dict,
    source: Mapping[str, object],
    *,
    aliases: Mapping[str, str],
    catalog_keys: set[str],
    reserved_keys: set[str],
    disabled_paths: set[str],
    sources: dict[str, str],
    source_label: str,
) -> None:
    source_values: dict[str, object] = {}
    source_keys: dict[str, str] = {}
    for key, value in source.items():
        if value is None:
            continue
        normalized_key = str(key)
        if normalized_key in _TRANSPORT_ROOTS:
            if _is_disabled_source(normalized_key, disabled_paths):
                continue
            _merge_control_root(
                fields,
                normalized_key,
                value,
                sources=sources,
                source_label=f"{source_label}.{normalized_key}",
            )
            continue
        if normalized_key in reserved_keys:
            continue
        canonical_key = aliases.get(normalized_key, normalized_key)
        if canonical_key not in catalog_keys and normalized_key not in _TRANSPORT_ROOTS:
            _set_known_field(
                fields,
                normalized_key,
                value,
                disabled_paths=disabled_paths,
                sources=sources,
                source_label=f"{source_label}.{normalized_key}",
            )
            continue
        if canonical_key in source_values and source_values[canonical_key] != value:
            previous_key = source_keys[canonical_key]
            raise ValueError(
                f"{source_label} 同时提供 {previous_key} 与 {normalized_key}，它们都映射到 {canonical_key} 且值不一致"
            )
        source_values[canonical_key] = value
        source_keys[canonical_key] = normalized_key
        _set_known_field(
            fields,
            canonical_key,
            value,
            disabled_paths=disabled_paths,
            sources=sources,
            source_label=f"{source_label}.{normalized_key}",
        )


def _set_known_field(
    fields: dict,
    key: str,
    value: object,
    *,
    disabled_paths: set[str],
    sources: dict[str, str],
    source_label: str,
    overwrite: bool = True,
    reject_conflict: bool = False,
) -> None:
    if _is_disabled_source(key, disabled_paths):
        return
    if reject_conflict and key in fields and fields[key] != value:
        previous_source = sources.get(key, key)
        raise ValueError(f"Host typed 字段 {key} 与 {previous_source} 同时存在且值不一致")
    if overwrite or key not in fields:
        fields[key] = value
        sources[key] = source_label


def _merge_control_root(
    fields: dict,
    key: str,
    value: object,
    *,
    sources: dict[str, str],
    source_label: str,
) -> None:
    incoming = json_mapping_or_none(value)
    if incoming is None:
        raise TypeError(f"extra_params.{key} 必须是 object，实际为 {type(value).__name__}")
    current = json_mapping_or_none(fields.get(key))
    merged = mapping_to_json_object(current) if current is not None else {}
    _deep_merge(merged, mapping_to_json_object(incoming))
    fields[key] = merged
    sources[key] = source_label


def _merge_transport_root(envelope: TranslationEnvelope, key: str, value: object) -> None:
    incoming = json_mapping_or_none(value)
    if incoming is None:
        raise TypeError(f"extra_params.{key} 必须是 object，实际为 {type(value).__name__}")
    incoming_object = mapping_to_json_object(incoming)
    if key == "body":
        _deep_merge(envelope.body, incoming_object)
    elif key == "query":
        _deep_merge(envelope.query, incoming_object)
    else:
        for header, header_value in incoming_object.items():
            if not isinstance(header_value, str):
                raise TypeError(f"extra_params.headers.{header} 必须是字符串，实际为 {type(header_value).__name__}")
            envelope.headers[header] = header_value


def _handle_unknown_field(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    key: str,
    value: object,
) -> None:
    unknown_policy: UnknownExtraParamsPolicy = context.policy.unknown_extra_params
    if unknown_policy == "drop":
        return
    if unknown_policy == "reject":
        raise ValueError(f"{context.provider_label} {context.capability} 不支持 extra_params 字段: {key}")
    set_target_value(envelope, ("body", key), value)


def _accepted_extra_params(source: Mapping[str, object]) -> dict:
    result: dict = {}
    for key, value in source.items():
        if value is not None:
            result[str(key)] = value
    return result


def _drop_disabled_normalized_fields(
    fields: dict,
    *,
    sources: dict[str, str],
    catalog: CapabilityParameterCatalog,
    disabled_paths: set[str],
) -> None:
    for field_def in catalog.fields:
        if dotted_path(field_def.target_path) in disabled_paths:
            fields.pop(field_def.key, None)
            sources.pop(field_def.key, None)
    for path in disabled_paths:
        parts = tuple(part for part in path.split(".") if part)
        if len(parts) == 1:
            fields.pop(parts[0], None)
            sources.pop(parts[0], None)


def _raise_for_rejected_normalized_fields(
    fields: Mapping[str, object],
    *,
    catalog: CapabilityParameterCatalog,
    rejected_paths: set[str],
    provider_label: str,
    capability: CapabilityKey,
) -> None:
    for field_def in catalog.fields:
        if field_def.key in fields and dotted_path(field_def.target_path) in rejected_paths:
            raise ValueError(f"{provider_label} {capability} 参数策略拒绝路径: {dotted_path(field_def.target_path)}")
    for path in rejected_paths:
        parts = tuple(part for part in path.split(".") if part)
        if len(parts) == 1 and parts[0] in fields:
            raise ValueError(f"{provider_label} {capability} 参数策略拒绝路径: {path}")


def _is_disabled_source(key: str, disabled_paths: set[str]) -> bool:
    return key in disabled_paths


def _normalized_paths(paths: tuple[str, ...]) -> set[str]:
    return {path.strip() for path in paths if path.strip()}


def _split_target_path(path: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if len(path) < 2 or path[0] not in _TRANSPORT_ROOTS:
        raise ValueError(f"目标路径必须以 body/headers/query 开头: {dotted_path(path)}")
    return path[0], path[1:]


def _set_nested_value(target: dict, parts: tuple[str, ...], value: object) -> None:
    if not parts:
        raise ValueError("目标路径不能为空")
    current = target
    for part in parts[:-1]:
        child = json_mapping_or_none(current.get(part))
        child_object = mapping_to_json_object(child) if child is not None else {}
        current[part] = child_object
        current = child_object
    current[parts[-1]] = normalize_json_value(value)


def _get_nested_value(target: object, parts: tuple[str, ...]) -> object | None:
    current = target
    for part in parts:
        mapping = json_mapping_or_none(current)
        if mapping is None or part not in mapping:
            return None
        current = mapping[part]
    return current


def _pop_nested_value(target: dict, parts: tuple[str, ...]) -> object | None:
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


def _deep_merge(target: dict, source: Mapping[str, object]) -> None:
    for key, value in source.items():
        normalized_key = str(key)
        source_mapping = json_mapping_or_none(value)
        target_mapping = json_mapping_or_none(target.get(normalized_key))
        if source_mapping is not None and target_mapping is not None:
            merged = mapping_to_json_object(target_mapping)
            _deep_merge(merged, mapping_to_json_object(source_mapping))
            target[normalized_key] = merged
            continue
        target[normalized_key] = normalize_json_value(value)
