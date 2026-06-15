from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ..schemas.base import ObjectFields
from ..schemas.host_snapshots import BaseProviderRequestSnapshot
from .json_types import JsonValue, is_json_list, json_mapping_or_none, mapping_to_json_object

type ProviderPolicyKey = Literal[
    "openai_responses",
    "anthropic_messages",
    "volcengine_ark",
    "dashscope",
    "siliconflow",
    "xiaomi_mimo",
]
type CapabilityKey = Literal[
    "response",
    "chat_completion",
    "embeddings",
    "audio_transcription",
    "image_generation",
]
type UnknownExtraParamsPolicy = Literal["forward", "drop", "reject"]

_TRANSPORT_ROOTS = {"body", "headers", "query"}
_CONTROL_EXTRA_PARAM_KEYS = _TRANSPORT_ROOTS


@dataclass(slots=True)
class ParameterPolicy:
    """Provider 能力的额外参数策略。"""

    accept_model_extra_params: bool = True
    accept_request_extra_params: bool = True
    disabled_paths: tuple[str, ...] = ()
    rejected_paths: tuple[str, ...] = ()
    default_params: dict = field(default_factory=dict)
    override_params: dict = field(default_factory=dict)
    unknown_extra_params: UnknownExtraParamsPolicy = "forward"


@dataclass(slots=True)
class ProviderCapabilityPolicies:
    """Provider 所暴露所有能力的策略集合。"""

    response: ParameterPolicy = field(default_factory=ParameterPolicy)
    chat_completion: ParameterPolicy = field(default_factory=ParameterPolicy)
    embeddings: ParameterPolicy = field(default_factory=ParameterPolicy)
    audio_transcription: ParameterPolicy = field(default_factory=ParameterPolicy)
    image_generation: ParameterPolicy = field(default_factory=ParameterPolicy)

    def get(self, capability: CapabilityKey) -> ParameterPolicy:
        match capability:
            case "response":
                return self.response
            case "chat_completion":
                return self.chat_completion
            case "embeddings":
                return self.embeddings
            case "audio_transcription":
                return self.audio_transcription
            case "image_generation":
                return self.image_generation
        raise ValueError(f"不支持的能力策略: {capability}")


@dataclass(slots=True)
class ParameterPolicyRegistry:
    """Provider/能力维度的参数策略注册表。"""

    openai_responses: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)
    anthropic_messages: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)
    volcengine_ark: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)
    dashscope: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)
    siliconflow: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)
    xiaomi_mimo: ProviderCapabilityPolicies = field(default_factory=ProviderCapabilityPolicies)

    def get(self, provider: ProviderPolicyKey, capability: CapabilityKey) -> ParameterPolicy:
        match provider:
            case "openai_responses":
                return self.openai_responses.get(capability)
            case "anthropic_messages":
                return self.anthropic_messages.get(capability)
            case "volcengine_ark":
                return self.volcengine_ark.get(capability)
            case "dashscope":
                return self.dashscope.get(capability)
            case "siliconflow":
                return self.siliconflow.get(capability)
            case "xiaomi_mimo":
                return self.xiaomi_mimo.get(capability)
        raise ValueError(f"不支持的 Provider 参数策略: {provider}")


@dataclass(slots=True)
class ResolvedParameterExtras:
    """经策略解析后的原始 extra_params，用于 Provider 拆分。"""

    extra_params: dict = field(default_factory=dict)


@dataclass(slots=True)
class TransportParameters:
    """策略应用后的最终 HTTP 传输参数。"""

    body: dict = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=lambda: {})
    query: dict = field(default_factory=dict)


def normalize_policy_params(value: object) -> dict:
    """将配置值规范化为普通 JSON object。"""

    return ObjectFields.from_unknown(value).to_plain_dict()


def normalize_policy_paths(paths: list[str]) -> tuple[str, ...]:
    """规范化点号分隔的策略路径并丢弃空条目。"""

    normalized: list[str] = []
    for path in paths:
        current = path.strip()
        if current:
            normalized.append(current)
    return tuple(normalized)


def resolve_request_parameter_policy(
    request: BaseProviderRequestSnapshot,
    *,
    policy: ParameterPolicy,
    provider_label: str,
    capability: CapabilityKey,
    direct_body_keys: set[str] | None = None,
    reserved_body_keys: set[str] | None = None,
) -> ResolvedParameterExtras:
    """根据 Provider 能力策略解析模型/请求的 extra_params。"""

    model_params = _accepted_extra_params(request.model_info.extra_params.fields)
    request_params = _request_only_extra_params(request.extra_params.fields, model_params)
    host_params: dict = {}
    if policy.accept_model_extra_params:
        _merge_shallow(host_params, model_params)
    if policy.accept_request_extra_params:
        _merge_shallow(host_params, request_params)

    _raise_for_rejected_paths(host_params, policy.rejected_paths, provider_label=provider_label, capability=capability)
    _remove_paths(host_params, policy.disabled_paths)
    _apply_unknown_policy(
        host_params,
        unknown_policy=policy.unknown_extra_params,
        provider_label=provider_label,
        capability=capability,
        direct_body_keys=direct_body_keys,
        reserved_body_keys=reserved_body_keys,
    )

    effective = _deep_json_object(policy.default_params)
    _merge_shallow(effective, host_params)
    _remove_paths(effective, policy.disabled_paths)
    _deep_merge(effective, policy.override_params)
    return ResolvedParameterExtras(extra_params=effective)


def apply_transport_parameter_policy(
    *,
    body: Mapping[str, JsonValue],
    headers: Mapping[str, str],
    query: Mapping[str, JsonValue],
    policy: ParameterPolicy,
    provider_label: str,
    capability: CapabilityKey,
) -> TransportParameters:
    """将显式声明的 body/header/query 策略路径应用到最终传输参数。"""

    envelope: dict = {
        "body": mapping_to_json_object(body),
        "headers": dict(headers),
        "query": mapping_to_json_object(query),
    }
    transport_rejected_paths = _transport_paths(policy.rejected_paths)
    _raise_for_rejected_paths(envelope, transport_rejected_paths, provider_label=provider_label, capability=capability)
    _remove_paths(envelope, _transport_paths(policy.disabled_paths))
    _apply_transport_overrides(envelope, policy.override_params)

    body_value = json_mapping_or_none(envelope.get("body")) or {}
    headers_value = json_mapping_or_none(envelope.get("headers")) or {}
    query_value = json_mapping_or_none(envelope.get("query")) or {}
    return TransportParameters(
        body=mapping_to_json_object(body_value),
        headers=_string_dict(headers_value, field_name="parameter_policy.override_params.headers"),
        query=mapping_to_json_object(query_value),
    )


def _accepted_extra_params(source: Mapping[str, JsonValue]) -> dict:
    result: dict = {}
    for key, value in source.items():
        if value is not None:
            result[str(key)] = value
    return result


def _request_only_extra_params(request_params: Mapping[str, JsonValue], model_params: Mapping[str, JsonValue]) -> dict:
    result: dict = {}
    for key, value in request_params.items():
        if value is None:
            continue
        normalized_key = str(key)
        if normalized_key in model_params and model_params[normalized_key] == value:
            continue
        result[normalized_key] = value
    return result


def _apply_unknown_policy(
    params: dict,
    *,
    unknown_policy: Literal["forward", "drop", "reject"],
    provider_label: str,
    capability: CapabilityKey,
    direct_body_keys: set[str] | None,
    reserved_body_keys: set[str] | None,
) -> None:
    if unknown_policy == "forward":
        return
    unknown_keys = _unknown_top_level_keys(
        params,
        direct_body_keys=direct_body_keys,
        reserved_body_keys=reserved_body_keys,
    )
    if not unknown_keys:
        return
    if unknown_policy == "drop":
        for key in unknown_keys:
            params.pop(key, None)
        return
    joined_keys = ", ".join(unknown_keys)
    raise ValueError(f"{provider_label} {capability} 不支持这些 extra_params 字段: {joined_keys}")


def _unknown_top_level_keys(
    params: Mapping[str, JsonValue],
    *,
    direct_body_keys: set[str] | None,
    reserved_body_keys: set[str] | None,
) -> list[str]:
    direct_keys = direct_body_keys or set()
    reserved_keys = reserved_body_keys or set()
    unknown_keys = [
        key
        for key in params
        if key not in _CONTROL_EXTRA_PARAM_KEYS and key not in direct_keys and key not in reserved_keys
    ]
    return sorted(unknown_keys)


def _parse_path(path: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in path.split(".") if part.strip())


def _transport_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if _path_root(path) in _TRANSPORT_ROOTS)


def _path_root(path: str) -> str | None:
    parts = _parse_path(path)
    return parts[0] if parts else None


def _raise_for_rejected_paths(
    payload: Mapping[str, JsonValue],
    paths: tuple[str, ...],
    *,
    provider_label: str,
    capability: CapabilityKey,
) -> None:
    for path in paths:
        parts = _parse_path(path)
        if parts and _has_path(payload, parts):
            raise ValueError(f"{provider_label} {capability} 参数策略拒绝路径: {path}")


def _has_path(payload: Mapping[str, JsonValue], parts: tuple[str, ...]) -> bool:
    current: object = payload
    for index, part in enumerate(parts):
        mapping = json_mapping_or_none(current)
        if mapping is None or part not in mapping:
            return False
        if index == len(parts) - 1:
            return True
        current = mapping[part]
    return False


def _remove_paths(payload: dict, paths: tuple[str, ...]) -> None:
    for path in paths:
        parts = _parse_path(path)
        if parts:
            _remove_path(payload, parts)


def _remove_path(payload: dict, parts: tuple[str, ...]) -> None:
    current: dict = payload
    for part in parts[:-1]:
        child = json_mapping_or_none(current.get(part))
        if child is None:
            return
        child_object = mapping_to_json_object(child)
        current[part] = child_object
        current = child_object
    current.pop(parts[-1], None)


def _apply_transport_overrides(envelope: dict, override_params: Mapping[str, JsonValue]) -> None:
    for root in _TRANSPORT_ROOTS:
        root_override = json_mapping_or_none(override_params.get(root))
        if root_override is None:
            continue
        target = json_mapping_or_none(envelope.get(root)) or {}
        target_object = mapping_to_json_object(target)
        _deep_merge(target_object, mapping_to_json_object(root_override))
        envelope[root] = target_object


def _merge_shallow(target: dict, source: Mapping[str, JsonValue]) -> None:
    for key, value in source.items():
        target[str(key)] = value


def _deep_merge(target: dict, source: Mapping[str, JsonValue]) -> None:
    for key, value in source.items():
        normalized_key = str(key)
        source_mapping = json_mapping_or_none(value)
        target_mapping = json_mapping_or_none(target.get(normalized_key))
        if source_mapping is not None and target_mapping is not None:
            merged = mapping_to_json_object(target_mapping)
            _deep_merge(merged, mapping_to_json_object(source_mapping))
            target[normalized_key] = merged
            continue
        if source_mapping is not None:
            target[normalized_key] = _deep_json_object(mapping_to_json_object(source_mapping))
            continue
        target[normalized_key] = _json_value(value)


def _deep_json_object(value: Mapping[str, JsonValue]) -> dict:
    result: dict = {}
    _deep_merge(result, value)
    return result


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    mapping = json_mapping_or_none(value)
    if mapping is not None:
        return _deep_json_object(mapping_to_json_object(mapping))
    if is_json_list(value):
        return [_json_value(item) for item in value]
    return str(value)


def _string_dict(value: Mapping[str, JsonValue], *, field_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise TypeError(f"{field_name}.{key} 必须是字符串，实际为 {type(item).__name__}")
        result[str(key)] = item
    return result
