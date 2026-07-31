from collections.abc import Callable

from ...core.common import RuntimeOptionsView, build_usage_from_snapshot, read_model_identifier
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import ProviderPolicyKey
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.payloads import raw_data_or_none
from .parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
)
from ...core.json_types import JsonValue

type EmbeddingBodyBuilder = Callable[[str, EmbeddingRequestSnapshot], dict[str, JsonValue]]
type EmbeddingParameterApplier = Callable[[TranslationContext, TranslationEnvelope], None]


def build_responses_family_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: RuntimeOptionsView,
    provider_label: str,
    policy_provider: ProviderPolicyKey,
    build_body: EmbeddingBodyBuilder,
    apply_parameters: EmbeddingParameterApplier,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue], str]:
    """通过 Responses Provider 的公共参数管线构建 Embedding 请求。"""

    model = read_model_identifier(request.model_info)
    overrides = options.parameter_overrides.get(policy_provider, "embeddings")
    catalog = get_parameter_catalog(policy_provider, "embeddings")
    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=provider_label,
        provider=policy_provider,
        capability="embeddings",
        model=model,
    )
    envelope = TranslationEnvelope(body=build_body(model, request))
    apply_parameters(context, envelope)
    encoding_format = str(envelope.body.get("encoding_format", "float"))
    return envelope.body, envelope.headers, envelope.query, encoding_format


def build_responses_family_embedding_response(
    candidate: object,
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
    provider_label: str,
    encoding_format: str = "float",
) -> ProviderResponse:
    """校验 Responses Provider 的向量并构造统一响应。"""

    return ProviderResponse(
        embedding=coerce_embedding_vector(
            candidate,
            provider_label=provider_label,
            encoding_format=encoding_format,
        ),
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=raw_data_or_none(payload, options=options),
    )
