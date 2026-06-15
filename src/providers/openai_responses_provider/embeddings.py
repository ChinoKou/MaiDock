from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    read_model_identifier,
)
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.parameter_translation import (
    build_translation_context,
    TranslationEnvelope,
)
from .parameter_translation import apply_openai_embedding_parameters
from .responses import OPENAI_PROVIDER_LABEL


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict, dict[str, str], dict, str]:
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("openai_responses", "embeddings")
    catalog = get_parameter_catalog("openai_responses", "embeddings")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=OPENAI_PROVIDER_LABEL,
        provider="openai_responses",
        capability="embeddings",
        model=model,
    )
    envelope = TranslationEnvelope(body={"model": model, "input": request.embedding_input})
    apply_openai_embedding_parameters(context, envelope)

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=OPENAI_PROVIDER_LABEL,
        capability="embeddings",
    )
    encoding_format = str(transport.body.get("encoding_format", "float"))
    return transport.body, transport.headers, transport.query, encoding_format


def extract_openai_embedding(payload: dict, *, encoding_format: str = "float") -> list[float]:
    data_items = json_list_or_none(payload.get("data"))
    first_data = json_mapping_or_none(data_items[0]) if data_items else None
    candidate = first_data.get("embedding") if first_data is not None else None
    return coerce_embedding_vector(candidate, provider_label="OpenAI Embeddings", encoding_format=encoding_format)


def build_openai_embedding_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
    encoding_format: str = "float",
) -> ProviderResponse:
    usage = build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {}))
    raw_data = (
        sanitize_json_object({"model": payload.get("model"), "usage": payload.get("usage")})
        if options.include_raw_data
        else None
    )
    return ProviderResponse(
        embedding=extract_openai_embedding(payload, encoding_format=encoding_format),
        usage=usage,
        raw_data=raw_data,
    )
