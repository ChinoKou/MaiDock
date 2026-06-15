from ...core.common import ProviderRuntimeOptions, build_usage_from_snapshot, read_model_identifier
from ...core.json_types import json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.parameter_translation import build_translation_context, TranslationEnvelope
from ..common.payloads import raw_data_or_none
from .parameter_translation import apply_ark_embedding_parameters
from .responses import VOLCENGINE_PROVIDER_LABEL


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict, dict[str, str], dict, str]:
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("volcengine_ark", "embeddings")
    catalog = get_parameter_catalog("volcengine_ark", "embeddings")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        provider="volcengine_ark",
        capability="embeddings",
        model=model,
    )

    envelope = TranslationEnvelope(
        body={
            "model": model,
            "encoding_format": "float",
            "input": [{"type": "text", "text": request.embedding_input}],
        }
    )
    apply_ark_embedding_parameters(context, envelope)

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        capability="embeddings",
    )
    encoding_format = str(transport.body.get("encoding_format", "float"))
    return transport.body, transport.headers, transport.query, encoding_format


def build_ark_embedding_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
    encoding_format: str = "float",
) -> ProviderResponse:
    data = json_mapping_or_none(payload.get("data"))
    candidate = data.get("embedding") if data is not None else None
    return ProviderResponse(
        embedding=coerce_embedding_vector(
            candidate, provider_label="Volcengine Ark Embeddings", encoding_format=encoding_format
        ),
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=raw_data_or_none(payload, options=options),
    )
