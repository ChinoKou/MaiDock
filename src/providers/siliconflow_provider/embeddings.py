from ...core.common import (
    ProviderRuntimeOptions,
    build_usage_from_snapshot,
    read_model_identifier,
)
from ...core.json_types import json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.parameter_translation import (
    build_translation_context,
    TranslationEnvelope,
)
from ..common.payloads import raw_data_or_none
from .chat import SILICONFLOW_PROVIDER_LABEL, qwen_supports_dimensions
from .parameter_translation import apply_siliconflow_embedding_parameters

SILICONFLOW_EMBEDDINGS_ENDPOINT = "embeddings"


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[dict, dict[str, str], dict, str]:
    model = read_model_identifier(request.model_info)
    policy = options.parameter_policies.get("siliconflow", "embeddings")
    catalog = get_parameter_catalog("siliconflow", "embeddings")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        provider="siliconflow",
        capability="embeddings",
        model=model,
    )

    envelope = TranslationEnvelope(
        body={
            "model": model,
            "input": request.embedding_input,
            "encoding_format": "float",
        }
    )
    apply_siliconflow_embedding_parameters(context, envelope)

    if not qwen_supports_dimensions(model) and "dimensions" in context.normalized.fields:
        raise ValueError(f"SiliconFlow 模型 {model} 不支持 dimensions 参数")

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        capability="embeddings",
    )
    if not qwen_supports_dimensions(model):
        transport.body.pop("dimensions", None)
    encoding_format = str(transport.body.get("encoding_format", "float"))
    return transport.body, transport.headers, transport.query, encoding_format


def build_siliconflow_embedding_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
    encoding_format: str = "float",
) -> ProviderResponse:
    data_items = json_list_or_none(payload.get("data"))
    first_data = json_mapping_or_none(data_items[0]) if data_items else None
    candidate = first_data.get("embedding") if first_data is not None else None
    return ProviderResponse(
        embedding=coerce_embedding_vector(
            candidate,
            provider_label="SiliconFlow Embeddings",
            encoding_format=encoding_format,
        ),
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=raw_data_or_none(payload, options=options),
    )
