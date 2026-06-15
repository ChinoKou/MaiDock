import re
from typing import Literal

from ...core.common import ProviderRuntimeOptions, build_usage_from_snapshot, read_model_identifier
from ...core.json_types import normalize_json_value, json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import apply_transport_parameter_policy
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.parameter_translation import build_translation_context, TranslationEnvelope
from ..common.payloads import raw_data_or_none
from .chat import DASHSCOPE_PROVIDER_LABEL
from .parameter_translation import apply_dashscope_embedding_parameters

DASHSCOPE_TEXT_EMBEDDING_ENDPOINT = "services/embeddings/text-embedding/text-embedding"
DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT = "services/embeddings/multimodal-embedding/multimodal-embedding"
QWEN_VL_EMBEDDING_PATTERN = re.compile(r"^qwen.*-vl-embedding$", re.IGNORECASE)
DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS = {"multimodal-embedding-one-peace-v1", "multimodal-embedding-v1"}

EmbeddingEndpoint = Literal[
    "services/embeddings/text-embedding/text-embedding",
    "services/embeddings/multimodal-embedding/multimodal-embedding",
]


def dashscope_embedding_endpoint(model: str) -> EmbeddingEndpoint:
    normalized = model.strip().lower()
    if normalized.startswith("text-embedding-v"):
        return DASHSCOPE_TEXT_EMBEDDING_ENDPOINT
    if normalized in DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS:
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    if QWEN_VL_EMBEDDING_PATTERN.match(normalized):
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    if normalized.startswith("tongyi-embedding-vision-"):
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    raise ValueError(
        "DashScope embedding 模型必须匹配 text-embedding-v*、multimodal-embedding-*、"
        "qwen*-vl-embedding 或 tongyi-embedding-vision-*"
    )


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: ProviderRuntimeOptions,
) -> tuple[EmbeddingEndpoint, dict, dict[str, str], dict, str]:
    model = read_model_identifier(request.model_info)
    endpoint = dashscope_embedding_endpoint(model)
    policy = options.parameter_policies.get("dashscope", "embeddings")
    catalog = get_parameter_catalog("dashscope", "embeddings")

    context = build_translation_context(
        request,
        policy=policy,
        catalog=catalog,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        provider="dashscope",
        capability="embeddings",
        model=model,
    )

    parameters: dict = {}
    if endpoint == DASHSCOPE_TEXT_EMBEDDING_ENDPOINT:
        input_value: dict = {"texts": [request.embedding_input]}
    else:
        factor = normalize_json_value(context.normalized.fields.pop("factor", 1.0))
        input_value = {"contents": [{"text": request.embedding_input, "factor": factor}]}
        if model.strip().lower() == "qwen3-vl-embedding" and "enable_fusion" not in context.normalized.fields:
            context.normalized.fields["enable_fusion"] = True
            context.normalized.sources["enable_fusion"] = "provider.default"

    body = {"model": model, "input": input_value, "parameters": parameters}
    envelope = TranslationEnvelope(body=body)
    apply_dashscope_embedding_parameters(context, envelope)

    transport = apply_transport_parameter_policy(
        body=envelope.body,
        headers=envelope.headers,
        query=envelope.query,
        policy=policy,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        capability="embeddings",
    )
    params = json_mapping_or_none(transport.body.get("parameters"))
    encoding_format = str(params.get("encoding_format", "float")) if params is not None else "float"
    return endpoint, transport.body, transport.headers, transport.query, encoding_format


def build_dashscope_embedding_response(
    payload: dict,
    *,
    options: ProviderRuntimeOptions,
    encoding_format: str = "float",
) -> ProviderResponse:
    output = json_mapping_or_none(payload.get("output"))
    embeddings = json_list_or_none(output.get("embeddings")) if output is not None else None
    first_embedding = json_mapping_or_none(embeddings[0]) if embeddings else None
    candidate = first_embedding.get("embedding") if first_embedding is not None else None
    return ProviderResponse(
        embedding=coerce_embedding_vector(
            candidate, provider_label="DashScope Embeddings", encoding_format=encoding_format
        ),
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=raw_data_or_none(payload, options=options),
    )
