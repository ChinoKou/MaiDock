import re
from typing import Literal

from ...core.common import RuntimeOptionsView, build_usage_from_snapshot, read_model_identifier
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...i18n import translate
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.parameter_translation import TranslationEnvelope, build_translation_context
from ..common.payloads import raw_data_or_none
from .chat import DASHSCOPE_PROVIDER_LABEL
from .errors import raise_for_dashscope_error
from .parameter_translation import apply_dashscope_embedding_parameters

DASHSCOPE_TEXT_EMBEDDING_ENDPOINT = "services/embeddings/text-embedding/text-embedding"
DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT = "services/embeddings/multimodal-embedding/multimodal-embedding"
QWEN_VL_EMBEDDING_PATTERN = re.compile(r"^qwen.*-vl-embedding$", re.IGNORECASE)
DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS = {
    "multimodal-embedding-one-peace-v1",
    "multimodal-embedding-v1",
}

EmbeddingEndpoint = Literal[
    "services/embeddings/text-embedding/text-embedding",
    "services/embeddings/multimodal-embedding/multimodal-embedding",
]


def dashscope_embedding_endpoint(model: str) -> EmbeddingEndpoint:
    normalized = model.strip().lower()
    if normalized.startswith(("text-embedding-v", "qwen3.7-text-embedding")):
        return DASHSCOPE_TEXT_EMBEDDING_ENDPOINT
    if normalized in DASHSCOPE_MULTIMODAL_EMBEDDING_MODELS:
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    if QWEN_VL_EMBEDDING_PATTERN.match(normalized):
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    if normalized.startswith("tongyi-embedding-vision-"):
        return DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT
    raise ValueError(
        translate(
            "runtime.error.unsupported_value",
            subject=f"{DASHSCOPE_PROVIDER_LABEL} embedding model",
            allowed=(
                "text-embedding-v*/qwen3.7-text-embedding*/multimodal-embedding-*/"
                "qwen*-vl-embedding/tongyi-embedding-vision-*"
            ),
        )
    )


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[EmbeddingEndpoint, dict[str, JsonValue], dict[str, str], dict[str, JsonValue]]:
    model = read_model_identifier(request.model_info)
    endpoint = dashscope_embedding_endpoint(model)
    overrides = options.parameter_overrides.get("dashscope", "embeddings")
    catalog = get_parameter_catalog("dashscope", "embeddings")

    context = build_translation_context(
        request,
        overrides=overrides,
        catalog=catalog,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        provider="dashscope",
        capability="embeddings",
        model=model,
    )

    parameters: dict[str, JsonValue] = {}
    if endpoint == DASHSCOPE_TEXT_EMBEDDING_ENDPOINT:
        input_value: dict[str, JsonValue] = {"texts": [request.embedding_input]}
    else:
        input_value = {"contents": [{"text": request.embedding_input}]}
        if model.strip().lower() == "qwen3-vl-embedding" and "enable_fusion" not in context.normalized.fields:
            context.normalized.fields["enable_fusion"] = True
            context.normalized.sources["enable_fusion"] = "provider.default"

    body = {"model": model, "input": input_value, "parameters": parameters}
    envelope = TranslationEnvelope(body=body)
    apply_dashscope_embedding_parameters(context, envelope)
    return endpoint, envelope.body, envelope.headers, envelope.query


def build_dashscope_embedding_response(
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
) -> ProviderResponse:
    raise_for_dashscope_error(payload)
    output = json_mapping_or_none(payload.get("output"))
    embeddings = json_list_or_none(output.get("embeddings")) if output is not None else None
    first_embedding = json_mapping_or_none(embeddings[0]) if embeddings else None
    candidate = first_embedding.get("embedding") if first_embedding is not None else None
    return ProviderResponse(
        embedding=coerce_embedding_vector(
            candidate,
            provider_label=f"{DASHSCOPE_PROVIDER_LABEL} Embeddings",
        ),
        usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
        raw_data=raw_data_or_none(payload, options=options),
    )
