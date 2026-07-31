from ...core.common import RuntimeOptionsView
from ...core.json_types import JsonValue, json_mapping_or_none
from ...schemas import EmbeddingRequestSnapshot, ProviderResponse
from ..responses_family.embeddings import (
    build_responses_family_embedding_request,
    build_responses_family_embedding_response,
)
from .parameter_translation import apply_ark_embedding_parameters
from .responses import VOLCENGINE_PROVIDER_LABEL


def _build_ark_embedding_body(model: str, request: EmbeddingRequestSnapshot) -> dict[str, JsonValue]:
    return {
        "model": model,
        "encoding_format": "float",
        "input": [{"type": "text", "text": request.embedding_input}],
    }


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue], str]:
    return build_responses_family_embedding_request(
        request,
        options=options,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        policy_provider="volcengine_ark",
        build_body=_build_ark_embedding_body,
        apply_parameters=apply_ark_embedding_parameters,
    )


def build_ark_embedding_response(
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
    encoding_format: str = "float",
) -> ProviderResponse:
    data = json_mapping_or_none(payload.get("data"))
    candidate = data.get("embedding") if data is not None else None
    return build_responses_family_embedding_response(
        candidate,
        payload,
        options=options,
        provider_label="Volcengine Ark Embeddings",
        encoding_format=encoding_format,
    )
