from ...core.common import RuntimeOptionsView
from ...schemas import EmbeddingRequestSnapshot, ProviderResponse
from ..openai_auxiliary_family.embeddings import (
    OpenAICompatibleEmbeddingMapper,
    extract_openai_compatible_embedding,
)
from .parameter_translation import apply_openai_embedding_parameters
from .responses import OPENAI_PROVIDER_LABEL
from ...core.json_types import JsonValue


def _create_mapper(options: RuntimeOptionsView) -> OpenAICompatibleEmbeddingMapper:
    return OpenAICompatibleEmbeddingMapper(
        options=options,
        provider_label=OPENAI_PROVIDER_LABEL,
        embedding_label="OpenAI Embeddings",
        policy_provider="openai_responses",
        apply_parameters=apply_openai_embedding_parameters,
        include_default_encoding_format=False,
    )


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue], str]:
    return _create_mapper(options).build_request(request)


def extract_openai_embedding(payload: dict[str, JsonValue], *, encoding_format: str = "float") -> list[float]:
    return extract_openai_compatible_embedding(
        payload,
        provider_label="OpenAI Embeddings",
        encoding_format=encoding_format,
    )


def build_openai_embedding_response(
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
    encoding_format: str = "float",
) -> ProviderResponse:
    return _create_mapper(options).build_response(payload, encoding_format=encoding_format)
