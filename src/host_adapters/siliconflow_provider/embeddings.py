from ...core.common import RuntimeOptionsView
from ...schemas import EmbeddingRequestSnapshot, ProviderResponse
from ..openai_auxiliary_family.embeddings import OpenAICompatibleEmbeddingMapper
from .chat import SILICONFLOW_PROVIDER_LABEL, qwen_supports_dimensions
from .parameter_translation import apply_siliconflow_embedding_parameters
from ...core.json_types import JsonValue

SILICONFLOW_EMBEDDINGS_ENDPOINT = "embeddings"


def _create_mapper(options: RuntimeOptionsView) -> OpenAICompatibleEmbeddingMapper:
    return OpenAICompatibleEmbeddingMapper(
        options=options,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        embedding_label="SiliconFlow Embeddings",
        policy_provider="siliconflow",
        apply_parameters=apply_siliconflow_embedding_parameters,
        supports_dimensions=qwen_supports_dimensions,
        include_full_raw_data=True,
    )


def build_embedding_request(
    request: EmbeddingRequestSnapshot,
    *,
    options: RuntimeOptionsView,
) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue], str]:
    return _create_mapper(options).build_request(request)


def build_siliconflow_embedding_response(
    payload: dict[str, JsonValue],
    *,
    options: RuntimeOptionsView,
    encoding_format: str = "float",
) -> ProviderResponse:
    return _create_mapper(options).build_response(payload, encoding_format=encoding_format)
