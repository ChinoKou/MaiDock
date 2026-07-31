from collections.abc import Callable

from ...core.common import RuntimeOptionsView, build_usage_from_snapshot, read_model_identifier
from ...core.diagnostics import sanitize_json_object
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import ProviderPolicyKey
from ...i18n import translate
from ...schemas import EmbeddingRequestSnapshot, GenericUsageSnapshot, ProviderResponse
from ..common.embeddings import coerce_embedding_vector
from ..common.payloads import raw_data_or_none
from .parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
)

type EmbeddingParameterApplier = Callable[[TranslationContext, TranslationEnvelope], None]
type DimensionSupportPredicate = Callable[[str], bool]


class OpenAICompatibleEmbeddingMapper:
    """构建并解析 OpenAI 兼容的 Embedding 请求。"""

    def __init__(
        self,
        *,
        options: RuntimeOptionsView,
        provider_label: str,
        embedding_label: str,
        policy_provider: ProviderPolicyKey,
        apply_parameters: EmbeddingParameterApplier,
        supports_dimensions: DimensionSupportPredicate | None = None,
        include_default_encoding_format: bool = True,
        include_full_raw_data: bool = False,
    ) -> None:
        self.options = options
        self.provider_label = provider_label
        self.embedding_label = embedding_label
        self.policy_provider: ProviderPolicyKey = policy_provider
        self.apply_parameters = apply_parameters
        self.supports_dimensions = supports_dimensions
        self.include_default_encoding_format = include_default_encoding_format
        self.include_full_raw_data = include_full_raw_data

    def build_request(
        self,
        request: EmbeddingRequestSnapshot,
    ) -> tuple[dict[str, JsonValue], dict[str, str], dict[str, JsonValue], str]:
        model = read_model_identifier(request.model_info)
        overrides = self.options.parameter_overrides.get(self.policy_provider, "embeddings")
        catalog = get_parameter_catalog(self.policy_provider, "embeddings")
        context = build_translation_context(
            request,
            overrides=overrides,
            catalog=catalog,
            provider_label=self.provider_label,
            provider=self.policy_provider,
            capability="embeddings",
            model=model,
        )
        body: dict[str, JsonValue] = {"model": model, "input": request.embedding_input}
        if self.include_default_encoding_format:
            body["encoding_format"] = "float"
        envelope = TranslationEnvelope(body=body)
        self.apply_parameters(context, envelope)

        dimensions_supported = self.supports_dimensions is None or self.supports_dimensions(model)
        if not dimensions_supported and "dimensions" in context.normalized.fields:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=f"{self.provider_label} model {model} dimensions",
                    allowed="unset",
                )
            )
        if not dimensions_supported:
            envelope.body.pop("dimensions", None)
        encoding_format = str(envelope.body.get("encoding_format", "float"))
        return envelope.body, envelope.headers, envelope.query, encoding_format

    def extract_embedding(self, payload: dict[str, JsonValue], *, encoding_format: str = "float") -> list[float]:
        return extract_openai_compatible_embedding(
            payload,
            provider_label=self.embedding_label,
            encoding_format=encoding_format,
        )

    def build_response(
        self,
        payload: dict[str, JsonValue],
        *,
        encoding_format: str = "float",
    ) -> ProviderResponse:
        if self.include_full_raw_data:
            raw_data = raw_data_or_none(payload, options=self.options)
        else:
            raw_data = (
                sanitize_json_object({"model": payload.get("model"), "usage": payload.get("usage")})
                if self.options.include_raw_data
                else None
            )
        return ProviderResponse(
            embedding=self.extract_embedding(payload, encoding_format=encoding_format),
            usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {})),
            raw_data=raw_data,
        )


def extract_openai_compatible_embedding(
    payload: dict[str, JsonValue],
    *,
    provider_label: str,
    encoding_format: str = "float",
) -> list[float]:
    data_items = json_list_or_none(payload.get("data"))
    first_data = json_mapping_or_none(data_items[0]) if data_items else None
    candidate = first_data.get("embedding") if first_data is not None else None
    return coerce_embedding_vector(
        candidate,
        provider_label=provider_label,
        encoding_format=encoding_format,
    )
