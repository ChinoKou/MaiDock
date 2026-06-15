from ..common.embeddings_translation import EMBEDDING_TRANSLATORS, apply_embedding_parameters
from ..common.parameter_translation import FieldTranslator, TranslationContext, TranslationEnvelope
from ..responses_family.parameter_translation import apply_responses_parameters

ARK_EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": EMBEDDING_TRANSLATORS["dimensions"],
    "sparse_embedding": EMBEDDING_TRANSLATORS["sparse_embedding"],
    "encoding_format": EMBEDDING_TRANSLATORS["encoding_format"],
}


def apply_ark_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_embedding_parameters(context, envelope, ARK_EMBEDDING_TRANSLATORS)


def apply_ark_responses_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_responses_parameters(context, envelope)
