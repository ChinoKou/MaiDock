from ..common.embeddings_translation import (
    EMBEDDING_TRANSLATORS,
    apply_embedding_parameters,
)
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    run_translators,
    set_target_value,
)
from ..responses_family.parameter_translation import RESPONSES_TRANSLATORS, apply_responses_parameters

ARK_EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": EMBEDDING_TRANSLATORS["dimensions"],
    "sparse_embedding": EMBEDDING_TRANSLATORS["sparse_embedding"],
    "encoding_format": EMBEDDING_TRANSLATORS["encoding_format"],
}


def _translate_ark_audio_identity(target_path: tuple[str, ...], *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, target_path, value)

    _translator.__name__ = f"translate_ark_audio_{field_name}"
    return _translator


ARK_AUDIO_TRANSLATORS: dict[str, FieldTranslator] = {
    "audio_format": _translate_ark_audio_identity(("body", "audio_format"), field_name="audio_format"),
    "format": _translate_ark_audio_identity(("body", "format"), field_name="format"),
    "max_tokens": RESPONSES_TRANSLATORS["max_tokens"],
    "prompt": _translate_ark_audio_identity(("body", "prompt"), field_name="prompt"),
}


def apply_ark_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_embedding_parameters(context, envelope, ARK_EMBEDDING_TRANSLATORS)


def apply_ark_audio_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, ARK_AUDIO_TRANSLATORS)


def apply_ark_responses_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_responses_parameters(context, envelope)
