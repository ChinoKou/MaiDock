from ..openai_auxiliary_family.parameter_translation import (
    EMBEDDING_TRANSLATORS,
    OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS,
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    apply_embedding_parameters,
    apply_openai_audio_transcription_parameters,
)
from ..responses_family.parameter_translation import apply_responses_parameters

OPENAI_EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": EMBEDDING_TRANSLATORS["dimensions"],
    "encoding_format": EMBEDDING_TRANSLATORS["encoding_format"],
    "user": EMBEDDING_TRANSLATORS["user"],
}
OPENAI_AUDIO_TRANSLATORS: dict[str, FieldTranslator] = dict(OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS)


def apply_openai_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_embedding_parameters(context, envelope, OPENAI_EMBEDDING_TRANSLATORS)


def apply_openai_audio_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_openai_audio_transcription_parameters(context, envelope, OPENAI_AUDIO_TRANSLATORS)


def apply_openai_responses_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_responses_parameters(context, envelope)
