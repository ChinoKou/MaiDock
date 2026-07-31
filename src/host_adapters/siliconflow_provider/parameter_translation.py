from ..chat_completions_family.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    apply_chat_completions_family_parameters,
)
from ..openai_auxiliary_family.parameter_translation import (
    EMBEDDING_TRANSLATORS,
    OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS,
    FieldTranslator,
    apply_embedding_parameters,
    apply_openai_audio_transcription_parameters,
)

SILICONFLOW_EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": EMBEDDING_TRANSLATORS["dimensions"],
    "encoding_format": EMBEDDING_TRANSLATORS["encoding_format"],
}
SILICONFLOW_AUDIO_TRANSLATORS: dict[str, FieldTranslator] = dict(OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS)


def apply_siliconflow_chat_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_chat_completions_family_parameters(context, envelope)


def apply_siliconflow_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_embedding_parameters(context, envelope, SILICONFLOW_EMBEDDING_TRANSLATORS)


def apply_siliconflow_audio_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_openai_audio_transcription_parameters(context, envelope, SILICONFLOW_AUDIO_TRANSLATORS)
