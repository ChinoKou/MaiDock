from ..common.embeddings_translation import (
    EMBEDDING_TRANSLATORS,
    apply_embedding_parameters,
)
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    build_translation_context,
    run_translators,
    set_target_value,
)


def translate_audio_identity(target_path: tuple[str, ...], *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, target_path, value)

    _translator.__name__ = f"translate_audio_{field_name}"
    return _translator


OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS: dict[str, FieldTranslator] = {
    "language": translate_audio_identity(("body", "language"), field_name="language"),
    "prompt": translate_audio_identity(("body", "prompt"), field_name="prompt"),
    "response_format": translate_audio_identity(("body", "response_format"), field_name="response_format"),
    "temperature": translate_audio_identity(("body", "temperature"), field_name="temperature"),
    "timestamp_granularities": translate_audio_identity(
        ("body", "timestamp_granularities"),
        field_name="timestamp_granularities",
    ),
    "chunking_strategy": translate_audio_identity(
        ("body", "chunking_strategy"),
        field_name="chunking_strategy",
    ),
    "include": translate_audio_identity(("body", "include"), field_name="include"),
    "stream": translate_audio_identity(("body", "stream"), field_name="stream"),
}


def apply_openai_audio_transcription_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    translators: dict[str, FieldTranslator] | None = None,
) -> None:
    run_translators(
        context,
        envelope,
        OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS if translators is None else translators,
    )


__all__ = [
    "EMBEDDING_TRANSLATORS",
    "OPENAI_AUDIO_TRANSCRIPTION_TRANSLATORS",
    "FieldTranslator",
    "TranslationContext",
    "TranslationEnvelope",
    "apply_embedding_parameters",
    "apply_openai_audio_transcription_parameters",
    "build_translation_context",
    "run_translators",
    "set_target_value",
]
