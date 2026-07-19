from ...i18n import runtime_expected, translate
from .parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    normalize_dimensions,
    run_translators,
    set_target_value,
)


def translate_embedding_dimensions(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "dimensions"), normalize_dimensions(value))


def translate_embedding_encoding_format(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if not isinstance(value, str):
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="encoding_format",
                expected=runtime_expected("string"),
                actual=type(value).__name__,
            )
        )
    set_target_value(envelope, ("body", "encoding_format"), value.strip().lower() or "float")


def translate_embedding_user(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "user"), value)


def translate_embedding_sparse_embedding(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if not isinstance(value, bool):
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="sparse_embedding",
                expected=runtime_expected("boolean"),
                actual=type(value).__name__,
            )
        )
    set_target_value(envelope, ("body", "sparse_embedding"), {"type": "enabled" if value else "disabled"})


EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": translate_embedding_dimensions,
    "encoding_format": translate_embedding_encoding_format,
    "user": translate_embedding_user,
    "sparse_embedding": translate_embedding_sparse_embedding,
}


def apply_embedding_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    translators: dict[str, FieldTranslator] | None = None,
) -> None:
    run_translators(context, envelope, EMBEDDING_TRANSLATORS if translators is None else translators)
