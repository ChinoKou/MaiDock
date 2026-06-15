from ..chat_completions_family.parameter_translation import apply_chat_completions_family_parameters
from ..common.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    set_target_value,
)


def _translate_mimo_thinking(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "thinking"), value)


def apply_mimo_chat_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
) -> None:
    apply_chat_completions_family_parameters(
        context,
        envelope,
        extra_translators={"thinking": _translate_mimo_thinking},
    )
