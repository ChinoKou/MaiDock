from ..responses_family.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
    apply_responses_parameters,
)


def apply_bailian_responses_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    apply_responses_parameters(context, envelope)
