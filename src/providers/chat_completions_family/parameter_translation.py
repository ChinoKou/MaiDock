from collections.abc import Mapping

from ..common.chat_completions_translation import (
    CHAT_COMPLETIONS_TRANSLATORS,
    apply_chat_completions_parameters,
)
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
)

CHAT_COMPLETIONS_FAMILY_TRANSLATORS: dict[str, FieldTranslator] = dict(CHAT_COMPLETIONS_TRANSLATORS)


def apply_chat_completions_family_parameters(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    *,
    extra_translators: Mapping[str, FieldTranslator] | None = None,
) -> None:
    """应用 Chat Completions 公共参数转译，可选注入 Provider 特有 translator。"""
    apply_chat_completions_parameters(context, envelope, extra_translators=extra_translators)
