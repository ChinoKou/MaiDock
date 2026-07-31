import logging

from ...core.common import RuntimeOptionsView
from ...schemas import (
    MessageSnapshot,
    OpenAIInputImageBlock,
    OpenAIInputTextBlock,
    OpenAIResponseOutputItem,
    OpenAIResponseSnapshot,
    OpenAIResponsesTool,
    ProviderToolCall,
    ToolOptionSnapshot,
)
from ..responses_family.parameter_translation import TranslationContext, TranslationEnvelope
from ..responses_family.responses import ResponsesMapper
from .parameter_translation import apply_bailian_responses_parameters

BAILIAN_PROVIDER_LABEL = "阿里云百炼 Responses"


class BailianResponsesMapper(ResponsesMapper):
    """通过百炼 Provider 门面调用 Responses Family。"""

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
        from .tools import convert_tools

        return convert_tools(tool_options)

    def _convert_user_content_parts(
        self,
        message: MessageSnapshot,
    ) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
        from .multimodal import convert_user_content_parts

        return convert_user_content_parts(message, logger=self.logger, options=self.options)

    def _apply_response_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_bailian_responses_parameters(context, envelope)

    def _extract_tool_calls(self, output: list[OpenAIResponseOutputItem]) -> list[ProviderToolCall]:
        from .tools import extract_tool_calls

        return extract_tool_calls(output, options=self.options)

    def _extract_text_content(self, response_model: OpenAIResponseSnapshot) -> str:
        from .multimodal import extract_text_content

        return extract_text_content(response_model)

    def _extract_reasoning_content(self, output: list[OpenAIResponseOutputItem]) -> str | None:
        from .multimodal import extract_reasoning_content

        return extract_reasoning_content(output)


def create_responses_mapper(*, options: RuntimeOptionsView, logger: logging.Logger) -> ResponsesMapper:
    return BailianResponsesMapper(
        options=options,
        logger=logger,
        provider_label=BAILIAN_PROVIDER_LABEL,
        raw_provider="bailian_responses",
        policy_provider="bailian_responses",
    )
