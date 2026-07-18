import logging

from ...core.common import (
    ProviderRuntimeOptions,
    build_openai_compatible_client_config,
    read_timeout,
    resolve_max_retries,
    resolve_retry_interval,
)
from ...schemas import (
    ApiProviderSnapshot,
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
from ..responses_family.transport import HttpxClientConfig
from . import multimodal, tools
from .parameter_translation import apply_openai_responses_parameters

OPENAI_PROVIDER_LABEL = "OpenAI Responses"
OPENAI_API_PREFIX = "v1"
OPENAI_RESPONSES_ENDPOINT = "responses"
OPENAI_EMBEDDINGS_ENDPOINT = "embeddings"
OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT = "audio/transcriptions"


class OpenAIResponsesMapper(ResponsesMapper):
    """通过 OpenAI Provider 门面调用 Responses Family。"""

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
        return tools.convert_tools(tool_options)

    def _convert_user_content_parts(
        self,
        message: MessageSnapshot,
    ) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
        return multimodal.convert_user_content_parts(message, logger=self.logger, options=self.options)

    def _apply_response_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_openai_responses_parameters(context, envelope)

    def _extract_tool_calls(self, output: list[OpenAIResponseOutputItem]) -> list[ProviderToolCall]:
        return tools.extract_tool_calls(output, options=self.options)

    def _extract_text_content(self, response_model: OpenAIResponseSnapshot) -> str:
        return multimodal.extract_text_content(response_model)

    def _extract_reasoning_content(self, output: list[OpenAIResponseOutputItem]) -> str | None:
        return multimodal.extract_reasoning_content(output)


def create_responses_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ResponsesMapper:
    return OpenAIResponsesMapper(
        options=options,
        logger=logger,
        provider_label=OPENAI_PROVIDER_LABEL,
        raw_provider="openai_responses",
        policy_provider="openai_responses",
    )


def build_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str,
    default_max_retries: int = 3,
    force_max_retries: bool = False,
    default_retry_interval: float = 5.0,
    force_retry_interval: bool = False,
) -> HttpxClientConfig:
    client_config = build_openai_compatible_client_config(api_provider, user_agent=user_agent)
    headers = dict(client_config.default_headers)
    if client_config.api_key:
        headers.setdefault("Authorization", f"Bearer {client_config.api_key}")
    if api_provider.organization is not None and api_provider.organization.strip():
        headers.setdefault("OpenAI-Organization", api_provider.organization.strip())
    if api_provider.project is not None and api_provider.project.strip():
        headers.setdefault("OpenAI-Project", api_provider.project.strip())
    headers.setdefault("Accept", "application/json")
    return HttpxClientConfig(
        base_url=client_config.base_url,
        default_headers=headers,
        default_query=dict(client_config.default_query),
        timeout=read_timeout(api_provider),
        max_retries=resolve_max_retries(
            api_provider,
            config_value=default_max_retries,
            force=force_max_retries,
            default=3,
        ),
        retry_interval=resolve_retry_interval(
            api_provider,
            config_value=default_retry_interval,
            force=force_retry_interval,
            default=5.0,
        ),
    )
