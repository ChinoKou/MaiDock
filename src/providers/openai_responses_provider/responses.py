import logging

from ...core.common import ProviderRuntimeOptions, build_openai_compatible_client_config, read_timeout
from ...schemas import ApiProviderSnapshot
from ..common.httpx import HttpxClientConfig
from ..responses_family.responses import ResponsesMapper

OPENAI_PROVIDER_LABEL = "OpenAI Responses"
OPENAI_API_PREFIX = "v1"
OPENAI_RESPONSES_ENDPOINT = "responses"
OPENAI_EMBEDDINGS_ENDPOINT = "embeddings"
OPENAI_AUDIO_TRANSCRIPTIONS_ENDPOINT = "audio/transcriptions"


def create_responses_mapper(*, options: ProviderRuntimeOptions, logger: logging.Logger) -> ResponsesMapper:
    return ResponsesMapper(
        options=options,
        logger=logger,
        provider_label=OPENAI_PROVIDER_LABEL,
        raw_provider="openai_responses",
        policy_provider="openai_responses",
    )


def build_client_config(api_provider: ApiProviderSnapshot, *, user_agent: str) -> HttpxClientConfig:
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
    )
