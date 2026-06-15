from collections.abc import Mapping

import httpx

from ...core.common import ProviderRuntimeOptions
from ...schemas import ProviderResponse
from ..chat_completions_family.streaming import collect_chat_completions_stream
from .chat import SILICONFLOW_PROVIDER_LABEL


async def collect_stream_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
    options: ProviderRuntimeOptions,
    max_retries: int,
    retry_interval: float,
) -> ProviderResponse:
    return await collect_chat_completions_stream(
        client,
        path,
        body,
        headers=headers,
        query=query,
        options=options,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
        max_retries=max_retries,
        retry_interval=retry_interval,
    )
