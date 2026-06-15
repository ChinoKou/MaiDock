from collections.abc import Mapping

import httpx

from ...core.common import ProviderRuntimeOptions
from ...schemas import ProviderResponse
from ..chat_completions_family.streaming import collect_chat_completions_stream

MIMO_PROVIDER_LABEL = "Xiaomi Mimo"


async def collect_mimo_stream_response(
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
        provider_label=MIMO_PROVIDER_LABEL,
        max_retries=max_retries,
        retry_interval=retry_interval,
    )
