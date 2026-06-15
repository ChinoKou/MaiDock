from collections.abc import Mapping

import httpx

from ..responses_family.streaming import collect_responses_stream
from .responses import VOLCENGINE_PROVIDER_LABEL


async def collect_ark_response_stream(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
    model: str,
    max_retries: int,
) -> Mapping:
    return await collect_responses_stream(
        client,
        path,
        body,
        headers=headers,
        query=query,
        model=model,
        provider_label=VOLCENGINE_PROVIDER_LABEL,
        tool_fallback_prefix="ark_tool",
        max_retries=max_retries,
    )
