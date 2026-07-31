from collections.abc import AsyncIterator, Mapping

from ...clients.common import SseJsonEvent

from ..responses_family.streaming import collect_responses_stream
from .responses import OPENAI_PROVIDER_LABEL
from ...core.json_types import JsonValue


async def collect_openai_response_stream(
    events: AsyncIterator[SseJsonEvent],
    *,
    model: str,
) -> Mapping[str, JsonValue]:
    return await collect_responses_stream(
        events,
        model=model,
        provider_label=OPENAI_PROVIDER_LABEL,
        tool_fallback_prefix="openai_tool",
    )
