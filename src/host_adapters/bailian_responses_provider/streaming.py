from collections.abc import AsyncIterator, Mapping

from ...clients.common import SseJsonEvent
from ...core.json_types import JsonValue
from ..responses_family.streaming import collect_responses_stream
from .responses import BAILIAN_PROVIDER_LABEL


async def collect_bailian_response_stream(
    events: AsyncIterator[SseJsonEvent],
    *,
    model: str,
) -> Mapping[str, JsonValue]:
    return await collect_responses_stream(
        events,
        model=model,
        provider_label=BAILIAN_PROVIDER_LABEL,
        tool_fallback_prefix="bailian_tool",
    )
