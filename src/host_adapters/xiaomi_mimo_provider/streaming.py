from collections.abc import AsyncIterator

from ...clients.common import SseJsonEvent

from ...core.common import RuntimeOptionsView
from ...schemas import ProviderResponse
from ..chat_completions_family.streaming import collect_chat_completions_stream

MIMO_PROVIDER_LABEL = "Xiaomi Mimo"


async def collect_mimo_stream_response(
    events: AsyncIterator[SseJsonEvent],
    *,
    options: RuntimeOptionsView,
) -> ProviderResponse:
    return await collect_chat_completions_stream(
        events,
        options=options,
        provider_label=MIMO_PROVIDER_LABEL,
    )
