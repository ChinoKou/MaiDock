from collections.abc import AsyncIterator

from ...clients.common import SseJsonEvent

from ...core.common import RuntimeOptionsView
from ...schemas import ProviderResponse
from ..chat_completions_family.streaming import collect_chat_completions_stream
from .chat import SILICONFLOW_PROVIDER_LABEL


async def collect_stream_response(
    events: AsyncIterator[SseJsonEvent],
    *,
    options: RuntimeOptionsView,
) -> ProviderResponse:
    return await collect_chat_completions_stream(
        events,
        options=options,
        provider_label=SILICONFLOW_PROVIDER_LABEL,
    )
