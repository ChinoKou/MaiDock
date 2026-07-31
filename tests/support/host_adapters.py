"""现有供应商契约测试使用的 Client + Host Adapter 构造器。"""

import httpx

from src.clients.anthropic import AnthropicClient
from src.clients.ark import ArkClient
from src.clients.dashscope import DashScopeClient
from src.clients.mimo import MimoClient
from src.clients.openai import OpenAIClient
from src.clients.siliconflow import SiliconFlowClient
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.host_adapters.anthropic_messages_provider.adapter import AnthropicHostAdapter
from src.host_adapters.bailian_responses_provider.adapter import BailianResponsesHostAdapter
from src.host_adapters.dashscope_provider.adapter import DashScopeHostAdapter
from src.host_adapters.openai_responses_provider.adapter import OpenAIHostAdapter
from src.host_adapters.siliconflow_provider.adapter import SiliconFlowHostAdapter
from src.host_adapters.volcengine_ark_provider.adapter import ArkHostAdapter
from src.host_adapters.xiaomi_mimo_provider.adapter import MimoHostAdapter


class OpenAIResponsesProvider(OpenAIHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(options=options, client=OpenAIClient(transport=transport))


class AnthropicMessagesProvider(AnthropicHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(options=options, client=AnthropicClient(transport=transport))


class VolcengineArkResponsesProvider(ArkHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
        state_store: PluginStateStore | None = None,
    ) -> None:
        super().__init__(
            options=options,
            client=ArkClient(transport=transport),
            state_store=state_store,
        )


class DashScopeProvider(DashScopeHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(options=options, client=DashScopeClient(transport=transport))


class BailianResponsesProvider(BailianResponsesHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(options=options, client=DashScopeClient(transport=transport))


class SiliconFlowProvider(SiliconFlowHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(options=options, client=SiliconFlowClient(transport=transport))


class XiaomiMimoProvider(MimoHostAdapter):
    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        transport: httpx.AsyncBaseTransport | None = None,
        state_store: PluginStateStore | None = None,
    ) -> None:
        super().__init__(
            options=options,
            client=MimoClient(transport=transport),
            state_store=state_store,
        )
