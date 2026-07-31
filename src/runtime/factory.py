from ..clients.anthropic import AnthropicClient
from ..clients.ark import ArkClient
from ..clients.dashscope import DashScopeClient
from ..clients.mimo import MimoClient
from ..clients.openai import OpenAIClient
from ..clients.siliconflow import SiliconFlowClient
from ..core.common import ProviderRuntimeOptions
from ..core.state_store import PluginStateStore
from ..host_adapters.anthropic_messages_provider.adapter import AnthropicHostAdapter
from ..host_adapters.bailian_responses_provider.adapter import BailianResponsesHostAdapter
from ..host_adapters.dashscope_provider.adapter import DashScopeHostAdapter
from ..host_adapters.openai_responses_provider.adapter import OpenAIHostAdapter
from ..host_adapters.siliconflow_provider.adapter import SiliconFlowHostAdapter
from ..host_adapters.volcengine_ark_provider.adapter import ArkHostAdapter
from ..host_adapters.xiaomi_mimo_provider.adapter import MimoHostAdapter
from .container import VendorRuntime
from .contracts import HostAdapter, ProviderCapability, RuntimeKey, VendorClient, VendorClientKey
from .ingress import LLMProviderIngress

_CAPABILITIES: dict[RuntimeKey, frozenset[ProviderCapability]] = {
    "openai": frozenset({"response", "embedding", "audio_transcription"}),
    "anthropic": frozenset({"response"}),
    "volcengine": frozenset({"response", "embedding", "audio_transcription"}),
    "dashscope": frozenset({"response", "embedding", "audio_transcription"}),
    "bailian_responses": frozenset({"response"}),
    "siliconflow": frozenset({"response", "embedding", "audio_transcription"}),
    "xiaomi_mimo": frozenset({"response", "audio_transcription"}),
}

_NAMES: dict[RuntimeKey, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "volcengine": "Volcengine Ark",
    "dashscope": "DashScope",
    "bailian_responses": "Bailian Responses",
    "siliconflow": "SiliconFlow",
    "xiaomi_mimo": "Xiaomi Mimo",
}


def create_vendor_runtime(
    key: RuntimeKey,
    options: ProviderRuntimeOptions,
    state_store: PluginStateStore | None,
    client: VendorClient,
) -> VendorRuntime:
    """创建一个配置代次内唯一的供应商 Runtime。"""

    adapter: HostAdapter
    match key:
        case "openai":
            if not isinstance(client, OpenAIClient):
                raise TypeError("openai Runtime 需要 OpenAIClient")
            adapter = OpenAIHostAdapter(options=options, client=client)
        case "anthropic":
            if not isinstance(client, AnthropicClient):
                raise TypeError("anthropic Runtime 需要 AnthropicClient")
            adapter = AnthropicHostAdapter(options=options, client=client)
        case "volcengine":
            if not isinstance(client, ArkClient):
                raise TypeError("volcengine Runtime 需要 ArkClient")
            adapter = ArkHostAdapter(options=options, client=client, state_store=state_store)
        case "dashscope":
            if not isinstance(client, DashScopeClient):
                raise TypeError("dashscope Runtime 需要 DashScopeClient")
            adapter = DashScopeHostAdapter(options=options, client=client)
        case "bailian_responses":
            if not isinstance(client, DashScopeClient):
                raise TypeError("bailian_responses Runtime 需要 DashScopeClient")
            adapter = BailianResponsesHostAdapter(options=options, client=client)
        case "siliconflow":
            if not isinstance(client, SiliconFlowClient):
                raise TypeError("siliconflow Runtime 需要 SiliconFlowClient")
            adapter = SiliconFlowHostAdapter(options=options, client=client)
        case "xiaomi_mimo":
            if state_store is None:
                raise RuntimeError("Xiaomi Mimo Runtime 需要插件持久化存储")
            if not isinstance(client, MimoClient):
                raise TypeError("xiaomi_mimo Runtime 需要 MimoClient")
            adapter = MimoHostAdapter(options=options, client=client, state_store=state_store)
        case _:
            raise ValueError(f"未知供应商 Runtime: {key}")
    ingress = LLMProviderIngress(
        adapter=adapter,
        capabilities=_CAPABILITIES[key],
        provider_name=_NAMES[key],
    )
    return VendorRuntime(
        client=client,
        host_adapter=adapter,
        ingress=ingress,
    )


def create_vendor_client(key: VendorClientKey) -> VendorClient:
    """创建只拥有协议资源和共享连接池的供应商 Client。"""

    match key:
        case "openai":
            return OpenAIClient()
        case "anthropic":
            return AnthropicClient()
        case "volcengine":
            return ArkClient()
        case "dashscope":
            return DashScopeClient()
        case "siliconflow":
            return SiliconFlowClient()
        case "xiaomi_mimo":
            return MimoClient()
        case _:
            raise ValueError(f"未知供应商 Client: {key}")
