from typing import Literal, Protocol

from ..host_adapters.common.rpc import HostRpcRequest, HostRpcResponse

type ProviderCapability = Literal["response", "embedding", "audio_transcription"]
type RuntimeKey = Literal[
    "openai",
    "anthropic",
    "volcengine",
    "dashscope",
    "bailian_responses",
    "siliconflow",
    "xiaomi_mimo",
]
type VendorClientKey = Literal[
    "openai",
    "anthropic",
    "volcengine",
    "dashscope",
    "siliconflow",
    "xiaomi_mimo",
]


class VendorClient(Protocol):
    """Runtime 统一管理的共享供应商 Client。"""

    async def aclose(self) -> None: ...


class HostAdapter(Protocol):
    """SDK ingress 调用的 Host Adapter 合约。"""

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse: ...

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse: ...

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse: ...
