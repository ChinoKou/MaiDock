from maibot_sdk import LLMProviderBase

from ..host_adapters.common.rpc import HostRpcRequest, HostRpcResponse
from ..i18n import translate
from .contracts import HostAdapter, ProviderCapability


class LLMProviderIngress(LLMProviderBase):
    """唯一接触 SDK Provider 基类的 Host 入口。"""

    def __init__(
        self,
        *,
        adapter: HostAdapter,
        capabilities: frozenset[ProviderCapability],
        provider_name: str,
    ) -> None:
        self.adapter = adapter
        self.capabilities = capabilities
        self.provider_name = provider_name

    def _require(self, capability: ProviderCapability) -> None:
        if capability not in self.capabilities:
            raise NotImplementedError(
                translate(
                    "runtime.error.capability_unsupported",
                    provider=self.provider_name,
                    capability=capability,
                )
            )

    async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
        self._require("response")
        return await self.adapter.get_response(request)

    async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
        self._require("embedding")
        return await self.adapter.get_embedding(request)

    async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
        self._require("audio_transcription")
        return await self.adapter.get_audio_transcriptions(request)
