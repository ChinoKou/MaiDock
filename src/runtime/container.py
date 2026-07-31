from collections.abc import Callable
from dataclasses import dataclass
import asyncio

from ..clients.ark import ArkClient
from ..clients.dashscope import DashScopeClient
from ..core.common import ProviderRuntimeOptions
from ..core.state_store import PluginStateStore
from .contracts import HostAdapter, RuntimeKey, VendorClient, VendorClientKey
from .ingress import LLMProviderIngress

# Host Runtime key -> Vendor Client key：bailian_responses 与 dashscope 共用
# 同一个 DashScopeClient 连接池，只关闭一次。
CLIENT_KEY_BY_RUNTIME: dict[RuntimeKey, VendorClientKey] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "volcengine": "volcengine",
    "dashscope": "dashscope",
    "bailian_responses": "dashscope",
    "siliconflow": "siliconflow",
    "xiaomi_mimo": "xiaomi_mimo",
}


@dataclass(frozen=True, slots=True)
class VendorRuntime:
    """一个供应商在同一配置代次内共享的运行时。"""

    client: VendorClient
    host_adapter: HostAdapter
    ingress: LLMProviderIngress


type RuntimeFactory = Callable[
    [RuntimeKey, ProviderRuntimeOptions, PluginStateStore | None, VendorClient], VendorRuntime
]
type ClientFactory = Callable[[VendorClientKey], VendorClient]


class VendorClientContainer:
    """插件生命周期内按供应商懒加载并独占共享连接池。"""

    def __init__(self, *, factory: ClientFactory) -> None:
        self._factory = factory
        self._clients: dict[VendorClientKey, VendorClient] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, key: VendorClientKey) -> VendorClient:
        async with self._lock:
            if self._closed:
                raise RuntimeError("VendorClient Container 已关闭")
            client = self._clients.get(key)
            if client is None:
                client = self._factory(key)
                self._clients[key] = client
            return client

    async def get_dashscope(self) -> DashScopeClient:
        client = await self.get("dashscope")
        if not isinstance(client, DashScopeClient):
            raise TypeError("dashscope Client factory 返回了错误类型")
        return client

    async def get_ark(self) -> ArkClient:
        # 公共媒体通路与 Host 通路共用同一个 "volcengine" Client——这正是架构里
        # 两条上层通路唯一的交汇点：共享供应商 Client，不共享彼此的 Adapter。
        client = await self.get("volcengine")
        if not isinstance(client, ArkClient):
            raise TypeError("volcengine Client factory 返回了错误类型")
        return client

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            clients = tuple(self._clients.values())
            self._clients.clear()
        await asyncio.gather(*(client.aclose() for client in clients))


class RuntimeContainer:
    """按供应商懒加载 Host Runtime，并统一切换配置代次。"""

    def __init__(
        self,
        *,
        options: ProviderRuntimeOptions,
        state_store: PluginStateStore | None,
        factory: RuntimeFactory,
        clients: VendorClientContainer,
    ) -> None:
        self.options = options
        self.state_store = state_store
        self._factory = factory
        self._clients = clients
        self._runtimes: dict[RuntimeKey, VendorRuntime] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, key: RuntimeKey) -> VendorRuntime:
        async with self._lock:
            if self._closed:
                raise RuntimeError("VendorRuntime Container 已关闭")
            runtime = self._runtimes.get(key)
            if runtime is None:
                client_key = CLIENT_KEY_BY_RUNTIME[key]
                client = await self._clients.get(client_key)
                runtime = self._factory(key, self.options, self.state_store, client)
                self._runtimes[key] = runtime
            return runtime

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._runtimes.clear()
