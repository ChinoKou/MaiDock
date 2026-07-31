from pathlib import Path
from inspect import Parameter, signature

import asyncio

import httpx
import pytest

from src import plugin as plugin_module
from src.clients.common import ClientClosedError, HttpConnection, RetryPolicy
from src.clients.dashscope import (
    DashScopeClient,
    DashScopeConnection,
    DashScopePaths,
    DashScopeResponsesConnection,
)
from src.clients.families import JsonResourceRequest
from src.clients.openai import OpenAIClient, OpenAIConnection
from src.core.common import ProviderRuntimeOptions
from src.core.state_store import PluginStateStore
from src.host_adapters.common.rpc import HostRpcRequest, HostRpcResponse
from src.runtime import (
    CLIENT_KEY_BY_RUNTIME,
    LLMProviderIngress,
    RuntimeKey,
    RuntimeContainer,
    VendorClient,
    VendorClientKey,
    VendorClientContainer,
    VendorRuntime,
)

from .plugin_test_support import create_plugin


def _connection(token: str) -> OpenAIConnection:
    return OpenAIConnection(
        http=HttpConnection(
            base_url="https://api.example/v1",
            default_headers=(("Authorization", f"Bearer {token}"),),
        ),
        retry=RetryPolicy(),
        responses_path="responses",
        embeddings_path="embeddings",
        audio_transcriptions_path="audio/transcriptions",
    )


@pytest.mark.asyncio
async def test_bailian_responses_shared_dashscope_client_pool() -> None:
    """bailian_responses 与 dashscope 共用一个 DashScopeClient，只创建一次。"""

    created: list[VendorClientKey] = []

    class CountingDashScopeClient(DashScopeClient):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1
            await super().aclose()

    def factory(raw_key: VendorClientKey) -> VendorClient:
        created.append(raw_key)
        return CountingDashScopeClient()

    class EmptyAdapter:
        async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
            del request
            return {}

        async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
            return await self.get_response(request)

        async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
            return await self.get_response(request)

    def runtime_factory(
        key: RuntimeKey,
        options: ProviderRuntimeOptions,
        state_store: PluginStateStore | None,
        client: VendorClient,
    ) -> VendorRuntime:
        del key, options, state_store
        adapter = EmptyAdapter()
        return VendorRuntime(
            client=client,
            host_adapter=adapter,
            ingress=LLMProviderIngress(
                adapter=adapter,
                capabilities=frozenset({"response"}),
                provider_name="test",
            ),
        )

    clients = VendorClientContainer(factory=factory)
    runtime_clients = RuntimeContainer(
        options=ProviderRuntimeOptions(),
        state_store=None,
        factory=runtime_factory,
        clients=clients,
    )

    dashscope_client = await runtime_clients.get("dashscope")
    bailian_client = await runtime_clients.get("bailian_responses")

    assert CLIENT_KEY_BY_RUNTIME["bailian_responses"] == "dashscope"
    # 只创建了一个 DashScopeClient，两个 Runtime 共享同一实例。
    assert created == ["dashscope"]
    assert dashscope_client.client is bailian_client.client
    assert isinstance(dashscope_client.client, CountingDashScopeClient)
    await runtime_clients.aclose()
    await clients.aclose()
    assert dashscope_client.client.close_count == 1


def test_dashscope_connection_contracts_keep_native_and_responses_paths_separate() -> None:
    required_native_paths = {
        "text_generation",
        "multimodal_generation",
        "embeddings",
        "image_generation",
        "text2image_synthesis",
        "image2image_synthesis",
        "video_generation",
    }
    path_parameters = signature(DashScopePaths).parameters

    assert all(path_parameters[name].default is Parameter.empty for name in required_native_paths)
    assert set(DashScopeResponsesConnection.__dataclass_fields__) == {"http", "retry", "responses_path"}
    assert "responses_path" not in DashScopeConnection.__dataclass_fields__
    assert "paths" not in DashScopeResponsesConnection.__dataclass_fields__


def test_client_key_mapping_targets_exist_in_factory() -> None:
    """映射表的目标 key 必须能通过 create_vendor_client 创建。"""

    from src.runtime import create_vendor_client

    for target in set(CLIENT_KEY_BY_RUNTIME.values()):
        client = create_vendor_client(target)
        assert client is not None
        assert hasattr(client, "aclose")


@pytest.mark.asyncio
async def test_session_exit_does_not_close_shared_pool() -> None:
    client = OpenAIClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    async with client.session(_connection("one")):
        pass

    assert not client.closed
    await client.aclose()
    assert client.closed


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_credentials_isolated() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        await asyncio.sleep(0)
        return httpx.Response(200, json={"id": request.headers["authorization"]})

    client = OpenAIClient(transport=httpx.MockTransport(handler))

    async def call(token: str) -> str:
        async with client.session(_connection(token)) as session:
            payload = await session.responses.create(
                JsonResourceRequest(body={"model": "test"}),
                retry=session.retry,
            )
        return str(payload["id"])

    results = await asyncio.gather(call("first"), call("second"))

    assert set(results) == {"Bearer first", "Bearer second"}
    assert set(seen) == {"Bearer first", "Bearer second"}
    await client.aclose()


@pytest.mark.asyncio
async def test_close_stops_new_leases_and_waits_for_active_session() -> None:
    client = OpenAIClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_lease() -> None:
        async with client.session(_connection("one")):
            entered.set()
            await release.wait()

    lease_task = asyncio.create_task(hold_lease())
    await entered.wait()
    close_task = asyncio.create_task(client.aclose())
    await asyncio.sleep(0)

    assert not close_task.done()
    release.set()
    await lease_task
    await close_task
    assert client.closed

    with pytest.raises(ClientClosedError):
        async with client.session(_connection("two")):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle_action", ["config_update", "unload"])
async def test_plugin_lifecycle_waits_for_inflight_client_lease(
    lifecycle_action: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    clients: list[OpenAIClient] = []
    generation = 0

    class LeaseAdapter:
        def __init__(self, client: OpenAIClient, runtime_generation: int) -> None:
            self.client = client
            self.generation = runtime_generation

        async def get_response(self, request: HostRpcRequest) -> HostRpcResponse:
            del request
            async with self.client.session(_connection(f"generation-{self.generation}")):
                if self.generation == 1:
                    started.set()
                    await release.wait()
            return {"generation": self.generation}

        async def get_embedding(self, request: HostRpcRequest) -> HostRpcResponse:
            return await self.get_response(request)

        async def get_audio_transcriptions(self, request: HostRpcRequest) -> HostRpcResponse:
            return await self.get_response(request)

    def factory(
        key: RuntimeKey,
        options: ProviderRuntimeOptions,
        state_store: PluginStateStore | None,
        raw_client: VendorClient,
    ) -> VendorRuntime:
        nonlocal generation
        del key, options, state_store
        generation += 1
        if not isinstance(raw_client, OpenAIClient):
            raise TypeError("生命周期测试需要 OpenAIClient")
        client = raw_client
        if client not in clients:
            clients.append(client)
        adapter = LeaseAdapter(client, generation)
        return VendorRuntime(
            client=client,
            host_adapter=adapter,
            ingress=LLMProviderIngress(
                adapter=adapter,
                capabilities=frozenset({"response", "embedding", "audio_transcription"}),
                provider_name="Lease Test",
            ),
        )

    monkeypatch.setattr(plugin_module, "create_vendor_runtime", factory)

    def client_factory(key: VendorClientKey) -> VendorClient:
        del key
        return OpenAIClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    monkeypatch.setattr(plugin_module, "create_vendor_client", client_factory)
    plugin = create_plugin(tmp_path)
    await plugin.on_load()
    request_task = asyncio.create_task(plugin.openai_responses_provider("response", {}))
    await started.wait()

    if lifecycle_action == "config_update":
        lifecycle_task = asyncio.create_task(plugin.on_config_update("self", {}, "2"))
    else:
        lifecycle_task = asyncio.create_task(plugin.on_unload())
    await asyncio.sleep(0)

    if lifecycle_action == "unload":
        assert not lifecycle_task.done()
    else:
        await lifecycle_task
    assert not clients[0].closed
    release.set()
    assert await request_task == {"generation": 1}
    if lifecycle_action == "unload":
        await lifecycle_task
        assert clients[0].closed

    if lifecycle_action == "config_update":
        assert await plugin.openai_responses_provider("response", {}) == {"generation": 2}
        assert len(clients) == 1
        await plugin.on_unload()
        assert clients[0].closed
