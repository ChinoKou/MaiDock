from pathlib import Path

from ..runtime import VendorClientContainer
from .catalog import build_public_driver_registry
from .config import PublicApiConfig
from .api.facade import PublicApiFacade
from .application.job_engine import MediaJobEngine
from .application.service import PublicApiService
from .providers import PUBLIC_PROVIDER_CONTRIBUTIONS


class PublicApiRuntime:
    """Public API 配置代次、持久化引擎和公开门面的生命周期容器。"""

    def __init__(
        self,
        *,
        data_dir: Path,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> None:
        self._data_dir = data_dir
        self._config = config
        self._clients = clients
        self.engine: MediaJobEngine | None = None
        self.facade: PublicApiFacade | None = None

    async def start(self) -> None:
        registry = await build_public_driver_registry(
            self._config,
            self._clients,
            PUBLIC_PROVIDER_CONTRIBUTIONS,
        )
        engine = MediaJobEngine(data_dir=self._data_dir, config=self._config, registry=registry)
        await engine.start()
        self.engine = engine
        self.facade = PublicApiFacade(PublicApiService(engine))

    async def update_config(self, config: PublicApiConfig) -> None:
        engine = self.require_engine()
        registry = await build_public_driver_registry(
            config,
            self._clients,
            PUBLIC_PROVIDER_CONTRIBUTIONS,
        )
        self._config = config
        engine.update_config(config, registry)

    async def stop(self) -> None:
        engine = self.engine
        self.engine = None
        self.facade = None
        if engine is not None:
            await engine.stop()

    def require_engine(self) -> MediaJobEngine:
        if self.engine is None:
            raise RuntimeError("Public API Runtime 尚未启动")
        return self.engine

    def require_facade(self) -> PublicApiFacade:
        if self.facade is None:
            raise RuntimeError("Public API Runtime 尚未启动")
        return self.facade
