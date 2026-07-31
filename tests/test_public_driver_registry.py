from pathlib import Path

import pytest

from src.public_api.catalog import (
    PublicApiWebUiField,
    PublicDriverRegistry,
    PublicProfileBinding,
    build_public_driver_registry,
)
from src.public_api.config import PublicApiConfig
from src.public_api.domain import (
    Completed,
    MediaCapability,
    MediaOutput,
    MediaRequest,
    MaterializedArtifact,
    ModelCapability,
    PreparedMediaOperation,
    PublicProviderDriver,
    VersionedOpaqueHandle,
)
from src.runtime import VendorClientContainer


class FakePublicDriver:
    def __init__(self, key: str) -> None:
        self._key = key

    @property
    def driver_key(self) -> str:
        return self._key

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (
            ModelCapability(
                model=f"{self._key}.image",
                capability=MediaCapability.IMAGE_GENERATION,
                modes=("text_to_image",),
                protocol_families=("fake",),
                max_outputs=1,
            ),
        )

    def prepare(self, profile_name: str, request: MediaRequest) -> PreparedMediaOperation:
        return PreparedMediaOperation(
            driver_key=self._key,
            payload_version=1,
            profile_name=profile_name,
            capability=request.capability,
            operation_type="fake",
            payload={"model": f"{self._key}.image"},
        )

    async def submit(self, operation: PreparedMediaOperation) -> Completed:
        return Completed(outputs=(MediaOutput(kind="text", text=operation.profile_name),))

    async def poll(self, handle: VersionedOpaqueHandle) -> Completed:
        del handle
        return Completed(outputs=())

    async def cancel(self, handle: VersionedOpaqueHandle) -> Completed:
        del handle
        return Completed(outputs=())

    async def upload_file(self, profile_name: str, *, model: str, path: Path, media_type: str) -> str:
        del profile_name, model, path, media_type
        return "oss://fake/input"

    async def materialize(
        self,
        profile_name: str,
        *,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> MaterializedArtifact:
        del profile_name, url, max_bytes
        destination.write_bytes(b"artifact")
        return MaterializedArtifact(
            path=destination,
            size=8,
            sha256="placeholder",
            media_type="application/octet-stream",
        )


class FakeClient:
    async def aclose(self) -> None:
        pass


class FakeContribution:
    def __init__(self, provider_key: str, driver: FakePublicDriver) -> None:
        self.provider_key = provider_key
        self.driver = driver
        self.config_path = f"public_api.{provider_key}"
        self.title_key = f"ui.public_api.{provider_key}.title"
        self.icon = "test"
        self.order = 100
        self.build_driver_calls = 0

    def is_configured(self, config: PublicApiConfig) -> bool:
        del config
        return True

    def build_webui_fields(self) -> tuple[PublicApiWebUiField, ...]:
        return ()

    async def build_driver(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> PublicProviderDriver:
        del config, clients
        self.build_driver_calls += 1
        return self.driver

    async def build_profiles(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> tuple[PublicProfileBinding, ...]:
        driver = await self.build_driver(config, clients)
        return (
            PublicProfileBinding(
                name=self.provider_key,
                provider_key=self.provider_key,
                driver_key=driver.driver_key,
                credential_fingerprint=f"{self.provider_key}-fingerprint",
                driver=driver,
            ),
        )


def test_registry_accepts_a_second_driver_without_public_application_changes() -> None:
    first = FakePublicDriver("first.v1")
    second = FakePublicDriver("second.v1")
    registry = PublicDriverRegistry(
        profiles=(
            PublicProfileBinding("first", "first", first.driver_key, "fp-1", first),
            PublicProfileBinding("second", "second", second.driver_key, "fp-2", second),
        ),
        default_image_profile="second",
    )

    binding = registry.resolve(MediaCapability.IMAGE_GENERATION, None)
    operation = binding.driver.prepare(
        binding.name,
        MediaRequest(capability=MediaCapability.IMAGE_GENERATION, mode="text_to_image"),
    )

    assert binding.driver is second
    assert operation.driver_key == "second.v1"
    assert registry.driver("first.v1") is first
    assert {model.model for model in registry.capabilities().models} == {
        "first.v1.image",
        "second.v1.image",
    }


@pytest.mark.asyncio
async def test_second_contribution_uses_the_same_registry_extension_point() -> None:
    first = FakePublicDriver("first.v1")
    second = FakePublicDriver("second.v1")
    first_contribution = FakeContribution("first", first)
    second_contribution = FakeContribution("second", second)
    client_container = VendorClientContainer(factory=lambda _key: FakeClient())
    registry = await build_public_driver_registry(
        PublicApiConfig(default_image_profile="second"),
        client_container,
        (first_contribution, second_contribution),
    )
    try:
        assert registry.resolve(MediaCapability.IMAGE_GENERATION, None).driver is second
        assert registry.profile("first") is not None
        assert registry.driver("second.v1") is second
        assert first_contribution.build_driver_calls == 1
        assert second_contribution.build_driver_calls == 1
    finally:
        await client_container.aclose()
