from dataclasses import dataclass
from typing import Protocol

from ..runtime import VendorClientContainer
from .config import PublicApiConfig
from .domain import MediaCapability, ModelCapability, PublicJsonValue, PublicProviderDriver


@dataclass(frozen=True, slots=True)
class PublicProfileBinding:
    name: str
    provider_key: str
    driver_key: str
    credential_fingerprint: str
    driver: PublicProviderDriver


@dataclass(frozen=True, slots=True)
class PublicApiWebUiField:
    name: str
    field_type: str
    label_key: str
    default: PublicJsonValue
    ui_type: str
    order: int
    hint_key: str = ""
    choices: tuple[str, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: float = 1.0
    rows: int = 3
    item_type: str | None = None
    item_fields: tuple["PublicApiWebUiField", ...] = ()


PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS = (
    PublicApiWebUiField(
        "name",
        "string",
        "ui.public_api.field.parameter_name",
        "",
        "text",
        0,
    ),
    PublicApiWebUiField(
        "value_type",
        "select",
        "ui.public_api.field.parameter_type",
        "string",
        "select",
        1,
        choices=("string", "integer", "number", "boolean", "json", "null"),
    ),
    PublicApiWebUiField(
        "value",
        "string",
        "ui.public_api.field.parameter_value",
        "",
        "text",
        2,
    ),
)


class PublicProviderContribution(Protocol):
    @property
    def provider_key(self) -> str: ...

    @property
    def config_path(self) -> str: ...

    @property
    def title_key(self) -> str: ...

    @property
    def icon(self) -> str: ...

    @property
    def order(self) -> int: ...

    def is_configured(self, config: PublicApiConfig) -> bool: ...

    def build_webui_fields(self) -> tuple[PublicApiWebUiField, ...]: ...

    async def build_profiles(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> tuple[PublicProfileBinding, ...]: ...

    async def build_driver(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> PublicProviderDriver: ...


@dataclass(frozen=True, slots=True)
class PublicCapabilities:
    models: tuple[ModelCapability, ...]
    profiles: tuple[PublicProfileBinding, ...]
    default_image_profile: str | None
    default_video_profile: str | None


class PublicDriverRegistry:
    """供应商无关的 Profile 和 Driver 注册表。"""

    def __init__(
        self,
        *,
        profiles: tuple[PublicProfileBinding, ...],
        default_image_profile: str = "",
        default_video_profile: str = "",
    ) -> None:
        self.default_image_profile = default_image_profile
        self.default_video_profile = default_video_profile
        self._profiles: dict[str, PublicProfileBinding] = {}
        self._drivers: dict[str, PublicProviderDriver] = {}
        for binding in profiles:
            if binding.name in self._profiles:
                raise ValueError(f"Public API Profile 重复: {binding.name}")
            self._profiles[binding.name] = binding
            existing = self._drivers.get(binding.driver_key)
            if existing is not None and existing is not binding.driver:
                raise ValueError(f"Public API Driver key 冲突: {binding.driver_key}")
            self._drivers[binding.driver_key] = binding.driver

    def resolve(
        self,
        capability: MediaCapability,
        requested_name: str | None,
    ) -> PublicProfileBinding:
        default_name = (
            self.default_image_profile if capability is MediaCapability.IMAGE_GENERATION else self.default_video_profile
        )
        profile_name = requested_name or default_name
        if not profile_name:
            raise KeyError("PROFILE_REQUIRED")
        binding = self._profiles.get(profile_name)
        if binding is None:
            raise KeyError("PROFILE_NOT_FOUND")
        return binding

    def profile(self, name: str) -> PublicProfileBinding | None:
        return self._profiles.get(name)

    def driver(self, driver_key: str) -> PublicProviderDriver | None:
        return self._drivers.get(driver_key)

    def capabilities(self) -> PublicCapabilities:
        unique_drivers: list[PublicProviderDriver] = []
        for binding in self._profiles.values():
            if all(binding.driver is not current for current in unique_drivers):
                unique_drivers.append(binding.driver)
        models = tuple(model for driver in unique_drivers for model in driver.capabilities())
        return PublicCapabilities(
            models=models,
            profiles=tuple(self._profiles.values()),
            default_image_profile=self.default_image_profile or None,
            default_video_profile=self.default_video_profile or None,
        )


async def build_public_driver_registry(
    config: PublicApiConfig,
    clients: VendorClientContainer,
    contributions: tuple[PublicProviderContribution, ...],
) -> PublicDriverRegistry:
    bindings: list[PublicProfileBinding] = []
    for contribution in contributions:
        if contribution.is_configured(config):
            bindings.extend(await contribution.build_profiles(config, clients))
    return PublicDriverRegistry(
        profiles=tuple(bindings),
        default_image_profile=config.default_image_profile,
        default_video_profile=config.default_video_profile,
    )
