from hashlib import sha256
import json

from ....clients.common import HttpConnection, RetryPolicy
from ....clients.dashscope import DashScopeConnection, DashScopePaths
from ....runtime import VendorClientContainer
from ....version import DEFAULT_USER_AGENT
from ...catalog import PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS, PublicApiWebUiField, PublicProfileBinding
from ...config import PublicApiConfig, parameter_entries_to_object
from ...domain import PublicProviderDriver
from .driver import DashScopePublicDriver
from .registry import DashScopeMediaProfile, DashScopeProtocolRoute


class DashScopePublicContribution:
    provider_key = "dashscope"
    config_path = "public_api.dashscope"
    title_key = "ui.public_api.dashscope.title"
    icon = "image-play"
    order = 30

    def is_configured(self, config: PublicApiConfig) -> bool:
        return bool(config.dashscope.profiles)

    def build_webui_fields(self) -> tuple[PublicApiWebUiField, ...]:
        return _PROFILE_FIELDS

    async def build_driver(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> PublicProviderDriver:
        client = await clients.get_dashscope()
        profiles, _fingerprints = _build_profiles(config)
        return DashScopePublicDriver(client=client, profiles=profiles)

    async def build_profiles(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> tuple[PublicProfileBinding, ...]:
        driver = await self.build_driver(config, clients)
        profiles, fingerprints = _build_profiles(config)
        return tuple(
            PublicProfileBinding(
                name=profile.name,
                provider_key=self.provider_key,
                driver_key=driver.driver_key,
                credential_fingerprint=fingerprints[profile.name],
                driver=driver,
            )
            for profile in profiles
        )


def _build_profiles(
    config: PublicApiConfig,
) -> tuple[tuple[DashScopeMediaProfile, ...], dict[str, str]]:
    profiles: list[DashScopeMediaProfile] = []
    fingerprints: dict[str, str] = {}
    for source in config.dashscope.profiles:
        headers = [
            ("Authorization", f"Bearer {source.api_key}"),
            ("User-Agent", DEFAULT_USER_AGENT),
        ]
        if source.workspace_id:
            headers.append(("X-DashScope-WorkSpace", source.workspace_id))
        connection = DashScopeConnection(
            http=HttpConnection(
                base_url=source.base_url,
                default_headers=tuple(headers),
                request_timeout=source.request_timeout_seconds,
                connect_timeout=source.connect_timeout_seconds,
            ),
            retry=RetryPolicy(max_retries=0, uncertain_on_timeout=True),
            safe_retry=RetryPolicy(
                max_retries=source.safe_max_retries,
                retry_interval=source.retry_interval_seconds,
            ),
            paths=DashScopePaths(
                text_generation="services/aigc/text-generation/generation",
                multimodal_generation="services/aigc/multimodal-generation/generation",
                embeddings="services/embeddings/text-embedding/text-embedding",
                image_generation="services/aigc/image-generation/generation",
                text2image_synthesis="services/aigc/text2image/image-synthesis",
                image2image_synthesis="services/aigc/image2image/image-synthesis",
                video_generation="services/aigc/video-generation/video-synthesis",
            ),
        )
        profiles.append(
            DashScopeMediaProfile(
                name=source.name,
                connection=connection,
                default_image_model=source.default_image_model or None,
                default_video_model=source.default_video_model or None,
                image_default_parameters=parameter_entries_to_object(source.image_default_parameters),
                image_override_parameters=parameter_entries_to_object(source.image_override_parameters),
                video_default_parameters=parameter_entries_to_object(source.video_default_parameters),
                video_override_parameters=parameter_entries_to_object(source.video_override_parameters),
                protocol_routes=tuple(
                    DashScopeProtocolRoute(
                        capability=route.capability,
                        model=route.model,
                        mode=route.mode,
                        protocol_family=route.protocol_family,
                    )
                    for route in source.protocol_routes
                ),
            )
        )
        fingerprints[source.name] = _credential_fingerprint(
            name=source.name,
            api_key=source.api_key,
            base_url=source.base_url,
            workspace_id=source.workspace_id,
        )
    return tuple(profiles), fingerprints


def _credential_fingerprint(*, name: str, api_key: str, base_url: str, workspace_id: str) -> str:
    canonical = json.dumps(
        {
            "api_key": api_key,
            "base_url": base_url,
            "name": name,
            "provider": "dashscope",
            "workspace_id": workspace_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


_ROUTE_FIELDS = (
    PublicApiWebUiField(
        "capability",
        "select",
        "ui.public_api.field.route_capability",
        "image_generation",
        "select",
        0,
        choices=("image_generation", "video_generation"),
    ),
    PublicApiWebUiField("model", "string", "ui.public_api.field.route_model", "", "text", 1),
    PublicApiWebUiField("mode", "string", "ui.public_api.field.route_mode", "", "text", 2),
    PublicApiWebUiField(
        "protocol_family",
        "select",
        "ui.public_api.field.route_family",
        "dashscope_multimodal_generation",
        "select",
        3,
        choices=(
            "dashscope_multimodal_generation",
            "dashscope_image_generation",
            "dashscope_text2image_synthesis",
            "dashscope_image2image_synthesis",
            "dashscope_video_generation",
        ),
    ),
)

_PROFILE_FIELDS = (
    PublicApiWebUiField("name", "string", "ui.public_api.field.profile_name", "", "text", 0),
    PublicApiWebUiField(
        "api_key",
        "string",
        "ui.public_api.field.api_key",
        "",
        "text",
        1,
        hint_key="ui.public_api.field.api_key.hint",
    ),
    PublicApiWebUiField(
        "base_url",
        "string",
        "ui.public_api.field.base_url",
        "https://dashscope.aliyuncs.com/api/v1",
        "text",
        2,
    ),
    PublicApiWebUiField("workspace_id", "string", "ui.public_api.field.workspace_id", "", "text", 3),
    PublicApiWebUiField(
        "default_image_model",
        "string",
        "ui.public_api.field.default_image_model",
        "",
        "text",
        4,
    ),
    PublicApiWebUiField(
        "default_video_model",
        "string",
        "ui.public_api.field.default_video_model",
        "",
        "text",
        5,
    ),
    PublicApiWebUiField(
        "connect_timeout_seconds",
        "number",
        "ui.public_api.field.connect_timeout",
        10.0,
        "number",
        6,
        minimum=0.1,
        maximum=120,
        step=0.1,
    ),
    PublicApiWebUiField(
        "request_timeout_seconds",
        "number",
        "ui.public_api.field.request_timeout",
        1800.0,
        "number",
        7,
        minimum=1,
        maximum=82800,
    ),
    PublicApiWebUiField(
        "safe_max_retries",
        "integer",
        "ui.public_api.field.safe_retries",
        3,
        "number",
        8,
        minimum=0,
        maximum=10,
    ),
    PublicApiWebUiField(
        "retry_interval_seconds",
        "number",
        "ui.public_api.field.retry_interval",
        1.0,
        "number",
        9,
        minimum=0,
        maximum=30,
        step=0.1,
    ),
    PublicApiWebUiField(
        "image_default_parameters",
        "array",
        "ui.public_api.field.image_defaults",
        [],
        "list",
        10,
        hint_key="ui.public_api.field.parameters.hint",
        item_type="object",
        item_fields=PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS,
    ),
    PublicApiWebUiField(
        "image_override_parameters",
        "array",
        "ui.public_api.field.image_overrides",
        [],
        "list",
        11,
        hint_key="ui.public_api.field.parameters.hint",
        item_type="object",
        item_fields=PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS,
    ),
    PublicApiWebUiField(
        "video_default_parameters",
        "array",
        "ui.public_api.field.video_defaults",
        [],
        "list",
        12,
        hint_key="ui.public_api.field.parameters.hint",
        item_type="object",
        item_fields=PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS,
    ),
    PublicApiWebUiField(
        "video_override_parameters",
        "array",
        "ui.public_api.field.video_overrides",
        [],
        "list",
        13,
        hint_key="ui.public_api.field.parameters.hint",
        item_type="object",
        item_fields=PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS,
    ),
    PublicApiWebUiField(
        "protocol_routes",
        "array",
        "ui.public_api.field.protocol_routes",
        [],
        "list",
        14,
        item_type="object",
        item_fields=_ROUTE_FIELDS,
    ),
)


DASHSCOPE_PUBLIC_CONTRIBUTION = DashScopePublicContribution()
