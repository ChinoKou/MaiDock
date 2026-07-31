import json
from hashlib import sha256

from ....clients.ark import (
    ARK_CONTENT_GENERATION_TASKS_PATH,
    ARK_IMAGES_GENERATIONS_PATH,
    ArkConnection,
)
from ....clients.common import HttpConnection, RetryPolicy
from ....runtime import VendorClientContainer
from ....version import DEFAULT_USER_AGENT
from ...catalog import PUBLIC_PARAMETER_ENTRY_WEBUI_FIELDS, PublicApiWebUiField, PublicProfileBinding
from ...config import PublicApiConfig, parameter_entries_to_object
from ...domain import PublicProviderDriver
from .driver import ArkPublicDriver
from .registry import ArkMediaProfile, ArkProtocolRoute


class ArkPublicContribution:
    provider_key = "volcengine_ark"
    config_path = "public_api.volcengine_ark"
    title_key = "ui.public_api.volcengine_ark.title"
    icon = "clapperboard"
    order = 31

    def is_configured(self, config: PublicApiConfig) -> bool:
        return bool(config.volcengine_ark.profiles)

    def build_webui_fields(self) -> tuple[PublicApiWebUiField, ...]:
        return _PROFILE_FIELDS

    async def build_driver(
        self,
        config: PublicApiConfig,
        clients: VendorClientContainer,
    ) -> PublicProviderDriver:
        client = await clients.get_ark()
        profiles, _fingerprints = _build_profiles(config)
        return ArkPublicDriver(client=client, profiles=profiles)

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


def _build_profiles(config: PublicApiConfig) -> tuple[tuple[ArkMediaProfile, ...], dict[str, str]]:
    profiles: list[ArkMediaProfile] = []
    fingerprints: dict[str, str] = {}
    for source in config.volcengine_ark.profiles:
        connection = ArkConnection(
            http=HttpConnection(
                base_url=source.base_url,
                default_headers=(
                    ("Authorization", f"Bearer {source.api_key}"),
                    ("User-Agent", DEFAULT_USER_AGENT),
                ),
                request_timeout=source.request_timeout_seconds,
                connect_timeout=source.connect_timeout_seconds,
            ),
            # 提交零重试、幂等操作可重试，与 DashScope 一致。
            retry=RetryPolicy(max_retries=0, uncertain_on_timeout=True),
            safe_retry=RetryPolicy(
                max_retries=source.safe_max_retries,
                retry_interval=source.retry_interval_seconds,
            ),
            # 公共媒体通路不碰这四个文本资源，但 ArkConnection 是同一个不可变快照，
            # 必须给出完整路径；取值与 Host 通路一致。
            responses_path="responses",
            embeddings_path="embeddings/multimodal",
            audio_transcriptions_path="responses",
            tokenization_path="tokenization",
            images_generations_path=ARK_IMAGES_GENERATIONS_PATH,
            content_generation_tasks_path=ARK_CONTENT_GENERATION_TASKS_PATH,
        )
        profiles.append(
            ArkMediaProfile(
                name=source.name,
                connection=connection,
                default_image_model=source.default_image_model or None,
                default_video_model=source.default_video_model or None,
                image_default_parameters=parameter_entries_to_object(source.image_default_parameters),
                image_override_parameters=parameter_entries_to_object(source.image_override_parameters),
                video_default_parameters=parameter_entries_to_object(source.video_default_parameters),
                video_override_parameters=parameter_entries_to_object(source.video_override_parameters),
                protocol_routes=tuple(
                    ArkProtocolRoute(
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
        )
    return tuple(profiles), fingerprints


def _credential_fingerprint(*, name: str, api_key: str, base_url: str) -> str:
    """凭据指纹用于恢复时检测 profile 是否被改过，因此必须是稳定的 canonical 形式。"""

    canonical = json.dumps(
        {
            "api_key": api_key,
            "base_url": base_url,
            "name": name,
            "provider": "volcengine_ark",
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
        "ark_images_generations",
        "select",
        3,
        choices=("ark_images_generations", "ark_content_generation_tasks"),
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
        "https://ark.cn-beijing.volces.com/api/v3",
        "text",
        2,
    ),
    PublicApiWebUiField(
        "default_image_model",
        "string",
        "ui.public_api.field.default_image_model",
        "",
        "text",
        3,
    ),
    PublicApiWebUiField(
        "default_video_model",
        "string",
        "ui.public_api.field.default_video_model",
        "",
        "text",
        4,
    ),
    PublicApiWebUiField(
        "connect_timeout_seconds",
        "number",
        "ui.public_api.field.connect_timeout",
        10.0,
        "number",
        5,
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
        6,
        minimum=1,
        maximum=82800,
    ),
    PublicApiWebUiField(
        "safe_max_retries",
        "integer",
        "ui.public_api.field.safe_retries",
        3,
        "number",
        7,
        minimum=0,
        maximum=10,
    ),
    PublicApiWebUiField(
        "retry_interval_seconds",
        "number",
        "ui.public_api.field.retry_interval",
        1.0,
        "number",
        8,
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
        9,
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
        10,
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
        11,
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
        12,
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
        13,
        item_type="object",
        item_fields=_ROUTE_FIELDS,
    ),
)


ARK_PUBLIC_CONTRIBUTION = ArkPublicContribution()
