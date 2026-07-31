import pytest
from pydantic import ValidationError

from src.clients.ark import ArkClient
from src.clients.dashscope import DashScopeClient
from src.public_api.config import ArkPublicProfileConfig, PublicApiConfig
from src.public_api.providers import PUBLIC_API_CONFIG_CATALOG, PUBLIC_PROVIDER_CONTRIBUTIONS
from src.public_api.providers.volcengine_ark.contribution import (
    ARK_PUBLIC_CONTRIBUTION,
    _build_profiles,
    _credential_fingerprint,
)
from src.runtime import VendorClientContainer
from src.runtime.contracts import VendorClient


def _ark_config(*profiles: dict[str, object]) -> PublicApiConfig:
    return PublicApiConfig.model_validate({"volcengine_ark": {"profiles": list(profiles)}})


def test_ark_profile_defaults_to_official_base_url() -> None:
    profile = ArkPublicProfileConfig.model_validate({"name": "ark", "api_key": "sk-test"})

    assert profile.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert profile.protocol_routes == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://ark.cn-beijing.volces.com/api/v3",  # 非 HTTPS
        "https://ark.cn-beijing.volces.com/api/v1",  # 前缀不对
        "https://ark.cn-beijing.volces.com",  # 缺前缀
        "https://user:pw@ark.cn-beijing.volces.com/api/v3",  # 带凭据
        "https://ark.cn-beijing.volces.com/api/v3?x=1",  # 带 query
    ],
)
def test_ark_profile_rejects_malformed_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        ArkPublicProfileConfig.model_validate({"name": "ark", "api_key": "sk-test", "base_url": base_url})


def test_ark_profile_accepts_trailing_slash_and_normalizes() -> None:
    profile = ArkPublicProfileConfig.model_validate(
        {"name": "ark", "api_key": "sk", "base_url": "https://ark.example/api/v3/"}
    )

    assert profile.base_url == "https://ark.example/api/v3"


def test_profile_names_must_be_unique_across_providers() -> None:
    """调用方只按 profile 名寻址、不带供应商前缀，同名会让请求落到哪一家不确定。"""

    with pytest.raises(ValidationError, match="必须全局唯一"):
        PublicApiConfig.model_validate(
            {
                "dashscope": {"profiles": [{"name": "shared", "api_key": "sk-a"}]},
                "volcengine_ark": {"profiles": [{"name": "shared", "api_key": "sk-b"}]},
            }
        )


def test_distinct_profile_names_across_providers_are_allowed() -> None:
    config = PublicApiConfig.model_validate(
        {
            "dashscope": {"profiles": [{"name": "aliyun", "api_key": "sk-a"}]},
            "volcengine_ark": {"profiles": [{"name": "volc", "api_key": "sk-b"}]},
        }
    )

    assert [profile.name for profile in config.dashscope.profiles] == ["aliyun"]
    assert [profile.name for profile in config.volcengine_ark.profiles] == ["volc"]


def test_duplicate_routes_within_one_profile_are_rejected() -> None:
    with pytest.raises(ValidationError, match="重复路由"):
        ArkPublicProfileConfig.model_validate(
            {
                "name": "ark",
                "api_key": "sk",
                "protocol_routes": [
                    {
                        "capability": "video_generation",
                        "model": "ep-1",
                        "protocol_family": "ark_content_generation_tasks",
                    },
                    {
                        "capability": "video_generation",
                        "model": "ep-1",
                        "protocol_family": "ark_content_generation_tasks",
                    },
                ],
            }
        )


def test_build_profiles_carries_connection_and_paths() -> None:
    config = _ark_config(
        {
            "name": "ark",
            "api_key": "sk-test",
            "base_url": "https://ark.example/api/v3",
            "default_video_model": "doubao-seedance-2-0",
            "safe_max_retries": 2,
            "image_default_parameters": [
                {"name": "size", "value_type": "string", "value": "2K"},
                {"name": "watermark", "value_type": "boolean", "value": "false"},
            ],
            "video_override_parameters": [{"name": "duration", "value_type": "integer", "value": "5"}],
        }
    )

    profiles, fingerprints = _build_profiles(config)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.default_video_model == "doubao-seedance-2-0"
    assert profile.connection.images_generations_path == "images/generations"
    assert profile.connection.content_generation_tasks_path == "contents/generations/tasks"
    # 提交零重试、幂等操作可重试。
    assert profile.connection.retry.max_retries == 0
    assert profile.connection.retry.uncertain_on_timeout is True
    assert profile.connection.safe_retry.max_retries == 2
    assert ("Authorization", "Bearer sk-test") in profile.connection.http.default_headers
    assert profile.image_default_parameters == {"size": "2K", "watermark": False}
    assert profile.video_override_parameters == {"duration": 5}
    assert fingerprints["ark"]


def test_credential_fingerprint_is_stable_and_key_sensitive() -> None:
    first = _credential_fingerprint(name="ark", api_key="sk-a", base_url="https://ark.example/api/v3")
    same = _credential_fingerprint(name="ark", api_key="sk-a", base_url="https://ark.example/api/v3")
    other_key = _credential_fingerprint(name="ark", api_key="sk-b", base_url="https://ark.example/api/v3")
    other_url = _credential_fingerprint(name="ark", api_key="sk-a", base_url="https://ark2.example/api/v3")

    assert first == same
    assert first != other_key
    assert first != other_url


def test_contribution_is_registered_in_both_catalogs() -> None:
    assert ARK_PUBLIC_CONTRIBUTION in PUBLIC_API_CONFIG_CATALOG
    assert ARK_PUBLIC_CONTRIBUTION in PUBLIC_PROVIDER_CONTRIBUTIONS
    assert ARK_PUBLIC_CONTRIBUTION.provider_key == "volcengine_ark"
    assert ARK_PUBLIC_CONTRIBUTION.config_path == "public_api.volcengine_ark"


def test_is_configured_tracks_profile_presence() -> None:
    assert ARK_PUBLIC_CONTRIBUTION.is_configured(PublicApiConfig()) is False
    assert ARK_PUBLIC_CONTRIBUTION.is_configured(_ark_config({"name": "ark", "api_key": "sk"})) is True


@pytest.mark.asyncio
async def test_get_ark_shares_the_volcengine_client() -> None:
    """两条上层通路共享同一个供应商 Client，这是它们唯一的交汇点。"""

    created: list[str] = []

    def factory(key: str) -> VendorClient:
        created.append(key)
        return ArkClient()

    container = VendorClientContainer(factory=factory)
    try:
        first = await container.get_ark()
        second = await container.get("volcengine")

        assert first is second
        assert created == ["volcengine"]
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_get_ark_guards_against_wrong_client_type() -> None:
    def factory(key: str) -> VendorClient:
        del key
        return DashScopeClient()

    container = VendorClientContainer(factory=factory)
    try:
        with pytest.raises(TypeError, match="volcengine Client factory"):
            await container.get_ark()
    finally:
        await container.aclose()


@pytest.mark.asyncio
async def test_build_profiles_binds_driver_key_and_fingerprint() -> None:
    config = _ark_config({"name": "ark", "api_key": "sk-test"})
    container = VendorClientContainer(factory=lambda key: ArkClient())
    try:
        bindings = await ARK_PUBLIC_CONTRIBUTION.build_profiles(config, container)
    finally:
        await container.aclose()

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.name == "ark"
    assert binding.provider_key == "volcengine_ark"
    assert binding.driver_key == "volcengine_ark.media.v1"
    assert binding.credential_fingerprint
