from src.core.common import (
    MAIDOCK_USER_AGENT,
    build_openai_compatible_client_config,
    with_default_user_agent,
)
from src.providers.anthropic_messages_provider.messages import (
    build_client_config as build_anthropic_http_config,
)
from src.providers.common.httpx import build_httpx_client_config
from src.schemas import ApiProviderSnapshot


def _api_provider(default_headers: dict | None = None) -> ApiProviderSnapshot:
    return ApiProviderSnapshot.model_validate(
        {
            "api_key": "test-key",
            "auth_type": "bearer",
            "base_url": "https://example.com/v1",
            "default_headers": default_headers or {},
        }
    )


def test_with_default_user_agent_injects_default_when_missing() -> None:
    assert with_default_user_agent({}) == {"User-Agent": MAIDOCK_USER_AGENT}


def test_with_default_user_agent_normalizes_blank_values() -> None:
    assert with_default_user_agent({}, "") == {"User-Agent": MAIDOCK_USER_AGENT}
    assert with_default_user_agent({}, "  ") == {"User-Agent": MAIDOCK_USER_AGENT}
    assert with_default_user_agent({}, None) == {"User-Agent": MAIDOCK_USER_AGENT}


def test_with_default_user_agent_trims_custom_value() -> None:
    assert with_default_user_agent({}, "  Custom-UA/1  ") == {"User-Agent": "Custom-UA/1"}


def test_with_default_user_agent_preserves_existing_header_case_insensitively() -> None:
    assert with_default_user_agent({"user-agent": "Existing-UA/1"}, "Custom-UA/1") == {"user-agent": "Existing-UA/1"}
    assert with_default_user_agent({"User-Agent": "Existing-UA/2"}, "Custom-UA/1") == {"User-Agent": "Existing-UA/2"}


def test_openai_client_config_uses_provider_user_agent() -> None:
    client_config = build_openai_compatible_client_config(_api_provider(), user_agent="OpenAI-UA/1")

    assert client_config.default_headers["User-Agent"] == "OpenAI-UA/1"


def test_anthropic_http_config_uses_provider_user_agent() -> None:
    client_config = build_anthropic_http_config(_api_provider(), user_agent="Anthropic-UA/1")

    assert client_config.default_headers["User-Agent"] == "Anthropic-UA/1"


def test_client_config_preserves_existing_provider_user_agent() -> None:
    openai_config = build_openai_compatible_client_config(
        _api_provider({"user-agent": "Existing-UA/1"}), user_agent="OpenAI-UA/1"
    )
    anthropic_config = build_anthropic_http_config(
        _api_provider({"User-Agent": "Existing-UA/2"}), user_agent="Anthropic-UA/1"
    )

    assert openai_config.default_headers == {"user-agent": "Existing-UA/1"}
    assert anthropic_config.default_headers["User-Agent"] == "Existing-UA/2"


def test_httpx_client_config_uses_provider_user_agent() -> None:
    client_config = build_httpx_client_config(
        _api_provider(),
        default_base_url="https://example.com/api/v1",
        user_agent="Httpx-UA/1",
    )

    assert client_config.default_headers["User-Agent"] == "Httpx-UA/1"


def test_httpx_client_config_preserves_existing_provider_user_agent() -> None:
    client_config = build_httpx_client_config(
        _api_provider({"user-agent": "Existing-UA/3"}),
        default_base_url="https://example.com/api/v1",
        user_agent="Httpx-UA/1",
    )

    assert client_config.default_headers["user-agent"] == "Existing-UA/3"


def test_httpx_client_config_supports_siliconflow_user_agent_pattern() -> None:
    client_config = build_httpx_client_config(
        _api_provider(),
        default_base_url="https://api.siliconflow.cn/v1",
        user_agent="SiliconFlow-UA/1",
    )

    assert client_config.default_headers["User-Agent"] == "SiliconFlow-UA/1"
