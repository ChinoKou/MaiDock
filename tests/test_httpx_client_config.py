import httpx
import pytest

from src.core.common import resolve_max_retries, resolve_retry_interval
from src.providers.common.httpx import build_httpx_client_config, create_async_client, resolve_endpoint_path
from tests.support.http import make_api_provider


@pytest.mark.parametrize(
    ("base_url", "api_prefix", "endpoint_path", "expected"),
    [
        ("https://ark.example/api/v3", "api/v3", "responses", "responses"),
        ("https://relay.example/custom/api/v3", "/api/v3/", "/responses/", "responses"),
        ("https://dashscope.example", "api/v1", "services/generation", "api/v1/services/generation"),
        ("https://dashscope.example/api/v1", "api/v1", "services/generation", "services/generation"),
        ("https://example.com/custom", "", "/responses/", "responses"),
    ],
)
def test_resolve_endpoint_path_preserves_existing_api_prefix(
    base_url: str,
    api_prefix: str,
    endpoint_path: str,
    expected: str,
) -> None:
    assert resolve_endpoint_path(base_url, api_prefix=api_prefix, endpoint_path=endpoint_path) == expected


def test_resolve_endpoint_path_rejects_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint_path"):
        resolve_endpoint_path("https://example.com", api_prefix="api/v1", endpoint_path=" / ")


def test_build_httpx_client_config_supports_header_auth_with_prefix() -> None:
    config = build_httpx_client_config(
        make_api_provider(
            auth_type="header",
            auth_header_name="X-Api-Key",
            auth_header_prefix=" Token ",
        ),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-Test/1",
    )

    assert config.default_headers["X-Api-Key"] == "Token test-key"


def test_build_httpx_client_config_supports_header_auth_without_prefix() -> None:
    config = build_httpx_client_config(
        make_api_provider(
            auth_type="header",
            auth_header_name="X-Api-Key",
            auth_header_prefix=" ",
        ),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-Test/1",
    )

    assert config.default_headers["X-Api-Key"] == "test-key"


def test_build_httpx_client_config_supports_bearer_query_and_none_auth() -> None:
    bearer = build_httpx_client_config(
        make_api_provider(auth_type="bearer"),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-Test/1",
    )
    query = build_httpx_client_config(
        make_api_provider(auth_type="query", auth_query_name="access_token"),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-Test/1",
    )
    none_auth = build_httpx_client_config(
        make_api_provider(auth_type="none", api_key=""),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-Test/1",
    )

    assert bearer.default_headers["Authorization"] == "Bearer test-key"
    assert query.default_query["access_token"] == "test-key"
    assert "Authorization" not in none_auth.default_headers
    assert "api_key" not in none_auth.default_query


def test_build_httpx_client_config_preserves_explicit_default_headers() -> None:
    config = build_httpx_client_config(
        make_api_provider(
            auth_type="none",
            api_key="",
            default_headers={
                "accept": "application/x-ndjson",
                "content-type": "application/custom+json",
                "user-agent": "Host-UA/1",
            },
        ),
        default_base_url="https://default.example/api/v1",
        user_agent="MaiDock-UA/1",
    )

    assert config.default_headers["accept"] == "application/x-ndjson"
    assert config.default_headers["content-type"] == "application/custom+json"
    assert config.default_headers["user-agent"] == "Host-UA/1"


def test_build_httpx_client_config_rejects_non_string_default_header() -> None:
    with pytest.raises(TypeError, match="default_headers"):
        build_httpx_client_config(
            make_api_provider(default_headers={"X-Retry": 3}),
            default_base_url="https://default.example/api/v1",
            user_agent="MaiDock-Test/1",
        )


def test_build_httpx_client_config_selects_host_or_forced_default_base() -> None:
    provider = make_api_provider(base_url="https://relay.example/custom")

    host_config = build_httpx_client_config(
        provider,
        default_base_url="https://official.example/api/v3",
        user_agent="MaiDock-Test/1",
    )
    forced_config = build_httpx_client_config(
        provider,
        default_base_url="https://official.example/api/v3",
        user_agent="MaiDock-Test/1",
        force_default_base_url=True,
    )

    assert host_config.base_url == "https://relay.example/custom"
    assert forced_config.base_url == "https://official.example/api/v3"


def test_build_httpx_client_config_requires_host_base_when_not_forced() -> None:
    with pytest.raises(ValueError, match="base_url"):
        build_httpx_client_config(
            make_api_provider(base_url=None),
            default_base_url="https://official.example/api/v3",
            user_agent="MaiDock-Test/1",
        )


def test_build_httpx_client_config_uses_host_timeout_and_retry_values() -> None:
    config = build_httpx_client_config(
        make_api_provider(timeout=45, max_retry=5, retry_interval=9),
        default_base_url="https://official.example/api/v1",
        user_agent="MaiDock-Test/1",
        default_timeout=300.0,
        default_max_retries=2,
        default_retry_interval=4.0,
    )

    assert config.timeout == 45.0
    assert config.max_retries == 5
    assert config.retry_interval == 9.0


def test_build_httpx_client_config_can_force_config_retry_values() -> None:
    config = build_httpx_client_config(
        make_api_provider(max_retry=5, retry_interval=9),
        default_base_url="https://official.example/api/v1",
        user_agent="MaiDock-Test/1",
        default_timeout=300.0,
        default_max_retries=2,
        force_max_retries=True,
        default_retry_interval=4.0,
        force_retry_interval=True,
    )

    assert config.timeout == 300.0
    assert config.max_retries == 2
    assert config.retry_interval == 4.0


def test_retry_resolvers_follow_force_host_and_config_precedence() -> None:
    host_values = make_api_provider(max_retry=5, retry_interval=10)
    missing_values = make_api_provider(max_retry=None, retry_interval=None)

    assert resolve_max_retries(host_values, config_value=3, force=True, default=7) == 3
    assert resolve_max_retries(host_values, config_value=3, force=False, default=7) == 5
    assert resolve_max_retries(missing_values, config_value=3, force=False, default=7) == 3
    assert resolve_retry_interval(host_values, config_value=5.0, force=True, default=7.0) == 5.0
    assert resolve_retry_interval(host_values, config_value=5.0, force=False, default=7.0) == 10.0
    assert resolve_retry_interval(missing_values, config_value=5.0, force=False, default=7.0) == 5.0


@pytest.mark.asyncio
async def test_create_async_client_serializes_default_query_values() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    config = build_httpx_client_config(
        make_api_provider(
            default_query={
                "enabled": False,
                "integer": 2,
                "nested": {"items": [1, True]},
                "ratio": 1.5,
                "skip": None,
                "text": "中文",
            }
        ),
        default_base_url="https://official.example/api/v1",
        user_agent="MaiDock-Test/1",
    )
    async with create_async_client(config, transport=httpx.MockTransport(handler)) as client:
        response = await client.get("probe")

    assert response.status_code == 204
    assert len(requests) == 1
    query = requests[0].url.params
    assert query["enabled"] == "False"
    assert query["integer"] == "2"
    assert query["nested"] == '{"items":[1,true]}'
    assert query["ratio"] == "1.5"
    assert query["text"] == "中文"
    assert "skip" not in query
