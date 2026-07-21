from collections.abc import AsyncIterator, Mapping

import httpx

from src.core.json_types import JsonValue
from src.schemas import ApiProviderSnapshot


def make_api_provider(
    *,
    base_url: str | None = "https://example.com/api/v1",
    auth_type: str = "bearer",
    api_key: str = "test-key",
    auth_header_name: str = "Authorization",
    auth_header_prefix: str = "Bearer",
    auth_query_name: str = "api_key",
    default_headers: Mapping[str, JsonValue] | None = None,
    default_query: Mapping[str, JsonValue] | None = None,
    timeout: int | float | None = None,
    max_retry: int | None = None,
    retry_interval: int | None = None,
) -> ApiProviderSnapshot:
    """构造 HTTP 公共层测试所需的 Host Provider 快照。"""

    return ApiProviderSnapshot.model_validate(
        {
            "api_key": api_key,
            "auth_header_name": auth_header_name,
            "auth_header_prefix": auth_header_prefix,
            "auth_query_name": auth_query_name,
            "auth_type": auth_type,
            "base_url": base_url,
            "default_headers": dict(default_headers or {}),
            "default_query": dict(default_query or {}),
            "max_retry": max_retry,
            "retry_interval": retry_interval,
            "timeout": timeout,
        }
    )


class TrackingByteStream(httpx.AsyncByteStream):
    """提供分块数据、可选读取异常和关闭次数观测的异步字节流。"""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: httpx.HTTPError | None = None,
        error_after_chunks: int | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.error_after_chunks = error_after_chunks
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if self.error is not None and self.error_after_chunks == index:
                raise self.error
            yield chunk
        if self.error is not None and self.error_after_chunks == len(self.chunks):
            raise self.error

    async def aclose(self) -> None:
        self.close_calls += 1
