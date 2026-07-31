from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from .common import HttpConnection, HttpSession, RetryPolicy, SharedHttpClient
from .families import JsonResource


@dataclass(frozen=True, slots=True)
class AnthropicConnection:
    http: HttpConnection
    retry: RetryPolicy
    messages_path: str


class AnthropicSession:
    def __init__(self, session: HttpSession, connection: AnthropicConnection) -> None:
        self.retry = connection.retry
        self.messages = JsonResource(session, path=connection.messages_path, subject="Anthropic Messages")


class AnthropicClient:
    """Anthropic Messages 原生资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: AnthropicConnection) -> AsyncIterator[AnthropicSession]:
        async with self._http.session(connection.http) as session:
            yield AnthropicSession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
