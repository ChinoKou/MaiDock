from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from .common import HttpConnection, HttpSession, RetryPolicy, SharedHttpClient
from .families import JsonResource


@dataclass(frozen=True, slots=True)
class MimoConnection:
    http: HttpConnection
    retry: RetryPolicy
    chat_completions_path: str


class MimoSession:
    def __init__(self, session: HttpSession, connection: MimoConnection) -> None:
        self.retry = connection.retry
        self.chat_completions = JsonResource(
            session,
            path=connection.chat_completions_path,
            subject="Xiaomi Mimo Chat Completions",
        )


class MimoClient:
    """Xiaomi Mimo 原生资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: MimoConnection) -> AsyncIterator[MimoSession]:
        async with self._http.session(connection.http) as session:
            yield MimoSession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
