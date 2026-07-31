from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from .common import HttpConnection, HttpSession, RetryPolicy, SharedHttpClient
from .families import JsonResource, MultipartResource


@dataclass(frozen=True, slots=True)
class SiliconFlowConnection:
    http: HttpConnection
    retry: RetryPolicy
    chat_completions_path: str
    embeddings_path: str
    audio_transcriptions_path: str


class SiliconFlowSession:
    def __init__(self, session: HttpSession, connection: SiliconFlowConnection) -> None:
        self.retry = connection.retry
        self.chat_completions = JsonResource(
            session,
            path=connection.chat_completions_path,
            subject="SiliconFlow Chat Completions",
        )
        self.embeddings = JsonResource(session, path=connection.embeddings_path, subject="SiliconFlow Embeddings")
        self.audio_transcriptions = MultipartResource(
            session,
            path=connection.audio_transcriptions_path,
            subject="SiliconFlow Audio Transcriptions",
        )


class SiliconFlowClient:
    """SiliconFlow 原生资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: SiliconFlowConnection) -> AsyncIterator[SiliconFlowSession]:
        async with self._http.session(connection.http) as session:
            yield SiliconFlowSession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
