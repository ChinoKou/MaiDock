from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from .common import HttpConnection, HttpSession, RetryPolicy, SharedHttpClient
from .families import JsonResource, MultipartResource


@dataclass(frozen=True, slots=True)
class OpenAIConnection:
    http: HttpConnection
    retry: RetryPolicy
    responses_path: str
    embeddings_path: str
    audio_transcriptions_path: str


class OpenAISession:
    def __init__(self, session: HttpSession, connection: OpenAIConnection) -> None:
        self.retry = connection.retry
        self.responses = JsonResource(session, path=connection.responses_path, subject="OpenAI Responses")
        self.embeddings = JsonResource(session, path=connection.embeddings_path, subject="OpenAI Embeddings")
        self.audio_transcriptions = MultipartResource(
            session,
            path=connection.audio_transcriptions_path,
            subject="OpenAI Audio Transcriptions",
        )


class OpenAIClient:
    """OpenAI 原生资源 Client。"""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = SharedHttpClient(transport=transport)

    @property
    def closed(self) -> bool:
        return self._http.closed

    @asynccontextmanager
    async def session(self, connection: OpenAIConnection) -> AsyncIterator[OpenAISession]:
        async with self._http.session(connection.http) as session:
            yield OpenAISession(session, connection)

    async def aclose(self) -> None:
        await self._http.aclose()
