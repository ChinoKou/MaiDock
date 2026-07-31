from pathlib import Path
from typing import Protocol

from .models import (
    MaterializedArtifact,
    MediaOutcome,
    MediaRequest,
    ModelCapability,
    PreparedMediaOperation,
    VersionedOpaqueHandle,
)


class PublicProviderDriver(Protocol):
    """Public API 应用层使用的供应商无关执行端口。"""

    @property
    def driver_key(self) -> str: ...

    def capabilities(self) -> tuple[ModelCapability, ...]: ...

    def prepare(self, profile_name: str, request: MediaRequest) -> PreparedMediaOperation: ...

    async def submit(self, operation: PreparedMediaOperation) -> MediaOutcome: ...

    async def poll(self, handle: VersionedOpaqueHandle) -> MediaOutcome: ...

    async def cancel(self, handle: VersionedOpaqueHandle) -> MediaOutcome: ...

    async def upload_file(
        self,
        profile_name: str,
        *,
        model: str,
        path: Path,
        media_type: str,
    ) -> str: ...

    async def materialize(
        self,
        profile_name: str,
        *,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> MaterializedArtifact: ...
