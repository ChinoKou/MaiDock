from pathlib import Path
from time import time
from uuid import uuid4
import os
import sqlite3

from ..config import PublicApiResourceConfig
from .database import SqliteDatabase
from .errors import PublicApiStorageError
from .quota import QuotaRepository
from .records import ArtifactRecord, StagingRecord


class ArtifactRepository:
    """产物 staging、原子提交、索引和随机读取。"""

    def __init__(
        self,
        database: SqliteDatabase,
        config: PublicApiResourceConfig,
        staging_dir: Path,
        artifacts_dir: Path,
        quota: QuotaRepository,
    ) -> None:
        self.database = database
        self.config = config
        self.staging_dir = staging_dir
        self.artifacts_dir = artifacts_dir
        self.quota = quota

    async def begin(self, job_id: str, ordinal: int, reserved_size: int) -> StagingRecord:
        current = time()
        staging_id = uuid4().hex
        path = self.staging_dir / f"{staging_id}.part"

        def operation(connection: sqlite3.Connection) -> StagingRecord:
            self.quota.reserve(connection, reserved_size)
            row = connection.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise PublicApiStorageError("JOB_NOT_FOUND", f"作业不存在: {job_id}")
            path.touch(exist_ok=False)
            connection.execute(
                "INSERT INTO artifact_staging VALUES (?, ?, ?, ?, ?, ?)",
                (staging_id, job_id, ordinal, str(path), reserved_size, current),
            )
            return StagingRecord(
                id=staging_id,
                job_id=job_id,
                ordinal=ordinal,
                path=str(path),
                reserved_size=reserved_size,
                created_at=current,
            )

        try:
            return await self.database.run(operation)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    async def commit(
        self,
        staging: StagingRecord,
        *,
        media_type: str,
        size: int,
        sha256_hex: str,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
    ) -> ArtifactRecord:
        if size > staging.reserved_size:
            raise PublicApiStorageError("ARTIFACT_TOO_LARGE", "产物大小超过预留空间")
        current = time()
        artifact_id = uuid4().hex
        source = Path(staging.path)
        destination = self.artifacts_dir / f"{artifact_id}.bin"
        os.replace(source, destination)
        expires_at = current + self.config.artifact_ttl_days * 86400

        def operation(connection: sqlite3.Connection) -> ArtifactRecord:
            connection.execute(
                """
                INSERT INTO artifacts(
                    id, job_id, path, media_type, size, sha256, width, height,
                    duration_seconds, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    staging.job_id,
                    str(destination),
                    media_type,
                    size,
                    sha256_hex,
                    width,
                    height,
                    duration_seconds,
                    current,
                    expires_at,
                ),
            )
            connection.execute("DELETE FROM artifact_staging WHERE id = ?", (staging.id,))
            connection.execute(
                "UPDATE job_outputs SET artifact_id = ? WHERE job_id = ? AND ordinal = ?",
                (artifact_id, staging.job_id, staging.ordinal),
            )
            return ArtifactRecord(
                id=artifact_id,
                job_id=staging.job_id,
                path=str(destination),
                media_type=media_type,
                size=size,
                sha256=sha256_hex,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                created_at=current,
                expires_at=expires_at,
            )

        try:
            return await self.database.run(operation)
        except BaseException:
            if destination.exists() and not source.exists():
                os.replace(destination, source)
            raise

    async def abort(self, staging: StagingRecord) -> None:
        await self.database.run(
            lambda connection: connection.execute(
                "DELETE FROM artifact_staging WHERE id = ?",
                (staging.id,),
            )
        )
        Path(staging.path).unlink(missing_ok=True)

    async def get(self, artifact_id: str) -> ArtifactRecord:
        return await self.database.run(lambda connection: self.require_in_transaction(connection, artifact_id))

    async def read(self, artifact_id: str, *, offset: int, length: int) -> tuple[ArtifactRecord, bytes]:
        record = await self.get(artifact_id)
        if offset > record.size:
            raise PublicApiStorageError("INVALID_OFFSET", "artifact offset 超过产物大小")
        with Path(record.path).open("rb") as stream:
            stream.seek(offset)
            return record, stream.read(length)

    @staticmethod
    def require_in_transaction(
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> ArtifactRecord:
        row = connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise PublicApiStorageError("ARTIFACT_NOT_FOUND", f"产物不存在: {artifact_id}")
        return ArtifactRecord.model_validate(dict(row))
