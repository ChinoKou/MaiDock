from pathlib import Path
from time import time
import sqlite3

from ..config import PublicApiResourceConfig
from ..domain import JobStatus
from .artifacts import ArtifactRepository
from .database import SqliteDatabase
from .jobs import JobRepository
from .quota import QuotaRepository
from .uploads import UploadRepository


class PublicApiStore:
    """Public API 存储装配、数据库生命周期和跨资源清理边界。"""

    def __init__(self, data_dir: Path, config: PublicApiResourceConfig) -> None:
        self.database = SqliteDatabase(data_dir / "maidock_public_api.sqlite3")
        self.root = data_dir / "public_api"
        self.uploads_dir = self.root / "uploads"
        self.artifacts_dir = self.root / "artifacts"
        self.staging_dir = self.root / "staging"
        self._config = config
        self.quota = QuotaRepository(config)
        self.uploads = UploadRepository(self.database, config, self.uploads_dir, self.quota)
        self.jobs = JobRepository(self.database, config, self.uploads)
        self.artifacts = ArtifactRepository(
            self.database,
            config,
            self.staging_dir,
            self.artifacts_dir,
            self.quota,
        )

    @property
    def config(self) -> PublicApiResourceConfig:
        return self._config

    @config.setter
    def config(self, value: PublicApiResourceConfig) -> None:
        self._config = value
        self.quota.config = value
        self.uploads.config = value
        self.jobs.config = value
        self.artifacts.config = value

    async def open(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        await self.database.open()

    async def close(self) -> None:
        await self.database.close()

    async def cleanup_expired(self, *, now: float | None = None) -> None:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> tuple[Path, ...]:
            rows = connection.execute(
                "SELECT path FROM uploads WHERE expires_at <= ? "
                "AND NOT EXISTS (SELECT 1 FROM job_upload_refs "
                "WHERE job_upload_refs.upload_id = uploads.id) "
                "UNION ALL SELECT path FROM artifacts WHERE expires_at <= ? "
                "UNION ALL SELECT path FROM artifact_staging WHERE created_at <= ?",
                (current, current, current - 86400),
            ).fetchall()
            connection.execute(
                "DELETE FROM uploads WHERE expires_at <= ? "
                "AND NOT EXISTS (SELECT 1 FROM job_upload_refs "
                "WHERE job_upload_refs.upload_id = uploads.id)",
                (current,),
            )
            connection.execute("DELETE FROM artifacts WHERE expires_at <= ?", (current,))
            connection.execute(
                "DELETE FROM artifact_staging WHERE created_at <= ?",
                (current - 86400,),
            )
            connection.execute("DELETE FROM idempotency WHERE expires_at <= ?", (current,))
            connection.execute("DELETE FROM deletion_tombstones WHERE expires_at <= ?", (current,))
            connection.execute(
                """
                UPDATE jobs SET status = ?, updated_at = ? WHERE expires_at <= ?
                    AND status NOT IN (?, ?, ?, ?)
                """,
                (
                    JobStatus.EXPIRED.value,
                    current,
                    current,
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELED.value,
                    JobStatus.EXPIRED.value,
                ),
            )
            return tuple(Path(str(row["path"])) for row in rows)

        paths = await self.database.run(operation)
        for path in paths:
            path.unlink(missing_ok=True)
