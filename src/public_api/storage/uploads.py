from hashlib import sha256
from pathlib import Path
from time import time
from uuid import uuid4
import os
import sqlite3

from ..config import PublicApiResourceConfig
from ..domain import UploadStatus
from .database import SqliteDatabase
from .errors import PublicApiStorageError
from .quota import QuotaRepository
from .records import UploadRecord


class UploadRepository:
    """顺序分块上传、校验、配额和文件生命周期。"""

    def __init__(
        self,
        database: SqliteDatabase,
        config: PublicApiResourceConfig,
        uploads_dir: Path,
        quota: QuotaRepository,
    ) -> None:
        self.database = database
        self.config = config
        self.uploads_dir = uploads_dir
        self.quota = quota

    async def create(
        self,
        *,
        media_type: str,
        size: int,
        expected_sha256: str,
        file_name: str,
        now: float | None = None,
    ) -> UploadRecord:
        current = time() if now is None else now
        if size > self.config.max_upload_mb * 1024 * 1024:
            raise PublicApiStorageError("UPLOAD_TOO_LARGE", "上传大小超过配置上限")
        upload_id = uuid4().hex
        path = self.uploads_dir / f"{upload_id}.part"
        expires_at = current + self.config.incomplete_upload_ttl_hours * 3600

        def operation(connection: sqlite3.Connection) -> UploadRecord:
            self.quota.reserve(connection, size)
            path.touch(exist_ok=False)
            connection.execute(
                """
                INSERT INTO uploads(
                    id, file_name, media_type, expected_size, expected_sha256,
                    received_size, status, path, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    file_name,
                    media_type,
                    size,
                    expected_sha256,
                    UploadStatus.INCOMPLETE.value,
                    str(path),
                    current,
                    current,
                    expires_at,
                ),
            )
            return UploadRecord(
                id=upload_id,
                file_name=file_name,
                media_type=media_type,
                expected_size=size,
                expected_sha256=expected_sha256,
                received_size=0,
                status=UploadStatus.INCOMPLETE.value,
                path=str(path),
                created_at=current,
                updated_at=current,
                expires_at=expires_at,
            )

        try:
            return await self.database.run(operation)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    async def write_chunk(
        self,
        upload_id: str,
        *,
        offset: int,
        data: bytes,
        now: float | None = None,
    ) -> UploadRecord:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> UploadRecord:
            record = self.require_in_transaction(connection, upload_id)
            if record.status != UploadStatus.INCOMPLETE.value:
                raise PublicApiStorageError("UPLOAD_ALREADY_COMPLETE", "上传已经完成")
            if offset != record.received_size:
                raise PublicApiStorageError("UPLOAD_OFFSET_MISMATCH", "上传块 offset 与已记录位置不一致")
            next_size = offset + len(data)
            if next_size > record.expected_size:
                raise PublicApiStorageError("UPLOAD_SIZE_MISMATCH", "上传内容超过声明大小")
            path = Path(record.path)
            with path.open("r+b") as stream:
                stream.truncate(record.received_size)
                stream.seek(record.received_size)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            connection.execute(
                "UPDATE uploads SET received_size = ?, updated_at = ? WHERE id = ?",
                (next_size, current, upload_id),
            )
            return record.model_copy(update={"received_size": next_size, "updated_at": current})

        return await self.database.run(operation)

    async def complete(self, upload_id: str, *, now: float | None = None) -> UploadRecord:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> UploadRecord:
            record = self.require_in_transaction(connection, upload_id)
            if record.status == UploadStatus.COMPLETE.value:
                return record
            if record.received_size != record.expected_size:
                raise PublicApiStorageError("UPLOAD_INCOMPLETE", "上传内容尚未完整写入")
            source = Path(record.path)
            if _file_sha256(source) != record.expected_sha256:
                raise PublicApiStorageError("UPLOAD_SHA256_MISMATCH", "上传内容 SHA-256 不匹配")
            destination = self.uploads_dir / f"{upload_id}.bin"
            os.replace(source, destination)
            expires_at = current + self.config.completed_upload_ttl_days * 86400
            connection.execute(
                "UPDATE uploads SET status = ?, path = ?, updated_at = ?, expires_at = ? WHERE id = ?",
                (UploadStatus.COMPLETE.value, str(destination), current, expires_at, upload_id),
            )
            return record.model_copy(
                update={
                    "status": UploadStatus.COMPLETE.value,
                    "path": str(destination),
                    "updated_at": current,
                    "expires_at": expires_at,
                }
            )

        try:
            return await self.database.run(operation)
        except BaseException:
            source = self.uploads_dir / f"{upload_id}.part"
            destination = self.uploads_dir / f"{upload_id}.bin"
            if destination.exists() and not source.exists():
                os.replace(destination, source)
            raise

    async def get(self, upload_id: str) -> UploadRecord:
        return await self.database.run(lambda connection: self.require_in_transaction(connection, upload_id))

    async def delete(self, upload_id: str, *, now: float | None = None) -> None:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> Path:
            record = self.require_in_transaction(connection, upload_id)
            referenced = connection.execute(
                "SELECT 1 FROM job_upload_refs WHERE upload_id = ? LIMIT 1",
                (upload_id,),
            ).fetchone()
            if referenced is not None:
                raise PublicApiStorageError("UPLOAD_IN_USE", "上传仍被作业引用")
            connection.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
            _insert_tombstone(
                connection,
                resource_type="upload",
                resource_id=upload_id,
                deleted_at=current,
                ttl_days=self.config.job_metadata_ttl_days,
            )
            return Path(record.path)

        path = await self.database.run(operation)
        path.unlink(missing_ok=True)

    @staticmethod
    def require_in_transaction(connection: sqlite3.Connection, upload_id: str) -> UploadRecord:
        row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        if row is None:
            raise PublicApiStorageError("UPLOAD_NOT_FOUND", f"上传不存在: {upload_id}")
        return UploadRecord.model_validate(dict(row))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _insert_tombstone(
    connection: sqlite3.Connection,
    *,
    resource_type: str,
    resource_id: str,
    deleted_at: float,
    ttl_days: int,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO deletion_tombstones VALUES (?, ?, ?, ?)",
        (resource_type, resource_id, deleted_at, deleted_at + ttl_days * 86400),
    )
