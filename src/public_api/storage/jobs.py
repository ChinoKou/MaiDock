from collections.abc import Sequence
from pathlib import Path
from time import time
from uuid import uuid4
import secrets
import sqlite3

from ..config import PublicApiResourceConfig
from ..domain import JobStage, JobStatus, PublicJsonObject, UploadStatus
from ..domain.json_types import normalize_public_json_object
from .database import SqliteDatabase
from .errors import PublicApiStorageError
from .records import (
    IdempotencyRecord,
    JobOutputRecord,
    JobRecord,
    NewJob,
    StoredJob,
    decode_json,
    encode_json,
)
from .uploads import UploadRepository


class JobRepository:
    """作业、输出、幂等、取消和恢复检查点。"""

    def __init__(
        self,
        database: SqliteDatabase,
        config: PublicApiResourceConfig,
        uploads: UploadRepository,
    ) -> None:
        self.database = database
        self.config = config
        self.uploads = uploads

    async def create(self, new: NewJob, *, now: float | None = None) -> tuple[StoredJob, bool]:
        current = time() if now is None else now
        job_id = uuid4().hex
        expires_at = current + self.config.job_metadata_ttl_days * 86400

        def operation(connection: sqlite3.Connection) -> tuple[StoredJob, bool]:
            if new.idempotency_key is not None:
                existing = self._find_idempotency(connection, new.idempotency_key)
                if existing is not None:
                    if not secrets.compare_digest(existing.request_digest, new.request_digest or ""):
                        raise PublicApiStorageError(
                            "IDEMPOTENCY_CONFLICT",
                            "幂等键对应的请求内容不同",
                        )
                    return self._require(connection, existing.job_id), False
            queued = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if queued is None or int(queued[0]) >= self.config.max_queued_jobs:
                raise PublicApiStorageError("QUEUE_FULL", "Public API 作业队列已满")
            for upload_id in new.upload_ids:
                upload = self.uploads.require_in_transaction(connection, upload_id)
                if upload.status != UploadStatus.COMPLETE.value:
                    raise PublicApiStorageError("UPLOAD_NOT_READY", f"上传 {upload_id} 尚未完成")
            connection.execute(
                """
                INSERT INTO jobs(
                    id, driver_key, payload_version, profile_name, credential_fingerprint,
                    capability, model, mode, protocol_family, prepared_payload,
                    stage, status, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    new.driver_key,
                    new.payload_version,
                    new.profile_name,
                    new.credential_fingerprint,
                    new.capability,
                    new.model,
                    new.mode,
                    new.protocol_family,
                    encode_json(new.prepared_payload),
                    JobStage.PREPARING.value,
                    JobStatus.QUEUED.value,
                    current,
                    current,
                    expires_at,
                ),
            )
            connection.executemany(
                "INSERT INTO job_upload_refs(job_id, upload_id) VALUES (?, ?)",
                ((job_id, upload_id) for upload_id in new.upload_ids),
            )
            if new.idempotency_key is not None and new.request_digest is not None:
                connection.execute(
                    "INSERT INTO idempotency VALUES (?, ?, ?, ?, ?)",
                    (new.idempotency_key, new.request_digest, job_id, current, expires_at),
                )
            return self._require(connection, job_id), True

        return await self.database.run(operation)

    async def get(self, job_id: str) -> StoredJob:
        return await self.database.run(lambda connection: self._require(connection, job_id))

    async def claim_next(self, *, now: float | None = None) -> StoredJob | None:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> StoredJob | None:
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["id"])
            connection.execute(
                "UPDATE jobs SET status = ?, stage = ?, updated_at = ? WHERE id = ?",
                (JobStatus.RUNNING.value, JobStage.SUBMITTING.value, current, job_id),
            )
            return self._require(connection, job_id)

        return await self.database.run(operation)

    async def list_recoverable(self) -> tuple[StoredJob, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[StoredJob, ...]:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?) ORDER BY created_at",
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchall()
            return tuple(self._require(connection, str(row["id"])) for row in rows)

        return await self.database.run(operation)

    async def set_stage(
        self,
        job_id: str,
        stage: JobStage,
        *,
        remote_handle: PublicJsonObject | None = None,
        next_poll_at: float | None = None,
        provider_request_id: str | None = None,
        now: float | None = None,
    ) -> None:
        current = time() if now is None else now
        encoded_handle = encode_json(remote_handle) if remote_handle is not None else None

        def operation(connection: sqlite3.Connection) -> None:
            self._require(connection, job_id)
            connection.execute(
                """
                UPDATE jobs SET stage = ?, status = ?, remote_handle = COALESCE(?, remote_handle),
                    next_poll_at = ?, provider_request_id = COALESCE(?, provider_request_id), updated_at = ?
                WHERE id = ?
                """,
                (
                    stage.value,
                    JobStatus.RUNNING.value,
                    encoded_handle,
                    next_poll_at,
                    provider_request_id,
                    current,
                    job_id,
                ),
            )

        await self.database.run(operation)

    async def save_remote_outputs(
        self,
        job_id: str,
        outputs: Sequence[JobOutputRecord],
        *,
        usage: PublicJsonObject,
        warnings: tuple[str, ...],
        failed_output_count: int,
        provider_request_id: str | None,
    ) -> None:
        current = time()

        def operation(connection: sqlite3.Connection) -> None:
            self._require(connection, job_id)
            connection.executemany(
                """
                INSERT OR REPLACE INTO job_outputs(
                    job_id, ordinal, kind, text, url, media_type, artifact_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        job_id,
                        item.ordinal,
                        item.kind,
                        item.text,
                        item.url,
                        item.media_type,
                        item.artifact_id,
                    )
                    for item in outputs
                ),
            )
            connection.execute(
                """
                UPDATE jobs SET stage = ?, usage = ?, warnings = ?, failed_output_count = ?,
                    provider_request_id = COALESCE(?, provider_request_id), updated_at = ? WHERE id = ?
                """,
                (
                    JobStage.MATERIALIZING.value,
                    encode_json(usage),
                    encode_json(warnings),
                    failed_output_count,
                    provider_request_id,
                    current,
                    job_id,
                ),
            )

        await self.database.run(operation)

    async def update_summary(
        self,
        job_id: str,
        *,
        usage: PublicJsonObject,
        warnings: tuple[str, ...],
        failed_output_count: int,
    ) -> None:
        await self.database.run(
            lambda connection: connection.execute(
                "UPDATE jobs SET usage = ?, warnings = ?, failed_output_count = ?, updated_at = ? WHERE id = ?",
                (encode_json(usage), encode_json(warnings), failed_output_count, time(), job_id),
            )
        )

    async def mark_terminal(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        error_retryable: bool = False,
        error_uncertain: bool = False,
        provider_request_id: str | None = None,
    ) -> None:
        if status not in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
            JobStatus.EXPIRED,
        }:
            raise ValueError("mark_terminal 只接受终态")
        await self.database.run(
            lambda connection: connection.execute(
                """
                UPDATE jobs SET status = ?, error_code = ?, error_message = ?, error_retryable = ?,
                    error_uncertain = ?, provider_request_id = COALESCE(?, provider_request_id),
                    next_poll_at = NULL, updated_at = ? WHERE id = ?
                """,
                (
                    status.value,
                    error_code,
                    error_message,
                    int(error_retryable),
                    int(error_uncertain),
                    provider_request_id,
                    time(),
                    job_id,
                ),
            )
        )

    async def request_cancel(self, job_id: str) -> StoredJob:
        def operation(connection: sqlite3.Connection) -> StoredJob:
            job = self._require(connection, job_id)
            if job.record.status == JobStatus.QUEUED.value:
                connection.execute(
                    "UPDATE jobs SET status = ?, cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (JobStatus.CANCELED.value, time(), job_id),
                )
            elif job.record.status == JobStatus.RUNNING.value:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (time(), job_id),
                )
            return self._require(connection, job_id)

        return await self.database.run(operation)

    async def delete(self, job_id: str, *, now: float | None = None) -> None:
        current = time() if now is None else now

        def operation(connection: sqlite3.Connection) -> tuple[Path, ...]:
            job = self._require(connection, job_id)
            if job.record.status not in {
                JobStatus.SUCCEEDED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELED.value,
                JobStatus.EXPIRED.value,
            }:
                raise PublicApiStorageError("JOB_NOT_TERMINAL", "运行中的作业不能删除")
            rows = connection.execute(
                "SELECT path FROM artifacts WHERE job_id = ?",
                (job_id,),
            ).fetchall()
            paths = tuple(Path(str(row["path"])) for row in rows)
            connection.execute("DELETE FROM artifacts WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM idempotency WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            connection.execute(
                "INSERT OR REPLACE INTO deletion_tombstones VALUES (?, ?, ?, ?)",
                (
                    "job",
                    job_id,
                    current,
                    current + self.config.job_metadata_ttl_days * 86400,
                ),
            )
            return paths

        paths = await self.database.run(operation)
        for path in paths:
            path.unlink(missing_ok=True)

    def _require(self, connection: sqlite3.Connection, job_id: str) -> StoredJob:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise PublicApiStorageError("JOB_NOT_FOUND", f"作业不存在: {job_id}")
        job = JobRecord(
            id=str(row["id"]),
            driver_key=str(row["driver_key"]),
            payload_version=int(row["payload_version"]),
            profile_name=str(row["profile_name"]),
            credential_fingerprint=str(row["credential_fingerprint"]),
            capability=str(row["capability"]),
            model=str(row["model"]),
            mode=str(row["mode"]),
            protocol_family=str(row["protocol_family"]),
            prepared_payload=normalize_public_json_object(decode_json(str(row["prepared_payload"]))),
            remote_handle=(
                normalize_public_json_object(decode_json(str(row["remote_handle"])))
                if row["remote_handle"] is not None
                else None
            ),
            stage=str(row["stage"]),
            status=str(row["status"]),
            error_code=str(row["error_code"]) if row["error_code"] is not None else None,
            error_message=(str(row["error_message"]) if row["error_message"] is not None else None),
            error_retryable=bool(row["error_retryable"]),
            error_uncertain=bool(row["error_uncertain"]),
            provider_request_id=(str(row["provider_request_id"]) if row["provider_request_id"] is not None else None),
            usage=normalize_public_json_object(decode_json(str(row["usage"]))),
            warnings=_string_tuple(decode_json(str(row["warnings"]))),
            failed_output_count=int(row["failed_output_count"]),
            cancel_requested=bool(row["cancel_requested"]),
            next_poll_at=(float(row["next_poll_at"]) if row["next_poll_at"] is not None else None),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            expires_at=float(row["expires_at"]),
        )
        output_rows = connection.execute(
            "SELECT * FROM job_outputs WHERE job_id = ? ORDER BY ordinal",
            (job_id,),
        ).fetchall()
        outputs = tuple(JobOutputRecord.model_validate(dict(output_row)) for output_row in output_rows)
        return StoredJob(job, outputs)

    @staticmethod
    def _find_idempotency(
        connection: sqlite3.Connection,
        key: str,
    ) -> IdempotencyRecord | None:
        row = connection.execute(
            "SELECT * FROM idempotency WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        return IdempotencyRecord.model_validate(dict(row)) if row is not None else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PublicApiStorageError("STORE_CORRUPT", "持久化 warnings 不是字符串数组")
    return tuple(value)
