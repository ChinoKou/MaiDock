import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class SqliteDatabase:
    """Public API 专用 SQLite 连接和事务边界。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        async with self._lock:
            if self._connection is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    driver_key TEXT NOT NULL,
                    payload_version INTEGER NOT NULL,
                    profile_name TEXT NOT NULL,
                    credential_fingerprint TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    protocol_family TEXT NOT NULL,
                    prepared_payload TEXT NOT NULL,
                    remote_handle TEXT,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    error_retryable INTEGER NOT NULL DEFAULT 0,
                    error_uncertain INTEGER NOT NULL DEFAULT 0,
                    provider_request_id TEXT,
                    usage TEXT NOT NULL DEFAULT '{}',
                    warnings TEXT NOT NULL DEFAULT '[]',
                    failed_output_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    next_poll_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_outputs (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT,
                    url TEXT,
                    media_type TEXT,
                    artifact_id TEXT,
                    PRIMARY KEY (job_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    expected_size INTEGER NOT NULL,
                    expected_sha256 TEXT NOT NULL,
                    received_size INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_upload_refs (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    upload_id TEXT NOT NULL REFERENCES uploads(id),
                    PRIMARY KEY(job_id, upload_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    duration_seconds REAL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_staging (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    reserved_size INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    job_id TEXT NOT NULL REFERENCES jobs(id),
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deletion_tombstones (
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    deleted_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY(resource_type, resource_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status, next_poll_at, created_at);
                CREATE INDEX IF NOT EXISTS jobs_expiry_idx ON jobs(expires_at);
                CREATE INDEX IF NOT EXISTS uploads_expiry_idx ON uploads(expires_at);
                CREATE INDEX IF NOT EXISTS artifacts_expiry_idx ON artifacts(expires_at);
                """
            )
            connection.commit()
            self._connection = connection

    async def run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        async with self._lock:
            connection = self._connection
            if connection is None:
                raise RuntimeError("Public API 数据库尚未打开")
            try:
                result = operation(connection)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is not None:
                connection.close()
