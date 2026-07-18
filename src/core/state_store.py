from pathlib import Path
from time import time
import asyncio
import json
import sqlite3

from .json_types import JsonValue, normalize_json_value


class PluginStateStore:
    """MaiDock 内部命名空间状态存储。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        """按需打开数据库并初始化表结构。"""

        async with self._lock:
            await self._ensure_open_locked()

    async def close(self) -> None:
        """关闭数据库连接。"""

        async with self._lock:
            connection = self._connection
            self._connection = None
            if connection is not None:
                await asyncio.to_thread(connection.close)

    async def get(self, namespace: str, key: str, *, now: float | None = None) -> JsonValue:
        """读取未过期的 JSON 值；不存在或已过期时返回 ``None``。"""

        checked_namespace = self._validate_identifier(namespace, field_name="namespace")
        checked_key = self._validate_identifier(key, field_name="key")
        current_time = time() if now is None else now
        async with self._lock:
            connection = await self._ensure_open_locked()
            row = await asyncio.to_thread(
                self._get_sync,
                connection,
                checked_namespace,
                checked_key,
            )
            if row is None:
                return None
            value_json, expires_at = row
            if expires_at is not None and expires_at <= current_time:
                await asyncio.to_thread(
                    self._delete_sync,
                    connection,
                    checked_namespace,
                    checked_key,
                )
                return None
        value: object = json.loads(value_json)
        return normalize_json_value(value)

    async def set(
        self,
        namespace: str,
        key: str,
        value: JsonValue,
        *,
        expires_at: float | None = None,
        now: float | None = None,
    ) -> None:
        """以 upsert 方式写入 JSON 值。"""

        checked_namespace = self._validate_identifier(namespace, field_name="namespace")
        checked_key = self._validate_identifier(key, field_name="key")
        value_json = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        updated_at = time() if now is None else now
        async with self._lock:
            connection = await self._ensure_open_locked()
            await asyncio.to_thread(
                self._set_sync,
                connection,
                checked_namespace,
                checked_key,
                value_json,
                expires_at,
                updated_at,
            )

    async def delete(self, namespace: str, key: str) -> bool:
        """删除单个状态条目，并返回是否实际删除。"""

        checked_namespace = self._validate_identifier(namespace, field_name="namespace")
        checked_key = self._validate_identifier(key, field_name="key")
        async with self._lock:
            connection = await self._ensure_open_locked()
            deleted = await asyncio.to_thread(
                self._delete_sync,
                connection,
                checked_namespace,
                checked_key,
            )
        return deleted > 0

    async def delete_expired(self, namespace: str | None = None, *, now: float | None = None) -> int:
        """删除指定命名空间或全部命名空间的过期条目。"""

        checked_namespace = (
            None if namespace is None else self._validate_identifier(namespace, field_name="namespace")
        )
        current_time = time() if now is None else now
        async with self._lock:
            connection = await self._ensure_open_locked()
            return await asyncio.to_thread(
                self._delete_expired_sync,
                connection,
                checked_namespace,
                current_time,
            )

    async def _ensure_open_locked(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = await asyncio.to_thread(self._open_sync)
        return self._connection

    def _open_sync(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30.0, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS state_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                expires_at REAL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS state_entries_expires_at_idx ON state_entries (expires_at)"
        )
        connection.commit()
        return connection

    @staticmethod
    def _get_sync(
        connection: sqlite3.Connection,
        namespace: str,
        key: str,
    ) -> tuple[str, float | None] | None:
        row = connection.execute(
            "SELECT value_json, expires_at FROM state_entries WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), None if row[1] is None else float(row[1])

    @staticmethod
    def _set_sync(
        connection: sqlite3.Connection,
        namespace: str,
        key: str,
        value_json: str,
        expires_at: float | None,
        updated_at: float,
    ) -> None:
        with connection:
            connection.execute(
                """
                INSERT INTO state_entries (namespace, key, value_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (namespace, key, value_json, expires_at, updated_at),
            )

    @staticmethod
    def _delete_sync(connection: sqlite3.Connection, namespace: str, key: str) -> int:
        with connection:
            cursor = connection.execute(
                "DELETE FROM state_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
        return cursor.rowcount

    @staticmethod
    def _delete_expired_sync(
        connection: sqlite3.Connection,
        namespace: str | None,
        current_time: float,
    ) -> int:
        with connection:
            if namespace is None:
                cursor = connection.execute(
                    "DELETE FROM state_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (current_time,),
                )
            else:
                cursor = connection.execute(
                    """
                    DELETE FROM state_entries
                    WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?
                    """,
                    (namespace, current_time),
                )
        return cursor.rowcount

    @staticmethod
    def _validate_identifier(value: str, *, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} 不能为空")
        return normalized
