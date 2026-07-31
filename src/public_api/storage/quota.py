import sqlite3

from ..config import PublicApiResourceConfig
from .errors import PublicApiStorageError


class QuotaRepository:
    """在调用方事务内计算并预留上传与产物空间。"""

    def __init__(self, config: PublicApiResourceConfig) -> None:
        self.config = config

    def reserve(self, connection: sqlite3.Connection, requested: int) -> None:
        row = connection.execute(
            """
            SELECT
                COALESCE((SELECT SUM(expected_size) FROM uploads), 0) +
                COALESCE((SELECT SUM(size) FROM artifacts), 0) +
                COALESCE((SELECT SUM(reserved_size) FROM artifact_staging), 0)
            """
        ).fetchone()
        used = int(row[0]) if row is not None else 0
        if used + requested > self.config.storage_quota_gb * 1024 * 1024 * 1024:
            raise PublicApiStorageError("STORAGE_QUOTA_EXCEEDED", "Public API 存储配额不足")
