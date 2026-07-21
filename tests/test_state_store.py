import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from src.core.state_store import PluginStateStore


@pytest.mark.asyncio
async def test_state_store_upsert_namespace_delete_and_expiry(tmp_path: Path) -> None:
    database_path = tmp_path / "maidock_state.sqlite3"
    store = PluginStateStore(database_path)

    await store.set("cache", "same-key", {"value": 1}, now=100.0)
    await store.set("cache", "same-key", {"value": 2}, now=101.0)
    await store.set("other", "same-key", {"value": 3}, now=102.0)
    await store.set("cache", "expired", {"value": 4}, expires_at=110.0, now=103.0)

    assert await store.get("cache", "same-key", now=105.0) == {"value": 2}
    assert await store.get("other", "same-key", now=105.0) == {"value": 3}
    assert await store.get("cache", "expired", now=110.0) is None
    assert await store.delete("cache", "same-key") is True
    assert await store.delete("cache", "same-key") is False

    await store.close()
    assert database_path.exists()


@pytest.mark.asyncio
async def test_state_store_delete_expired_and_concurrent_writes(tmp_path: Path) -> None:
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")

    await asyncio.gather(
        *(
            store.set(
                "concurrent",
                f"key-{index}",
                {"index": index},
                expires_at=50.0 if index % 2 == 0 else 150.0,
                now=10.0,
            )
            for index in range(20)
        )
    )

    assert await store.delete_expired("concurrent", now=100.0) == 10
    remaining = await asyncio.gather(*(store.get("concurrent", f"key-{index}", now=100.0) for index in range(20)))
    assert sum(value is not None for value in remaining) == 10
    await store.close()


@pytest.mark.asyncio
async def test_state_store_exposes_invalid_persisted_json(tmp_path: Path) -> None:
    database_path = tmp_path / "maidock_state.sqlite3"
    store = PluginStateStore(database_path)
    await store.open()

    connection = sqlite3.connect(database_path)
    with connection:
        connection.execute(
            """
            INSERT INTO state_entries (namespace, key, value_json, expires_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("cache", "broken", "{not-json", None, 1.0),
        )
    connection.close()

    with pytest.raises(json.JSONDecodeError):
        await store.get("cache", "broken")
    await store.close()


@pytest.mark.asyncio
async def test_state_store_cannot_reopen_after_close(tmp_path: Path) -> None:
    store = PluginStateStore(tmp_path / "maidock_state.sqlite3")
    await store.set("test", "key", {"value": True})
    await store.close()

    with pytest.raises(RuntimeError, match="MaiDock"):
        await store.get("test", "key")
