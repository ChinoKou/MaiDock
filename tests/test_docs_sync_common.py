"""测试供应商文档同步脚本共享的文件系统操作。"""

import json
from hashlib import sha256
from pathlib import Path

import docs_sync_common as docs_sync_common_module
import pytest
from docs_sync_common import (
    PROJECT_ROOT,
    DocsSyncError,
    SyncStats,
    atomic_write,
    encode_stable_manifest,
    project_output_path,
    remove_empty_directories,
    remove_tracked_file,
)


def test_sync_stats_starts_from_zero() -> None:
    assert SyncStats() == SyncStats(created=0, updated=0, unchanged=0, removed=0, preserved=0)


def test_project_output_path_accepts_repository_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "docs"
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    assert project_output_path(output) == output.resolve()


@pytest.mark.parametrize("output", [PROJECT_ROOT, PROJECT_ROOT.parent])
def test_project_output_path_rejects_root_and_escape(output: Path) -> None:
    with pytest.raises(DocsSyncError, match="必须位于 MaiDock 仓库内"):
        project_output_path(output)


def test_atomic_write_creates_and_replaces_file(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"

    atomic_write(target, b"first")
    atomic_write(target, b"second")

    assert target.read_bytes() == b"second"
    assert not list(tmp_path.glob(".*.tmp"))


def test_remove_tracked_file_handles_missing_directory_and_digest(
    tmp_path: Path,
) -> None:
    missing = remove_tracked_file(tmp_path, "missing.md", "unused")
    directory = tmp_path / "directory"
    directory.mkdir()
    directory_result = remove_tracked_file(tmp_path, "directory", "unused")
    modified = tmp_path / "modified.md"
    modified.write_bytes(b"local edit")
    modified_result = remove_tracked_file(tmp_path, "modified.md", "wrong")
    tracked = tmp_path / "tracked.md"
    tracked.write_bytes(b"remote")
    tracked_result = remove_tracked_file(tmp_path, "tracked.md", sha256(b"remote").hexdigest())

    assert missing is True
    assert directory_result is False
    assert modified_result is False
    assert tracked_result is True
    assert modified.exists()
    assert not tracked.exists()


def test_remove_tracked_file_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(DocsSyncError, match="越界路径"):
        remove_tracked_file(tmp_path, "../outside.md", "unused")


def test_remove_empty_directories_keeps_root_and_nonempty_parent(
    tmp_path: Path,
) -> None:
    empty_leaf = tmp_path / "empty" / "nested"
    empty_leaf.mkdir(parents=True)
    nonempty = tmp_path / "kept"
    nonempty.mkdir()
    (nonempty / "document.md").write_text("content", encoding="utf-8")

    remove_empty_directories(tmp_path)

    assert tmp_path.exists()
    assert not empty_leaf.parent.exists()
    assert nonempty.exists()


def test_encode_stable_manifest_preserves_timestamp_until_content_changes() -> None:
    existing = {
        "format_version": 1,
        "generated_at": "2026-01-02T03:04:05+00:00",
        "documents": {"one": {"path": "one.md"}},
    }
    unchanged_payload = {"format_version": 1, "documents": {"one": {"path": "one.md"}}}

    unchanged = encode_stable_manifest(existing, unchanged_payload)
    changed = encode_stable_manifest(existing, {"format_version": 1, "documents": {}})

    assert json.loads(unchanged)["generated_at"] == existing["generated_at"]
    assert json.loads(changed)["generated_at"] != existing["generated_at"]
    assert unchanged.endswith(b"\n")
    assert changed.endswith(b"\n")
