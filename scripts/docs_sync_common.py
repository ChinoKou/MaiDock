"""供应商文档同步脚本共享的文件系统操作。"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_DOCS_ROOT = PROJECT_ROOT / "docs" / "provider_docs"


class DocsSyncError(RuntimeError):
    """表示文档同步脚本的本地文件操作不安全或失败。"""


@dataclass(slots=True)
class SyncStats:
    """统计一次文档同步产生的文件变化。"""

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    preserved: int = 0


def project_output_path(output: Path) -> Path:
    """解析输出目录，并确保它位于 MaiDock 仓库内。"""
    project_root = PROJECT_ROOT.resolve()
    resolved = output.resolve()
    if resolved == project_root or not resolved.is_relative_to(project_root):
        raise DocsSyncError(f"输出目录必须位于 MaiDock 仓库内：{output}")
    return resolved


def atomic_write(path: Path, content: bytes) -> None:
    """先写同目录临时文件，再以原子替换提交内容。"""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise DocsSyncError(f"写入文件失败：{path}：{exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def remove_tracked_file(output_root: Path, relative_path: str, expected_digest: str) -> bool:
    """只删除清单内且内容未被本地修改的普通文件。"""
    root = output_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise DocsSyncError(f"清单包含越界路径：{relative_path}")
    if not candidate.exists():
        return True
    if not candidate.is_file():
        return False
    if sha256(candidate.read_bytes()).hexdigest() != expected_digest:
        return False
    candidate.unlink()
    return True


def remove_empty_directories(output_root: Path) -> None:
    """清除同步后产生的空目录，但保留输出根目录。"""
    directories = sorted(
        (path for path in output_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def encode_stable_manifest(
    existing_manifest: Mapping[str, object],
    payload_without_generated_at: Mapping[str, object],
) -> bytes:
    """仅在清单实质内容变化时更新生成时间，并编码为稳定 JSON。"""
    previous_payload = dict(existing_manifest)
    previous_generated_at = previous_payload.pop("generated_at", None)
    current_payload = dict(payload_without_generated_at)
    if previous_payload == current_payload and isinstance(previous_generated_at, str):
        generated_at = previous_generated_at
    else:
        generated_at = datetime.now(UTC).isoformat()
    current_payload["generated_at"] = generated_at
    return (json.dumps(current_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
