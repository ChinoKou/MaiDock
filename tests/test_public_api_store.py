from hashlib import sha256
from pathlib import Path

import pytest

from src.public_api.config import PublicApiResourceConfig
from src.public_api.domain import JobStatus
from src.public_api.storage.records import NewJob
from src.public_api.storage.errors import PublicApiStorageError
from src.public_api.storage.store import PublicApiStore


def _new_job(
    *,
    key: str | None = None,
    digest: str | None = None,
    upload_ids: tuple[str, ...] = (),
) -> NewJob:
    return NewJob(
        driver_key="fake.v1",
        payload_version=1,
        profile_name="main",
        credential_fingerprint="fingerprint",
        capability="image_generation",
        model="fake-image",
        mode="text_to_image",
        protocol_family="fake_generation",
        prepared_payload={"model": "fake-image"},
        upload_ids=upload_ids,
        idempotency_key=key,
        request_digest=digest,
    )


def _store(tmp_path: Path, **values: int) -> PublicApiStore:
    return PublicApiStore(
        tmp_path / "data",
        PublicApiResourceConfig.model_validate(values),
    )


@pytest.mark.asyncio
async def test_upload_repository_enforces_order_truncation_and_hash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.open()
    try:
        content = b"abcdef"
        upload = await store.uploads.create(
            media_type="image/png",
            size=len(content),
            expected_sha256=sha256(content).hexdigest(),
            file_name="source.png",
        )
        with pytest.raises(PublicApiStorageError, match="offset") as exc_info:
            await store.uploads.write_chunk(upload.id, offset=1, data=b"abc")
        assert exc_info.value.code == "UPLOAD_OFFSET_MISMATCH"

        await store.uploads.write_chunk(upload.id, offset=0, data=b"abc")
        path = Path((await store.uploads.get(upload.id)).path)
        path.write_bytes(b"abcuncommitted-tail")
        await store.uploads.write_chunk(upload.id, offset=3, data=b"def")
        completed = await store.uploads.complete(upload.id)
        assert Path(completed.path).read_bytes() == content
        assert completed.status == "complete"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_job_repository_handles_idempotency_queue_and_cancellation(tmp_path: Path) -> None:
    store = _store(tmp_path, max_queued_jobs=2)
    await store.open()
    try:
        first, created = await store.jobs.create(_new_job(key="same", digest="one"))
        repeated, repeated_created = await store.jobs.create(_new_job(key="same", digest="one"))
        assert created is True
        assert repeated_created is False
        assert repeated.record.id == first.record.id

        with pytest.raises(PublicApiStorageError) as conflict:
            await store.jobs.create(_new_job(key="same", digest="two"))
        assert conflict.value.code == "IDEMPOTENCY_CONFLICT"
        await store.jobs.create(_new_job(key="second", digest="second"))
        with pytest.raises(PublicApiStorageError) as full:
            await store.jobs.create(_new_job(key="third", digest="third"))
        assert full.value.code == "QUEUE_FULL"

        canceled = await store.jobs.request_cancel(first.record.id)
        assert canceled.record.status == JobStatus.CANCELED.value
        await store.jobs.delete(first.record.id)
        with pytest.raises(PublicApiStorageError) as missing:
            await store.jobs.get(first.record.id)
        assert missing.value.code == "JOB_NOT_FOUND"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_artifact_repository_supports_random_access_and_atomic_commit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.open()
    try:
        job, _ = await store.jobs.create(_new_job())
        staging = await store.artifacts.begin(job.record.id, 0, 1024)
        Path(staging.path).write_bytes(b"0123456789")
        artifact = await store.artifacts.commit(
            staging,
            media_type="video/mp4",
            size=10,
            sha256_hex=sha256(b"0123456789").hexdigest(),
        )
        record, chunk = await store.artifacts.read(artifact.id, offset=4, length=3)
        assert record.id == artifact.id
        assert chunk == b"456"
        with pytest.raises(PublicApiStorageError) as invalid_offset:
            await store.artifacts.read(artifact.id, offset=11, length=1)
        assert invalid_offset.value.code == "INVALID_OFFSET"

        await store.jobs.mark_terminal(job.record.id, JobStatus.SUCCEEDED)
        artifact_path = Path(artifact.path)
        await store.jobs.delete(job.record.id)
        assert not artifact_path.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_recovery_cleanup_removes_expired_staging(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.open()
    try:
        job, _ = await store.jobs.create(_new_job())
        staging = await store.artifacts.begin(job.record.id, 0, 1024)
        Path(staging.path).write_bytes(b"partial")
        await store.cleanup_expired(now=staging.created_at + 86401)
        assert not Path(staging.path).exists()
        assert store.database.path.name == "maidock_public_api.sqlite3"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cleanup_keeps_expired_uploads_referenced_by_jobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.open()
    try:
        content = b"source"
        upload = await store.uploads.create(
            media_type="image/png",
            size=len(content),
            expected_sha256=sha256(content).hexdigest(),
            file_name="source.png",
            now=0,
        )
        await store.uploads.write_chunk(upload.id, offset=0, data=content, now=0)
        complete = await store.uploads.complete(upload.id, now=0)
        await store.jobs.create(_new_job(upload_ids=(complete.id,)))

        await store.cleanup_expired(now=8 * 86400)

        retained = await store.uploads.get(complete.id)
        assert retained.status == "complete"
        assert Path(retained.path).exists()
    finally:
        await store.close()
