import asyncio
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
import pytest

from src.public_api.application.job_engine import MediaJobEngine
from src.public_api.catalog import PublicDriverRegistry, PublicProfileBinding
from src.public_api.config import PublicApiConfig, PublicApiResourceConfig
from src.public_api.domain import (
    Canceled,
    Completed,
    JobStage,
    JobStatus,
    MediaCapability,
    MediaOutcome,
    MediaOutput,
    MediaRequest,
    MaterializedArtifact,
    ModelCapability,
    PreparedMediaOperation,
    VersionedOpaqueHandle,
)
from src.public_api.storage.records import NewJob


class FakePublicDriver:
    driver_key = "fake.media.v1"

    def __init__(self) -> None:
        self.submit_outcome: MediaOutcome = Completed(outputs=(MediaOutput(kind="text", text="done"),))
        self.poll_outcomes: list[MediaOutcome] = []
        self.cancel_count = 0
        self.submit_entered = asyncio.Event()
        self.submit_release: asyncio.Event | None = None
        self.materialized_urls: list[str] = []
        # 置位后 materialize 落盘一张真实 PNG，用于验证引擎侧的元数据探测。
        self.materialize_image_size: tuple[int, int] | None = None

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return (
            ModelCapability(
                model="fake-image",
                capability=MediaCapability.IMAGE_GENERATION,
                modes=("text_to_image",),
                protocol_families=("fake_generation",),
                max_outputs=1,
            ),
        )

    def prepare(self, profile_name: str, request: MediaRequest) -> PreparedMediaOperation:
        return PreparedMediaOperation(
            driver_key=self.driver_key,
            payload_version=1,
            profile_name=profile_name,
            capability=request.capability,
            operation_type="fake_generation",
            payload={"model": "fake-image"},
        )

    async def submit(self, operation: PreparedMediaOperation) -> MediaOutcome:
        del operation
        self.submit_entered.set()
        if self.submit_release is not None:
            await self.submit_release.wait()
        return self.submit_outcome

    async def poll(self, handle: VersionedOpaqueHandle) -> MediaOutcome:
        del handle
        return self.poll_outcomes.pop(0)

    async def cancel(self, handle: VersionedOpaqueHandle) -> Canceled:
        del handle
        self.cancel_count += 1
        return Canceled(request_id="cancel-request")

    async def upload_file(self, profile_name: str, *, model: str, path: Path, media_type: str) -> str:
        del profile_name, model, path, media_type
        return "oss://fake/input"

    async def materialize(
        self,
        profile_name: str,
        *,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> MaterializedArtifact:
        del profile_name, max_bytes
        self.materialized_urls.append(url)
        data = b"artifact" if self.materialize_image_size is None else _png_bytes(self.materialize_image_size)
        destination.write_bytes(data)
        return MaterializedArtifact(
            path=destination,
            size=len(data),
            sha256=sha256(data).hexdigest(),
            media_type="application/octet-stream",
        )


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = BytesIO()
    with PILImage.new("RGB", size, color=(200, 100, 50)) as image:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _config(*, enabled: bool = True, workers: int = 1) -> PublicApiConfig:
    return PublicApiConfig(
        enabled=enabled,
        default_image_profile="main",
        default_video_profile="main",
        resources=PublicApiResourceConfig(max_concurrent_jobs=workers),
    )


def _registry(driver: FakePublicDriver, fingerprint: str = "fingerprint") -> PublicDriverRegistry:
    return PublicDriverRegistry(
        profiles=(PublicProfileBinding("main", "fake", driver.driver_key, fingerprint, driver),),
        default_image_profile="main",
        default_video_profile="main",
    )


async def _wait_terminal(engine: MediaJobEngine, job_id: str) -> str:
    for _ in range(200):
        job = await engine.get(job_id)
        if job.record.status in {"succeeded", "failed", "canceled", "expired"}:
            return job.record.status
        await asyncio.sleep(0.01)
    raise AssertionError("作业没有进入终态")


async def _submit(engine: MediaJobEngine, driver: FakePublicDriver) -> str:
    binding = _registry(driver).resolve(MediaCapability.IMAGE_GENERATION, None)
    operation = driver.prepare(
        binding.name,
        MediaRequest(capability=MediaCapability.IMAGE_GENERATION, mode="text_to_image"),
    )
    job, created = await engine.create(
        operation,
        binding=binding,
        model="fake-image",
        mode="text_to_image",
        protocol_family="fake_generation",
        upload_ids=(),
        idempotency_key=None,
        request_digest=None,
    )
    assert created is True
    return job.record.id


@pytest.mark.asyncio
async def test_engine_persists_partial_media_outputs_and_uses_artifact_repository(tmp_path: Path) -> None:
    driver = FakePublicDriver()
    driver.submit_outcome = Completed(
        outputs=(
            MediaOutput(kind="text", text="说明"),
            MediaOutput(kind="media", url="https://example.com/result.bin"),
        ),
        usage={"generated": 1},
    )
    engine = MediaJobEngine(data_dir=tmp_path / "data", config=_config(), registry=_registry(driver))
    await engine.start()
    try:
        job_id = await _submit(engine, driver)
        assert await _wait_terminal(engine, job_id) == "succeeded"
        job = await engine.get(job_id)
        assert [output.kind for output in job.outputs] == ["text", "media"]
        assert job.outputs[1].artifact_id is not None
        assert job.record.usage["output_count"] == 1
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_stop_waits_for_current_submission_checkpoint(tmp_path: Path) -> None:
    driver = FakePublicDriver()
    driver.submit_release = asyncio.Event()
    engine = MediaJobEngine(data_dir=tmp_path / "data", config=_config(), registry=_registry(driver))
    await engine.start()
    job_id = await _submit(engine, driver)
    await driver.submit_entered.wait()

    stop_task = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    driver.submit_release.set()
    await stop_task

    await engine.store.open()
    try:
        completed = await engine.get(job_id)
        assert completed.record.status == JobStatus.SUCCEEDED.value
        assert completed.record.stage == JobStage.MATERIALIZING.value
    finally:
        await engine.store.close()


@pytest.mark.asyncio
async def test_engine_persists_probed_image_dimensions(tmp_path: Path) -> None:
    driver = FakePublicDriver()
    driver.materialize_image_size = (23, 17)
    driver.submit_outcome = Completed(outputs=(MediaOutput(kind="media", url="https://example.com/result.png"),))
    engine = MediaJobEngine(data_dir=tmp_path / "data", config=_config(), registry=_registry(driver))
    await engine.start()
    try:
        job_id = await _submit(engine, driver)
        assert await _wait_terminal(engine, job_id) == "succeeded"
        job = await engine.get(job_id)
        artifact_id = job.outputs[0].artifact_id
        assert artifact_id is not None
        artifact = await engine.store.artifacts.get(artifact_id)
        # driver 报的是通用二进制类型，探测仍应生效；时长对图片保持 None。
        assert artifact.media_type == "application/octet-stream"
        assert (artifact.width, artifact.height) == (23, 17)
        assert artifact.duration_seconds is None
    finally:
        await engine.stop()


async def _wait_live_workers(engine: MediaJobEngine, expected: int) -> set[int]:
    for _ in range(200):
        live = {index for index, task in engine._workers.items() if not task.done()}
        if len(live) == expected:
            return live
        await asyncio.sleep(0.01)
    raise AssertionError(f"在岗 worker 数没有收敛到 {expected}")


@pytest.mark.asyncio
async def test_engine_scales_workers_down_then_up_without_losing_jobs(tmp_path: Path) -> None:
    driver = FakePublicDriver()
    engine = MediaJobEngine(data_dir=tmp_path / "data", config=_config(workers=4), registry=_registry(driver))
    await engine.start()
    try:
        assert await _wait_live_workers(engine, 4) == {0, 1, 2, 3}

        engine.update_config(_config(workers=1), _registry(driver))
        assert await _wait_live_workers(engine, 1) == {0}

        job_ids = [await _submit(engine, driver) for _ in range(3)]
        for job_id in job_ids:
            assert await _wait_terminal(engine, job_id) == "succeeded"

        engine.update_config(_config(workers=3), _registry(driver))
        assert await _wait_live_workers(engine, 3) == {0, 1, 2}
        assert max(engine._workers) == 3
    finally:
        await engine.stop()
    assert not engine._workers


@pytest.mark.asyncio
async def test_engine_restarts_polling_jobs_but_never_resubmits_uncertain_submit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    driver = FakePublicDriver()
    config = _config(enabled=False)
    original = MediaJobEngine(data_dir=data_dir, config=config, registry=_registry(driver))
    await original.store.open()
    uncertain_job, _ = await original.store.jobs.create(
        NewJob(
            driver_key=driver.driver_key,
            payload_version=1,
            profile_name="main",
            credential_fingerprint="fingerprint",
            capability="image_generation",
            model="fake-image",
            mode="text_to_image",
            protocol_family="fake_generation",
            prepared_payload={"model": "fake-image"},
        )
    )
    await original.store.jobs.claim_next()
    await original.store.close()

    recovered_driver = FakePublicDriver()
    recovered = MediaJobEngine(data_dir=data_dir, config=config, registry=_registry(recovered_driver))
    await recovered.start()
    try:
        uncertain = await recovered.get(uncertain_job.record.id)
        assert uncertain.record.status == "failed"
        assert uncertain.record.error_code == "EXECUTION_UNCERTAIN"
        assert uncertain.record.error_uncertain is True
        assert not recovered_driver.submit_entered.is_set()
    finally:
        await recovered.stop()


@pytest.mark.asyncio
async def test_engine_recovery_uses_remote_handle_and_rejects_changed_profile(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    driver = FakePublicDriver()
    original = MediaJobEngine(data_dir=data_dir, config=_config(enabled=False), registry=_registry(driver))
    await original.store.open()
    job, _ = await original.store.jobs.create(
        NewJob(
            driver_key=driver.driver_key,
            payload_version=1,
            profile_name="main",
            credential_fingerprint="fingerprint",
            capability="image_generation",
            model="fake-image",
            mode="text_to_image",
            protocol_family="fake_generation",
            prepared_payload={"model": "fake-image"},
        )
    )
    await original.store.jobs.claim_next()
    await original.store.jobs.set_stage(
        job.record.id,
        JobStage.POLLING,
        remote_handle={"driver_key": driver.driver_key, "payload_version": 1, "payload": {"task_id": "task-1"}},
    )
    await original.store.close()

    driver.poll_outcomes = [Completed(outputs=(MediaOutput(kind="text", text="resumed"),))]
    recovered = MediaJobEngine(data_dir=data_dir, config=_config(), registry=_registry(driver))
    await recovered.start()
    try:
        assert await _wait_terminal(recovered, job.record.id) == "succeeded"
        assert driver.submit_entered.is_set() is False
    finally:
        await recovered.stop()

    changed_data = tmp_path / "changed"
    changed_original = MediaJobEngine(
        data_dir=changed_data,
        config=_config(enabled=False),
        registry=_registry(FakePublicDriver(), "fingerprint"),
    )
    await changed_original.store.open()
    changed_job, _ = await changed_original.store.jobs.create(
        NewJob(
            driver_key=driver.driver_key,
            payload_version=1,
            profile_name="main",
            credential_fingerprint="fingerprint",
            capability="image_generation",
            model="fake-image",
            mode="text_to_image",
            protocol_family="fake_generation",
            prepared_payload={"model": "fake-image"},
        )
    )
    await changed_original.store.jobs.claim_next()
    await changed_original.store.jobs.set_stage(
        changed_job.record.id,
        JobStage.POLLING,
        remote_handle={"driver_key": driver.driver_key, "payload_version": 1, "payload": {"task_id": "changed"}},
    )
    await changed_original.store.close()

    changed = MediaJobEngine(data_dir=changed_data, config=_config(), registry=_registry(FakePublicDriver(), "changed"))
    await changed.start()
    try:
        changed_record = await changed.get(changed_job.record.id)
        assert changed_record.record.status == "failed"
        assert changed_record.record.error_code == "PROFILE_CHANGED"
    finally:
        await changed.stop()
