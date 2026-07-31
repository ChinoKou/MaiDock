import asyncio
from pathlib import Path
import secrets
from time import time

from ..catalog import PublicCapabilities, PublicDriverRegistry, PublicProfileBinding
from ..config import PublicApiConfig
from ..domain import (
    Accepted,
    Canceled,
    Completed,
    Failed,
    JobStage,
    JobStatus,
    MediaCapability,
    MediaError,
    MediaOutcome,
    PreparedMediaOperation,
    PublicJsonObject,
    PublicJsonValue,
    PublicDriverOperationError,
    Running,
    VersionedOpaqueHandle,
)
from ..errors import MediaApiError
from ..storage.records import JobOutputRecord, NewJob, StoredJob
from ..storage.errors import PublicApiStorageError
from ..storage.store import PublicApiStore
from .media_probe import probe_media_metadata
from .serialization import JobSerialization

_UPLOAD_TOKEN_PREFIX = "oss://maidock-upload/"


class MediaJobEngine:
    """只负责队列、远程任务状态机和产物物化。"""

    def __init__(
        self,
        *,
        data_dir: Path,
        config: PublicApiConfig,
        registry: PublicDriverRegistry,
    ) -> None:
        self.config = config
        self.resources = config.resources
        self.registry = registry
        self.store = PublicApiStore(data_dir, self.resources)
        self._workers: dict[int, asyncio.Task[None]] = {}
        self._worker_target = self.resources.max_concurrent_jobs
        self._wake = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._stopping = False
        self._accepting = config.enabled
        self._bindings: dict[str, PublicProfileBinding] = {}
        self._recovery_queue: asyncio.Queue[StoredJob] = asyncio.Queue()

    async def start(self) -> None:
        await self.store.open()
        await self.store.cleanup_expired()
        await self._prepare_recovery()
        self._ensure_workers()

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._accepting = False
        self._stop_event.set()
        self._wake.set()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()
        await self.store.close()

    def update_config(self, config: PublicApiConfig, registry: PublicDriverRegistry) -> None:
        self.config = config
        self.resources = config.resources
        self.registry = registry
        self.store.config = config.resources
        self._accepting = config.enabled
        self._ensure_workers()

    def _ensure_workers(self) -> None:
        """把在岗 worker 数对齐到当前并发上限：补齐缺位，超编的由 worker 自行退出。"""
        self._worker_target = self.resources.max_concurrent_jobs
        for index in range(self._worker_target):
            task = self._workers.get(index)
            if task is None or task.done():
                self._workers[index] = asyncio.create_task(self._worker(index), name=f"maidock-public-api-{index}")
        self._wake.set()

    def set_accepting(self, accepting: bool) -> None:
        self._accepting = accepting

    def capabilities(self) -> PublicCapabilities:
        return self.registry.capabilities()

    async def create(
        self,
        operation: PreparedMediaOperation,
        *,
        binding: PublicProfileBinding,
        model: str,
        mode: str,
        protocol_family: str,
        upload_ids: tuple[str, ...],
        idempotency_key: str | None,
        request_digest: str | None,
    ) -> tuple[StoredJob, bool]:
        self._ensure_accepting()
        new = NewJob(
            driver_key=operation.driver_key,
            payload_version=operation.payload_version,
            profile_name=binding.name,
            credential_fingerprint=binding.credential_fingerprint,
            capability=operation.capability.value,
            model=model,
            mode=mode,
            protocol_family=protocol_family,
            prepared_payload=operation.payload,
            upload_ids=upload_ids,
            idempotency_key=idempotency_key,
            request_digest=request_digest,
        )
        try:
            job, created = await self.store.jobs.create(new)
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc
        if created:
            self._bindings[job.record.id] = binding
        else:
            self._binding_for_job(job)
        self._wake.set()
        return job, created

    async def get(self, job_id: str) -> StoredJob:
        try:
            return await self.store.jobs.get(job_id)
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc

    async def cancel(self, job_id: str) -> StoredJob:
        try:
            job = await self.store.jobs.request_cancel(job_id)
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc
        self._wake.set()
        return job

    async def delete(self, job_id: str) -> None:
        try:
            await self.store.jobs.delete(job_id)
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc

    async def _worker(self, index: int) -> None:
        while not self._stopping and index < self._worker_target:
            recovered = True
            try:
                job = self._recovery_queue.get_nowait()
            except asyncio.QueueEmpty:
                recovered = False
                job = await self.store.jobs.claim_next()
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
                continue
            try:
                if recovered:
                    await self._resume(job)
                else:
                    await self._process(job)
            except asyncio.CancelledError:
                raise
            except MediaApiError as exc:
                await self._mark_error(
                    job.record.id,
                    MediaError(exc.code, str(exc), retryable=exc.retryable, uncertain=exc.uncertain),
                )
            except PublicDriverOperationError as exc:
                await self._mark_error(job.record.id, exc.error)
            except Exception as exc:
                await self._mark_error(job.record.id, MediaError("INTERNAL_ERROR", str(exc)))

    async def _prepare_recovery(self) -> None:
        for job in await self.store.jobs.list_recoverable():
            if job.record.status == JobStatus.QUEUED.value:
                continue
            try:
                self._binding_for_job(job)
            except MediaApiError as exc:
                await self._mark_error(job.record.id, MediaError(exc.code, str(exc)))
                continue
            if job.record.stage == JobStage.SUBMITTING.value and job.record.remote_handle is None:
                await self._mark_error(
                    job.record.id,
                    MediaError(
                        "EXECUTION_UNCERTAIN",
                        "上次提交在取得远程句柄前中断，无法确认供应商是否已接收",
                        uncertain=True,
                    ),
                )
                continue
            await self._recovery_queue.put(job)

    async def _resume(self, job: StoredJob) -> None:
        binding = self._binding_for_job(job)
        if job.record.stage == JobStage.MATERIALIZING.value:
            await self._materialize_saved_outputs(
                job.record.id,
                binding,
                usage=job.record.usage,
                warnings=job.record.warnings,
                failed_output_count=job.record.failed_output_count,
            )
            return
        if job.record.remote_handle is None:
            raise MediaApiError("STORE_CORRUPT", f"运行中作业 {job.record.id} 缺少远程句柄")
        handle = JobSerialization.handle_from_json(job.record.remote_handle)
        delay = max((job.record.next_poll_at or time()) - time(), 0.0)
        await self._poll(job.record.id, binding, handle, delay)

    async def _process(self, job: StoredJob) -> None:
        binding = self._binding_for_job(job)
        operation = PreparedMediaOperation(
            driver_key=job.record.driver_key,
            payload_version=job.record.payload_version,
            profile_name=job.record.profile_name,
            capability=MediaCapability(job.record.capability),
            operation_type=job.record.protocol_family,
            payload=job.record.prepared_payload,
        )
        operation = await self._replace_upload_tokens(operation, binding)
        outcome = await binding.driver.submit(operation)
        await self._consume(job.record.id, binding, outcome)

    async def _replace_upload_tokens(
        self,
        operation: PreparedMediaOperation,
        binding: PublicProfileBinding,
    ) -> PreparedMediaOperation:
        upload_ids = sorted(_collect_upload_tokens(operation.payload))
        if not upload_ids:
            return operation
        model = _required_string(operation.payload, "model")
        replacements: dict[str, str] = {}
        for upload_id in upload_ids:
            try:
                upload = await self.store.uploads.get(upload_id)
                replacements[upload_id] = await binding.driver.upload_file(
                    binding.name,
                    model=model,
                    path=Path(upload.path),
                    media_type=upload.media_type,
                )
            except PublicApiStorageError as exc:
                raise MediaApiError(exc.code, str(exc)) from exc
        return PreparedMediaOperation(
            driver_key=operation.driver_key,
            payload_version=operation.payload_version,
            profile_name=operation.profile_name,
            capability=operation.capability,
            operation_type=operation.operation_type,
            payload=_replace_upload_tokens(operation.payload, replacements),
        )

    async def _consume(
        self,
        job_id: str,
        binding: PublicProfileBinding,
        outcome: MediaOutcome,
    ) -> None:
        if isinstance(outcome, Failed):
            await self._mark_error(job_id, outcome.error)
            return
        if isinstance(outcome, Canceled):
            await self.store.jobs.mark_terminal(job_id, JobStatus.CANCELED, provider_request_id=outcome.request_id)
            return
        if isinstance(outcome, Completed):
            await self._save_completed(job_id, binding, outcome)
            return
        await self.store.jobs.set_stage(
            job_id,
            JobStage.POLLING,
            remote_handle=JobSerialization.handle_to_json(outcome.remote_handle),
            next_poll_at=time() + outcome.next_poll_after_seconds,
            provider_request_id=outcome.request_id,
        )
        await self._poll(job_id, binding, outcome.remote_handle, outcome.next_poll_after_seconds)

    async def _poll(
        self,
        job_id: str,
        binding: PublicProfileBinding,
        handle: VersionedOpaqueHandle,
        delay: float,
    ) -> None:
        next_delay = max(delay, 0.1)
        while True:
            job = await self.get(job_id)
            if time() - job.record.created_at >= self.resources.max_tracking_hours * 3600:
                await self.store.jobs.mark_terminal(job_id, JobStatus.EXPIRED, error_code="TASK_TRACKING_EXPIRED")
                return
            if next_delay:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=next_delay)
                except TimeoutError:
                    pass
                if self._stopping:
                    return
            outcome = await (
                binding.driver.cancel(handle) if job.record.cancel_requested else binding.driver.poll(handle)
            )
            if isinstance(outcome, (Accepted, Running)):
                handle = outcome.remote_handle
                next_delay = max(outcome.next_poll_after_seconds, 0.1)
                await self.store.jobs.set_stage(
                    job_id,
                    JobStage.POLLING,
                    remote_handle=JobSerialization.handle_to_json(handle),
                    next_poll_at=time() + next_delay,
                    provider_request_id=outcome.request_id,
                )
                continue
            await self._consume(job_id, binding, outcome)
            return

    async def _save_completed(
        self,
        job_id: str,
        binding: PublicProfileBinding,
        outcome: Completed,
    ) -> None:
        outputs = tuple(
            JobOutputRecord(
                job_id=job_id,
                ordinal=ordinal,
                kind=output.kind,
                text=output.text,
                url=output.url,
                media_type=output.media_type,
            )
            for ordinal, output in enumerate(outcome.outputs)
        )
        await self.store.jobs.save_remote_outputs(
            job_id,
            outputs,
            usage=outcome.usage,
            warnings=outcome.warnings,
            failed_output_count=outcome.failed_output_count,
            provider_request_id=outcome.request_id,
        )
        await self._materialize_saved_outputs(
            job_id,
            binding,
            usage=outcome.usage,
            warnings=outcome.warnings,
            failed_output_count=outcome.failed_output_count,
        )

    async def _materialize_saved_outputs(
        self,
        job_id: str,
        binding: PublicProfileBinding,
        *,
        usage: PublicJsonObject,
        warnings: tuple[str, ...],
        failed_output_count: int,
    ) -> None:
        job = await self.get(job_id)
        accumulated_warnings = list(warnings)
        failed_count = failed_output_count
        materialized_count = sum(output.artifact_id is not None for output in job.outputs)
        for output in job.outputs:
            if output.kind == "text" or output.artifact_id is not None:
                continue
            if not output.url:
                failed_count += 1
                accumulated_warnings.append(f"输出 {output.ordinal} 缺少媒体 URL")
                continue
            staging = None
            try:
                staging = await self.store.artifacts.begin(
                    job_id,
                    output.ordinal,
                    self.resources.max_artifact_mb * 1024 * 1024,
                )
                materialized = await binding.driver.materialize(
                    binding.name,
                    url=output.url,
                    destination=Path(staging.path),
                    max_bytes=self.resources.max_artifact_mb * 1024 * 1024,
                )
                media_type = materialized.media_type or output.media_type or "application/octet-stream"
                probed = probe_media_metadata(Path(staging.path), media_type)
                await self.store.artifacts.commit(
                    staging,
                    media_type=media_type,
                    size=materialized.size,
                    sha256_hex=materialized.sha256,
                    width=probed.width,
                    height=probed.height,
                    duration_seconds=probed.duration_seconds,
                )
                materialized_count += 1
            except Exception as exc:
                failed_count += 1
                accumulated_warnings.append(f"输出 {output.ordinal} 落盘失败: {exc}")
                if staging is not None:
                    await self.store.artifacts.abort(staging)
        persisted_usage = dict(usage)
        persisted_usage["output_count"] = materialized_count
        await self.store.jobs.update_summary(
            job_id,
            usage=persisted_usage,
            warnings=tuple(accumulated_warnings),
            failed_output_count=failed_count,
        )
        status = (
            JobStatus.SUCCEEDED
            if materialized_count or any(item.kind == "text" for item in job.outputs)
            else JobStatus.FAILED
        )
        await self.store.jobs.mark_terminal(
            job_id, status, error_code=None if status is JobStatus.SUCCEEDED else "NO_MEDIA_OUTPUT"
        )

    async def _mark_error(self, job_id: str, error: MediaError) -> None:
        await self.store.jobs.mark_terminal(
            job_id,
            JobStatus.FAILED,
            error_code=error.code,
            error_retryable=error.retryable,
            error_uncertain=error.uncertain,
            provider_request_id=error.request_id,
        )

    def _binding_for_job(self, job: StoredJob) -> PublicProfileBinding:
        cached = self._bindings.get(job.record.id)
        if cached is not None:
            return cached
        binding = self.registry.profile(job.record.profile_name)
        if (
            binding is None
            or binding.driver_key != job.record.driver_key
            or not secrets.compare_digest(
                binding.credential_fingerprint,
                job.record.credential_fingerprint,
            )
        ):
            raise MediaApiError("PROFILE_CHANGED", f"作业 {job.record.id} 所需 Profile 已变化")
        self._bindings[job.record.id] = binding
        return binding

    def _ensure_accepting(self) -> None:
        if not self._accepting:
            raise MediaApiError("MEDIA_API_DISABLED", "MaiDock Public API 当前未启用")


def _collect_upload_tokens(value: PublicJsonValue) -> set[str]:
    if isinstance(value, str) and value.startswith(_UPLOAD_TOKEN_PREFIX):
        return {value.removeprefix(_UPLOAD_TOKEN_PREFIX)}
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _collect_upload_tokens(nested)}
    if isinstance(value, list):
        return {item for nested in value for item in _collect_upload_tokens(nested)}
    return set()


def _replace_upload_tokens(value: PublicJsonValue, replacements: dict[str, str]) -> PublicJsonObject:
    replaced = _replace_value(value, replacements)
    if not isinstance(replaced, dict):
        raise MediaApiError("STORE_CORRUPT", "prepared payload 必须是 object")
    return replaced


def _replace_value(value: PublicJsonValue, replacements: dict[str, str]) -> PublicJsonValue:
    if isinstance(value, str) and value.startswith(_UPLOAD_TOKEN_PREFIX):
        return replacements.get(value.removeprefix(_UPLOAD_TOKEN_PREFIX), value)
    if isinstance(value, dict):
        return {key: _replace_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_value(item, replacements) for item in value]
    return value


def _required_string(payload: PublicJsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise MediaApiError("STORE_CORRUPT", f"prepared payload 缺少 {name}")
    return value
