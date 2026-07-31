from datetime import UTC, datetime
from hashlib import sha256
import json

from ..api.requests import (
    CreateJobRequest,
    CreateUploadRequest,
    JobIdRequest,
    OneShotUploadRequest,
    ReadArtifactRequest,
    UploadIdRequest,
    WriteUploadChunkRequest,
)
from ..api.responses import (
    ArtifactChunkData,
    ArtifactOutputData,
    CapabilityData,
    DeletedJobData,
    DeletedUploadData,
    JobData,
    JobErrorData,
    ModelCapabilityData,
    ProfileData,
    TextOutputData,
    UploadData,
)
from ..domain import (
    JobStatus,
    MediaCapability,
    MediaInput,
    MediaRequest,
    MediaSource,
    PublicJsonObject,
    UploadStatus,
)
from ..errors import MediaApiError
from ..storage.errors import PublicApiStorageError
from ..storage.records import StoredJob, UploadRecord
from .job_engine import MediaJobEngine


class PublicApiService:
    """把严格 RPC Command 转成领域命令，并生成严格响应数据。"""

    def __init__(self, engine: MediaJobEngine) -> None:
        self.engine = engine

    async def capabilities(self) -> CapabilityData:
        snapshot = self.engine.capabilities()
        return CapabilityData(
            models=tuple(
                ModelCapabilityData(
                    model=item.model,
                    capability=item.capability,
                    modes=item.modes,
                    protocol_families=item.protocol_families,
                    max_outputs=item.max_outputs,
                )
                for item in snapshot.models
            ),
            profiles=tuple(
                ProfileData(name=item.name, provider=item.provider_key, driver=item.driver_key)
                for item in snapshot.profiles
            ),
            default_image_profile=snapshot.default_image_profile,
            default_video_profile=snapshot.default_video_profile,
        )

    async def create_job(self, request: CreateJobRequest) -> JobData:
        registry = self.engine.registry
        try:
            binding = registry.resolve(request.capability, request.profile)
            media_request = _media_request(request)
            operation = binding.driver.prepare(binding.name, media_request)
        except (KeyError, ValueError) as exc:
            raise MediaApiError(_profile_error(exc), str(exc)) from exc
        upload_ids = tuple(
            source.upload_id
            for item in request.inputs
            for source in (item.source, item.reference_voice)
            if source is not None and source.upload_id is not None
        )
        model = request.model or _operation_model(operation.payload)
        digest = _request_digest(request)
        _job, _created = await self.engine.create(
            operation,
            binding=binding,
            model=model,
            mode=request.mode,
            protocol_family=operation.operation_type,
            upload_ids=tuple(dict.fromkeys(upload_ids)),
            idempotency_key=request.idempotency_key,
            request_digest=digest,
        )
        return await self.job_data(_job)

    async def get_job(self, request: JobIdRequest) -> JobData:
        return await self.job_data(await self.engine.get(request.job_id))

    async def cancel_job(self, request: JobIdRequest) -> JobData:
        return await self.job_data(await self.engine.cancel(request.job_id))

    async def delete_job(self, request: JobIdRequest) -> DeletedJobData:
        await self.engine.delete(request.job_id)
        return DeletedJobData(job_id=request.job_id, deleted=True)

    async def create_upload(self, request: CreateUploadRequest) -> UploadData:
        try:
            record = await self.engine.store.uploads.create(
                media_type=request.media_type,
                size=request.size,
                expected_sha256=request.sha256,
                file_name=request.file_name,
            )
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc
        return _upload_data(record)

    async def one_shot_upload(self, request: OneShotUploadRequest) -> UploadData:
        digest = sha256(request.data).hexdigest()
        if request.sha256 is not None and request.sha256 != digest:
            raise MediaApiError("UPLOAD_SHA256_MISMATCH", "上传内容 SHA-256 不匹配")
        created = await self.create_upload(
            CreateUploadRequest(
                media_type=request.media_type,
                size=len(request.data),
                sha256=digest,
                file_name=request.file_name,
            )
        )
        try:
            await self.engine.store.uploads.write_chunk(created.upload_id, offset=0, data=request.data)
            return _upload_data(await self.engine.store.uploads.complete(created.upload_id))
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc

    async def get_upload(self, request: UploadIdRequest) -> UploadData:
        try:
            return _upload_data(await self.engine.store.uploads.get(request.upload_id))
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc

    async def write_upload_chunk(self, request: WriteUploadChunkRequest) -> UploadData:
        try:
            record = await self.engine.store.uploads.write_chunk(
                request.upload_id,
                offset=request.offset,
                data=request.data,
            )
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc
        return _upload_data(record)

    async def complete_upload(self, request: UploadIdRequest) -> UploadData:
        try:
            return _upload_data(await self.engine.store.uploads.complete(request.upload_id))
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc

    async def delete_upload(self, request: UploadIdRequest) -> DeletedUploadData:
        try:
            await self.engine.store.uploads.delete(request.upload_id)
        except PublicApiStorageError as exc:
            raise MediaApiError(exc.code, str(exc)) from exc
        return DeletedUploadData(upload_id=request.upload_id, deleted=True)

    async def read_artifact(self, request: ReadArtifactRequest) -> ArtifactChunkData:
        try:
            artifact, chunk = await self.engine.store.artifacts.read(
                request.artifact_id,
                offset=request.offset,
                length=request.length,
            )
        except (PublicApiStorageError, ValueError) as exc:
            code = exc.code if isinstance(exc, PublicApiStorageError) else "INVALID_OFFSET"
            raise MediaApiError(code, str(exc)) from exc
        next_offset = request.offset + len(chunk)
        return ArtifactChunkData(
            artifact_id=artifact.id,
            offset=request.offset,
            next_offset=next_offset,
            eof=next_offset >= artifact.size,
            chunk=chunk,
            media_type=artifact.media_type,
            size=artifact.size,
            sha256=artifact.sha256,
        )

    async def job_data(self, job: StoredJob) -> JobData:
        outputs = []
        for output in job.outputs:
            if output.kind == "text" and output.text is not None:
                outputs.append(TextOutputData(text=output.text))
            elif output.artifact_id is not None:
                artifact = await self.engine.store.artifacts.get(output.artifact_id)
                outputs.append(
                    ArtifactOutputData(
                        artifact_id=artifact.id,
                        media_type=artifact.media_type,
                        size=artifact.size,
                        sha256=artifact.sha256,
                        width=artifact.width,
                        height=artifact.height,
                        duration_seconds=artifact.duration_seconds,
                        expires_at=_rfc3339(artifact.expires_at),
                    )
                )
        error = None
        if job.record.error_code is not None:
            error = JobErrorData(
                code=job.record.error_code,
                retryable=job.record.error_retryable,
                uncertain=job.record.error_uncertain,
                provider_request_id=job.record.provider_request_id,
            )
        provider_binding = self.engine.registry.profile(job.record.profile_name)
        return JobData(
            job_id=job.record.id,
            status=JobStatus(job.record.status),
            capability=MediaCapability(job.record.capability),
            profile=job.record.profile_name,
            provider=provider_binding.provider_key if provider_binding is not None else "unknown",
            model=job.record.model,
            mode=job.record.mode,
            protocol_family=job.record.protocol_family,
            created_at=_rfc3339(job.record.created_at),
            updated_at=_rfc3339(job.record.updated_at),
            outputs=tuple(outputs),
            warnings=job.record.warnings,
            failed_output_count=job.record.failed_output_count,
            usage=job.record.usage,
            error=error,
        )


def _media_request(request: CreateJobRequest) -> MediaRequest:
    return MediaRequest(
        capability=request.capability,
        mode=request.mode,
        prompt=request.prompt or "",
        negative_prompt=request.negative_prompt or "",
        model=request.model,
        protocol_family=request.protocol_family,
        inputs=tuple(
            MediaInput(
                role=item.role,
                source=MediaSource(item.source.url or f"oss://maidock-upload/{item.source.upload_id}"),
                reference_voice=(
                    MediaSource(item.reference_voice.url or f"oss://maidock-upload/{item.reference_voice.upload_id}")
                    if item.reference_voice is not None
                    else None
                ),
            )
            for item in request.inputs
        ),
        parameters=request.parameters,
    )


def _operation_model(payload: PublicJsonObject) -> str:
    model = payload.get("model")
    if not isinstance(model, str):
        raise MediaApiError("STORE_CORRUPT", "prepared payload 缺少 model")
    return model


def _request_digest(request: CreateJobRequest) -> str:
    canonical = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _upload_data(record: UploadRecord) -> UploadData:
    return UploadData(
        upload_id=record.id,
        file_name=record.file_name,
        media_type=record.media_type,
        status=UploadStatus(record.status),
        size=record.expected_size,
        received_size=record.received_size,
        sha256=record.expected_sha256,
        created_at=_rfc3339(record.created_at),
        updated_at=_rfc3339(record.updated_at),
        expires_at=_rfc3339(record.expires_at),
    )


def _profile_error(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return str(exc.args[0])
    return "INVALID_REQUEST"


def _rfc3339(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")
