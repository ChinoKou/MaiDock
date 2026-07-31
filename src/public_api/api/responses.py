from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain import JobStatus, MediaCapability, PublicJsonObject, PublicRpcObject, UploadStatus
from ..domain.json_types import normalize_public_rpc, normalize_public_rpc_object


class ApiResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)


class ErrorData(ApiResponseModel):
    code: str
    message: str
    retryable: bool = False
    uncertain: bool = False
    provider_request_id: str | None = None


class SuccessEnvelope[T: ApiResponseModel](ApiResponseModel):
    ok: Literal[True] = True
    data: T
    error: None = None


class ErrorEnvelope(ApiResponseModel):
    ok: Literal[False] = False
    data: None = None
    error: ErrorData


class ModelCapabilityData(ApiResponseModel):
    model: str
    capability: MediaCapability
    modes: tuple[str, ...]
    protocol_families: tuple[str, ...]
    max_outputs: int = Field(ge=1)


class ProfileData(ApiResponseModel):
    name: str
    provider: str
    driver: str


class CapabilityData(ApiResponseModel):
    models: tuple[ModelCapabilityData, ...]
    profiles: tuple[ProfileData, ...]
    default_image_profile: str | None = None
    default_video_profile: str | None = None


class TextOutputData(ApiResponseModel):
    type: Literal["text"] = "text"
    text: str


class ArtifactOutputData(ApiResponseModel):
    type: Literal["artifact"] = "artifact"
    artifact_id: str
    media_type: str
    size: int
    sha256: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    expires_at: str


type JobOutputData = TextOutputData | ArtifactOutputData


class JobErrorData(ApiResponseModel):
    code: str
    retryable: bool = False
    uncertain: bool = False
    provider_request_id: str | None = None


class JobData(ApiResponseModel):
    job_id: str
    status: JobStatus
    capability: MediaCapability
    profile: str
    provider: str
    model: str
    mode: str
    protocol_family: str
    created_at: str
    updated_at: str
    outputs: tuple[JobOutputData, ...] = ()
    warnings: tuple[str, ...] = ()
    failed_output_count: int = 0
    usage: PublicJsonObject = Field(default_factory=dict)
    error: JobErrorData | None = None


class DeletedJobData(ApiResponseModel):
    job_id: str
    deleted: bool


class UploadData(ApiResponseModel):
    upload_id: str
    file_name: str
    media_type: str
    status: UploadStatus
    size: int
    received_size: int
    sha256: str
    created_at: str
    updated_at: str
    expires_at: str


class DeletedUploadData(ApiResponseModel):
    upload_id: str
    deleted: bool


class ArtifactChunkData(ApiResponseModel):
    artifact_id: str
    offset: int
    next_offset: int
    eof: bool
    chunk: bytes
    media_type: str
    size: int
    sha256: str


def dump_response(model: BaseModel) -> PublicRpcObject:
    return normalize_public_rpc_object(model.model_dump(mode="python"))


def success_response(data: ApiResponseModel) -> PublicRpcObject:
    return normalize_public_rpc_object(
        {
            "ok": True,
            "data": normalize_public_rpc(data.model_dump(mode="python")),
            "error": None,
        }
    )


def error_response(error: ErrorData) -> PublicRpcObject:
    envelope = ErrorEnvelope(error=error)
    if envelope.ok or envelope.data is not None:
        raise AssertionError("错误 Envelope 状态无效")
    return dump_response(envelope)
