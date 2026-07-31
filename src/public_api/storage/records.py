from dataclasses import dataclass
import json

from pydantic import BaseModel, ConfigDict, Field

from ..domain import PublicJsonObject, PublicJsonValue


class PersistenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JobRecord(PersistenceRecord):
    id: str
    driver_key: str
    payload_version: int
    profile_name: str
    credential_fingerprint: str
    capability: str
    model: str
    mode: str
    protocol_family: str
    prepared_payload: PublicJsonObject
    remote_handle: PublicJsonObject | None = None
    stage: str
    status: str
    error_code: str | None = None
    error_message: str | None = None
    error_retryable: bool = False
    error_uncertain: bool = False
    provider_request_id: str | None = None
    usage: PublicJsonObject = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failed_output_count: int = 0
    cancel_requested: bool = False
    next_poll_at: float | None = None
    created_at: float
    updated_at: float
    expires_at: float


class JobOutputRecord(PersistenceRecord):
    job_id: str
    ordinal: int
    kind: str
    text: str | None = None
    url: str | None = None
    media_type: str | None = None
    artifact_id: str | None = None


class UploadRecord(PersistenceRecord):
    id: str
    file_name: str
    media_type: str
    expected_size: int
    expected_sha256: str
    received_size: int
    status: str
    path: str
    created_at: float
    updated_at: float
    expires_at: float


class ArtifactRecord(PersistenceRecord):
    id: str
    job_id: str | None = None
    path: str
    media_type: str
    size: int
    sha256: str
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    created_at: float
    expires_at: float


class StagingRecord(PersistenceRecord):
    id: str
    job_id: str
    ordinal: int
    path: str
    reserved_size: int
    created_at: float


class IdempotencyRecord(PersistenceRecord):
    idempotency_key: str
    request_digest: str
    job_id: str
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class StoredJob:
    record: JobRecord
    outputs: tuple[JobOutputRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class NewJob:
    driver_key: str
    payload_version: int
    profile_name: str
    credential_fingerprint: str
    capability: str
    model: str
    mode: str
    protocol_family: str
    prepared_payload: PublicJsonObject
    upload_ids: tuple[str, ...] = ()
    idempotency_key: str | None = None
    request_digest: str | None = None


def encode_json(value: PublicJsonValue | tuple[str, ...]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_json(value: str) -> object:
    return json.loads(value)
