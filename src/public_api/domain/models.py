from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from .json_types import PublicJsonObject


class MediaCapability(StrEnum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"


class MediaInputRole(StrEnum):
    SOURCE_IMAGE = "source_image"
    REFERENCE_IMAGE = "reference_image"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    FIRST_CLIP = "first_clip"
    REFERENCE_VIDEO = "reference_video"
    VIDEO = "video"
    DRIVING_AUDIO = "driving_audio"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class JobStage(StrEnum):
    PREPARING = "preparing"
    SUBMITTING = "submitting"
    POLLING = "polling"
    MATERIALIZING = "materializing"


class UploadStatus(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class MediaSource:
    value: str

    def __post_init__(self) -> None:
        if self.value.startswith("oss://"):
            return
        parsed = urlsplit(self.value)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError("媒体来源必须是有效的 HTTPS URL 或 oss:// 引用")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("媒体来源 URL 不允许包含用户凭据")


@dataclass(frozen=True, slots=True)
class MediaInput:
    role: MediaInputRole
    source: MediaSource
    reference_voice: MediaSource | None = None


@dataclass(frozen=True, slots=True)
class MediaRequest:
    capability: MediaCapability
    mode: str
    prompt: str = ""
    negative_prompt: str = ""
    model: str | None = None
    protocol_family: str | None = None
    inputs: tuple[MediaInput, ...] = ()
    parameters: PublicJsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedMediaOperation:
    driver_key: str
    payload_version: int
    profile_name: str
    capability: MediaCapability
    operation_type: str
    payload: PublicJsonObject


@dataclass(frozen=True, slots=True)
class VersionedOpaqueHandle:
    driver_key: str
    payload_version: int
    payload: PublicJsonObject


@dataclass(frozen=True, slots=True)
class MediaOutput:
    kind: str
    text: str | None = None
    url: str | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class MediaError:
    code: str
    message: str
    retryable: bool = False
    uncertain: bool = False
    request_id: str | None = None


class PublicDriverOperationError(RuntimeError):
    """Driver 在 outcome 之外报告的稳定操作错误。"""

    def __init__(self, error: MediaError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    path: Path
    size: int
    sha256: str
    media_type: str


@dataclass(frozen=True, slots=True)
class ModelCapability:
    model: str
    capability: MediaCapability
    modes: tuple[str, ...]
    protocol_families: tuple[str, ...]
    max_outputs: int


@dataclass(frozen=True, slots=True)
class Completed:
    outputs: tuple[MediaOutput, ...]
    usage: PublicJsonObject = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    failed_output_count: int = 0
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class Accepted:
    remote_handle: VersionedOpaqueHandle
    next_poll_after_seconds: float = 1.0
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class Running:
    remote_handle: VersionedOpaqueHandle
    next_poll_after_seconds: float = 1.0
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class Failed:
    error: MediaError


@dataclass(frozen=True, slots=True)
class Canceled:
    request_id: str | None = None


type MediaOutcome = Completed | Accepted | Running | Failed | Canceled
