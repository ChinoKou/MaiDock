from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..domain import MediaCapability, MediaInputRole, PublicJsonObject
from ..domain.json_types import normalize_public_json_object

_IMAGE_MODES = frozenset({"text_to_image", "image_edit"})
_VIDEO_MODES = frozenset(
    {
        "text_to_video",
        "first_frame_to_video",
        "first_last_frame_to_video",
        "video_continuation",
        "reference_to_video",
        "video_edit",
    }
)


class ApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyRequest(ApiRequest):
    pass


class SourceRequest(ApiRequest):
    url: str | None = None
    upload_id: str | None = None

    @field_validator("url", "upload_id", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if (self.url is None) == (self.upload_id is None):
            raise ValueError("source 必须且只能提供 url 或 upload_id 之一")
        if self.url is not None:
            parsed = urlsplit(self.url)
            if (
                parsed.scheme.lower() != "https"
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("远程媒体 source 仅允许不含用户凭据的 HTTPS URL")
        if self.upload_id == "":
            raise ValueError("upload_id 不能为空")
        return self


class MediaInputRequest(ApiRequest):
    role: MediaInputRole
    source: SourceRequest
    reference_voice: SourceRequest | None = None

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, value: object) -> MediaInputRole:
        if isinstance(value, MediaInputRole):
            return value
        if isinstance(value, str):
            return MediaInputRole(value)
        raise TypeError("role 必须是字符串")


class CreateJobRequest(ApiRequest):
    capability: MediaCapability
    profile: str | None = None
    model: str | None = None
    mode: str = Field(min_length=1, max_length=100)
    protocol_family: str | None = Field(default=None, min_length=1, max_length=100)
    prompt: str | None = None
    negative_prompt: str | None = None
    inputs: list[MediaInputRequest] = Field(default_factory=list, max_length=16)
    parameters: PublicJsonObject = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=200)

    @field_validator("capability", mode="before")
    @classmethod
    def parse_capability(cls, value: object) -> MediaCapability:
        if isinstance(value, MediaCapability):
            return value
        if isinstance(value, str):
            return MediaCapability(value)
        raise TypeError("capability 必须是字符串")

    @field_validator(
        "profile",
        "model",
        "mode",
        "protocol_family",
        "prompt",
        "negative_prompt",
        "idempotency_key",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> PublicJsonObject:
        return normalize_public_json_object(value)

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        modes = _IMAGE_MODES if self.capability is MediaCapability.IMAGE_GENERATION else _VIDEO_MODES
        if self.mode not in modes:
            raise ValueError(f"capability {self.capability} 不支持 mode {self.mode}")
        return self


class JobIdRequest(ApiRequest):
    job_id: str = Field(min_length=1, max_length=200)

    @field_validator("job_id", mode="before")
    @classmethod
    def strip_job_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class CreateUploadRequest(ApiRequest):
    media_type: str = Field(min_length=1, max_length=200)
    size: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    file_name: str = Field(default="", max_length=255)

    @field_validator("media_type", "sha256", "file_name", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class OneShotUploadRequest(ApiRequest):
    media_type: str = Field(min_length=1, max_length=200)
    data: bytes = Field(min_length=1, max_length=8 * 1024 * 1024)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    file_name: str = Field(default="", max_length=255)

    @field_validator("media_type", "sha256", "file_name", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str | None) -> str | None:
        return value.lower() if value else None


class UploadIdRequest(ApiRequest):
    upload_id: str = Field(min_length=1, max_length=200)

    @field_validator("upload_id", mode="before")
    @classmethod
    def strip_upload_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class WriteUploadChunkRequest(UploadIdRequest):
    offset: int = Field(ge=0)
    data: bytes = Field(min_length=1, max_length=1024 * 1024)


class ReadArtifactRequest(ApiRequest):
    artifact_id: str = Field(min_length=1, max_length=200)
    offset: int = Field(default=0, ge=0)
    length: int = Field(default=1024 * 1024, ge=1, le=1024 * 1024)

    @field_validator("artifact_id", mode="before")
    @classmethod
    def strip_artifact_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value
