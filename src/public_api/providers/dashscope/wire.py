from pydantic import BaseModel, ConfigDict, Field

from ...domain import PublicJsonObject


class DashScopeWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class DashScopeCreateRequest(DashScopeWireModel):
    model: str
    input: PublicJsonObject
    parameters: PublicJsonObject = Field(default_factory=dict)


class MultimodalGenerationRequest(DashScopeCreateRequest):
    pass


class ImageGenerationRequest(DashScopeCreateRequest):
    pass


class Text2ImageSynthesisRequest(DashScopeCreateRequest):
    pass


class Image2ImageSynthesisRequest(DashScopeCreateRequest):
    pass


class VideoGenerationRequest(DashScopeCreateRequest):
    pass


class DashScopeContentItem(DashScopeWireModel):
    text: str | None = None
    image: str | None = None
    video: str | None = None
    url: str | None = None


class DashScopeMessage(DashScopeWireModel):
    content: list[DashScopeContentItem] = Field(default_factory=list)


class DashScopeChoice(DashScopeWireModel):
    message: DashScopeMessage | None = None


class DashScopeResultItem(DashScopeWireModel):
    url: str | None = None
    actual_url: str | None = None
    code: str | None = None
    message: str | None = None


class DashScopeTaskOutput(DashScopeWireModel):
    task_id: str | None = None
    task_status: str | None = None
    code: str | None = None
    message: str | None = None
    choices: list[DashScopeChoice] = Field(default_factory=list)
    results: list[DashScopeResultItem] = Field(default_factory=list)
    video_url: str | None = None


class DashScopeResponse(DashScopeWireModel):
    output: DashScopeTaskOutput | None = None
    usage: PublicJsonObject = Field(default_factory=dict)
    request_id: str | None = None
    requestId: str | None = None


class DashScopeUploadPolicy(DashScopeWireModel):
    upload_dir: str
    upload_host: str
    oss_access_key_id: str
    signature: str
    policy: str
    x_oss_object_acl: str
    x_oss_forbid_overwrite: str


class DashScopeUploadPolicyResponse(DashScopeWireModel):
    output: DashScopeUploadPolicy


class DashScopePreparedPayload(DashScopeWireModel):
    model: str
    mode: str
    body: PublicJsonObject
    headers: dict[str, str] = Field(default_factory=dict)
    use_sse: bool = False


class DashScopeRemoteHandlePayload(DashScopeWireModel):
    profile_name: str
    capability: str
    task_id: str
