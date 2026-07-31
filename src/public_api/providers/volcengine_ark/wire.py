from pydantic import BaseModel, ConfigDict, Field

from ...domain import PublicJsonObject


class ArkWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class ArkError(ArkWireModel):
    """ARK 的错误对象，图片顶层、逐图项与视频任务里用的是同一个形状。"""

    code: str | None = None
    message: str | None = None


class ArkImageDataItem(ArkWireModel):
    """图片响应 `data[]` 的一项。

    组图模式下每一项可能独立失败：成功项带 url（或 b64_json），失败项带 error。
    因此这里三个字段都是可选的，由 driver 逐项判定，而不是整单成败。
    """

    url: str | None = None
    b64_json: str | None = None
    size: str | None = None
    output_format: str | None = None
    error: ArkError | None = None


class ArkImagesResponse(ArkWireModel):
    """`POST /images/generations` 的同步响应。"""

    model: str | None = None
    created: int | None = None
    data: list[ArkImageDataItem] = Field(default_factory=list)
    # 顶层 error 表示整单失败，2xx 也可能带；Client 的 ark_error_factory 已先拦一道，
    # 这里保留字段是为了让 driver 在解析阶段也能读到同样的信息。
    error: ArkError | None = None
    usage: PublicJsonObject = Field(default_factory=dict)


class ArkTaskContent(ArkWireModel):
    """视频任务成功后的产物。"""

    video_url: str | None = None
    # 仅当创建任务时设置 return_last_frame=true 才会返回。
    last_frame_url: str | None = None


class ArkTaskResponse(ArkWireModel):
    """创建与查询视频生成任务共用的响应形状。

    status 是顶层字段（不像 DashScope 嵌在 output 里），error 则是嵌套的
    `{code, message}`，任务成功时为 null。
    """

    id: str | None = None
    model: str | None = None
    status: str | None = None
    content: ArkTaskContent | None = None
    error: ArkError | None = None
    usage: PublicJsonObject = Field(default_factory=dict)
    created_at: int | None = None
    updated_at: int | None = None


class ArkPreparedPayload(ArkWireModel):
    """prepare 阶段固化下来的请求，提交阶段只负责发出去。"""

    model: str
    mode: str
    body: PublicJsonObject
    headers: dict[str, str] = Field(default_factory=dict)


class ArkRemoteHandlePayload(ArkWireModel):
    """异步视频任务的远端句柄载荷。"""

    profile_name: str
    capability: str
    task_id: str
