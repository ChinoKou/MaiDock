import base64
from pathlib import Path

from ....clients.ark import ArkClient, ArkClientError
from ....clients.common import ClientHttpError, ClientTimeoutError, JsonObject, JsonValue, RetryPolicy
from ....clients.families import JsonResourceRequest
from ...domain import (
    Accepted,
    Canceled,
    Completed,
    Failed,
    MaterializedArtifact,
    MediaCapability,
    MediaError,
    MediaInputRole,
    MediaOutcome,
    MediaOutput,
    MediaRequest,
    ModelCapability,
    PreparedMediaOperation,
    PublicDriverOperationError,
    PublicJsonObject,
    PublicJsonValue,
    Running,
    VersionedOpaqueHandle,
)
from ...domain.json_types import normalize_public_json_object
from .registry import (
    ArkMediaProfile,
    ArkMediaProtocolFamily,
    ArkResolvedMediaRequest,
    media_capabilities,
    resolve_media_request,
)
from .wire import ArkImagesResponse, ArkPreparedPayload, ArkRemoteHandlePayload, ArkTaskResponse

_DRIVER_KEY = "volcengine_ark.media.v1"
_PAYLOAD_VERSION = 1
# 提交零重试：ARK 一次成功的创建就会计费，超时后重发可能产生第二笔任务。
_SUBMISSION_RETRY = RetryPolicy(max_retries=0, uncertain_on_timeout=True)

# 图片与音频走内联 base64；上限取文档里对单文件的限制。
_MAX_INLINE_IMAGE_BYTES = 30 * 1024 * 1024
_MAX_INLINE_AUDIO_BYTES = 15 * 1024 * 1024

# 只有 queued 的任务 DELETE 才是"取消"；终态上 DELETE 会不可逆地删除任务记录。
_CANCELLABLE_STATUS = "queued"

# MediaInputRole 到 ARK content[] 里 image_url.role 的映射。
_IMAGE_ROLE_NAMES: dict[MediaInputRole, str] = {
    MediaInputRole.FIRST_FRAME: "first_frame",
    MediaInputRole.LAST_FRAME: "last_frame",
    MediaInputRole.REFERENCE_IMAGE: "reference_image",
}


class ArkPublicDriver:
    """Public API 合约到 Volcengine ARK 图片/视频资源的供应商执行器。"""

    def __init__(self, *, client: ArkClient, profiles: tuple[ArkMediaProfile, ...] = ()) -> None:
        self.client = client
        self._profiles: dict[str, ArkMediaProfile] = {}
        for profile in profiles:
            if profile.name in self._profiles:
                raise ValueError(f"Volcengine ARK 媒体 profile 重复: {profile.name}")
            self._profiles[profile.name] = profile

    @property
    def driver_key(self) -> str:
        return _DRIVER_KEY

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return media_capabilities()

    def prepare(self, profile_name: str, request: MediaRequest) -> PreparedMediaOperation:
        profile = self._require_profile(profile_name)
        resolved = resolve_media_request(request, profile)
        payload = ArkPreparedPayload(
            model=resolved.model,
            mode=resolved.request.mode,
            body=build_media_body(resolved),
        )
        return PreparedMediaOperation(
            driver_key=self.driver_key,
            payload_version=_PAYLOAD_VERSION,
            profile_name=profile_name,
            capability=request.capability,
            operation_type=resolved.family,
            payload=normalize_public_json_object(payload.model_dump(mode="json")),
        )

    async def submit(self, operation: PreparedMediaOperation) -> MediaOutcome:
        try:
            self._validate_prepared(operation)
            profile = self._require_profile(operation.profile_name)
            family = _protocol_family(operation.operation_type)
            payload = ArkPreparedPayload.model_validate(operation.payload)
            body = _to_client_json_object(payload.body)
            async with self.client.session(profile.connection) as session:
                if family == "ark_images_generations":
                    # 图片是同步接口：一次 POST 就拿到全部结果，没有任务句柄。
                    raw = await session.images_generations.create(
                        JsonResourceRequest(body=body, headers=payload.headers),
                        retry=_SUBMISSION_RETRY,
                    )
                    return _parse_images_outcome(raw)
                raw = await session.content_generation_tasks.create(body, retry=_SUBMISSION_RETRY)
            return _parse_task_outcome(
                raw,
                profile_name=profile.name,
                capability=operation.capability,
            )
        except ClientHttpError as exc:
            return Failed(_media_error(exc, submission=True))

    async def poll(self, handle: VersionedOpaqueHandle) -> MediaOutcome:
        try:
            profile, task_id, capability = self._resolve_handle(handle)
            async with self.client.session(profile.connection) as session:
                raw = await session.content_generation_tasks.retrieve(task_id, retry=session.safe_retry)
            return _parse_task_outcome(
                raw,
                profile_name=profile.name,
                capability=capability,
                fallback_task_id=task_id,
            )
        except ClientHttpError as exc:
            return Failed(_media_error(exc))

    async def cancel(self, handle: VersionedOpaqueHandle) -> MediaOutcome:
        """先查后删的防误删状态机。

        ARK 的 DELETE 一词两义：`queued` 时是取消，`succeeded`/`failed`/`expired` 时是
        **不可逆地删除任务记录**（此后连查询都查不到）。所以这里绝不能直接 DELETE：
        必须先 retrieve 拿到当前状态，只有 queued 才发 DELETE；running 说明来不及取消，
        原样返回让引擎继续轮询；已在终态的一律原样返回，绝不触碰远端记录。
        """

        try:
            profile, task_id, capability = self._resolve_handle(handle)
            async with self.client.session(profile.connection) as session:
                current = await session.content_generation_tasks.retrieve(task_id, retry=session.safe_retry)
                status = _task_status(current)
                if status != _CANCELLABLE_STATUS:
                    return _parse_task_outcome(
                        current,
                        profile_name=profile.name,
                        capability=capability,
                        fallback_task_id=task_id,
                    )
                # 取消动作本身不重试：重试意味着可能在任务已经转成 running/终态之后再发一次
                # DELETE，而那时同一个请求的含义已经变成"删除记录"。
                try:
                    await session.content_generation_tasks.delete(task_id, retry=RetryPolicy(max_retries=0))
                except ClientHttpError:
                    # DELETE 被拒说明竞态坐实：retrieve 到 DELETE 之间任务离开了 queued
                    # （转成 running，或已被别处取消），ARK 对这些状态拒绝 DELETE。
                    # 「取消没赶上」不是作业失败——吞掉拒绝，落到下面的确认查询，
                    # 返回远端真实状态（running 会让引擎继续轮询）。
                    pass
                # DELETE 无响应体，且可能与状态流转竞态（刚好转成 running 则会被拒），
                # 因此重新查询一次，返回远端的真实状态而不是想当然的 Canceled。
                confirmed = await session.content_generation_tasks.retrieve(task_id, retry=session.safe_retry)
            return _parse_task_outcome(
                confirmed,
                profile_name=profile.name,
                capability=capability,
                fallback_task_id=task_id,
            )
        except ClientHttpError as exc:
            return Failed(_media_error(exc))

    async def upload_file(
        self,
        profile_name: str,
        *,
        model: str,
        path: Path,
        media_type: str,
    ) -> str:
        """把本地文件转成 data URL 内联返回。

        ARK 没有 DashScope 那样的 OSS 直传流程，图片与音频直接以
        `data:<mime>;base64,<...>` 的形式写进请求体即可，因此这里不发任何网络请求，
        引擎的 upload token 替换机制拿到这个字符串就能用。视频没有内联形式——请求体
        整体上限 64 MB，把视频塞进 base64 必然超限——只能接受 HTTPS URL。
        """

        del model
        self._require_profile(profile_name)
        essence = media_type.split(";", 1)[0].strip().lower()
        if essence.startswith("video/"):
            raise PublicDriverOperationError(
                MediaError("UPLOAD_UNSUPPORTED", "Volcengine ARK 不支持上传视频，请改用 HTTPS URL")
            )
        if essence.startswith("audio/"):
            limit = _MAX_INLINE_AUDIO_BYTES
        elif essence.startswith("image/"):
            limit = _MAX_INLINE_IMAGE_BYTES
        else:
            raise PublicDriverOperationError(
                MediaError("UPLOAD_UNSUPPORTED", f"Volcengine ARK 不支持上传该类型: {media_type}")
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PublicDriverOperationError(MediaError("UPLOAD_NOT_FOUND", f"读取上传文件失败: {exc}")) from exc
        if len(data) > limit:
            raise PublicDriverOperationError(
                MediaError("UPLOAD_TOO_LARGE", f"Volcengine ARK 内联上传上限 {limit} 字节，实得 {len(data)}")
            )
        return f"data:{essence};base64,{base64.b64encode(data).decode('ascii')}"

    async def materialize(
        self,
        profile_name: str,
        *,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> MaterializedArtifact:
        profile = self._require_profile(profile_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with self.client.session(profile.connection) as session:
                downloaded = await session.artifacts.download(
                    url,
                    destination,
                    max_bytes=max_bytes,
                    retry=session.safe_retry,
                )
            return MaterializedArtifact(
                path=downloaded.path,
                size=downloaded.size,
                sha256=downloaded.sha256,
                media_type=downloaded.media_type,
            )
        except ClientHttpError as exc:
            raise PublicDriverOperationError(_media_error(exc, code="ARTIFACT_DOWNLOAD_FAILED")) from exc

    def _require_profile(self, profile_name: str) -> ArkMediaProfile:
        try:
            return self._profiles[profile_name]
        except KeyError as exc:
            raise KeyError(f"未知 Volcengine ARK 媒体 profile: {profile_name}") from exc

    def _validate_prepared(self, operation: PreparedMediaOperation) -> None:
        if operation.driver_key != self.driver_key:
            raise ValueError("prepared operation 不属于 Volcengine ARK Public Driver")
        if operation.payload_version != _PAYLOAD_VERSION:
            raise ValueError(f"不支持的 Volcengine ARK prepared payload 版本: {operation.payload_version}")

    def _resolve_handle(self, handle: VersionedOpaqueHandle) -> tuple[ArkMediaProfile, str, MediaCapability]:
        if handle.driver_key != self.driver_key or handle.payload_version != _PAYLOAD_VERSION:
            raise ValueError("远端句柄不属于当前 Volcengine ARK Public Driver 版本")
        payload = ArkRemoteHandlePayload.model_validate(handle.payload)
        capability = MediaCapability(payload.capability)
        return self._require_profile(payload.profile_name), payload.task_id, capability


def build_media_body(request: ArkResolvedMediaRequest) -> PublicJsonObject:
    if request.family == "ark_images_generations":
        return _build_images_body(request)
    return _build_video_body(request)


def _build_images_body(request: ArkResolvedMediaRequest) -> PublicJsonObject:
    parameters = dict(request.parameters)
    # max_images 在 wire 上是 sequential_image_generation_options 的嵌套字段，
    # 目录里按扁平参数校验，发出去之前折进去。
    max_images = parameters.pop("max_images", None)
    body: PublicJsonObject = {"model": request.model, "prompt": request.request.prompt}
    images = [item.source.value for item in request.request.inputs]
    if len(images) == 1:
        body["image"] = images[0]
    elif images:
        body["image"] = list(images)
    body.update(parameters)
    if max_images is not None:
        body["sequential_image_generation_options"] = {"max_images": max_images}
    return body


def _build_video_body(request: ArkResolvedMediaRequest) -> PublicJsonObject:
    content: list[PublicJsonValue] = []
    if request.request.prompt:
        content.append({"type": "text", "text": request.request.prompt})
    for item in request.request.inputs:
        content.append(_content_item(item.role, item.source.value))
    body: PublicJsonObject = {"model": request.model, "content": content}
    # 生成参数按"新方式"直接放 request body 顶层（强校验），而不是拼进 prompt 的
    # `--key value` 后缀（弱校验，填错会被静默忽略）。
    body.update(request.parameters)
    return body


def _content_item(role: MediaInputRole, url: str) -> PublicJsonObject:
    if role in {MediaInputRole.VIDEO, MediaInputRole.REFERENCE_VIDEO}:
        return {"type": "video_url", "video_url": {"url": url}}
    if role is MediaInputRole.DRIVING_AUDIO:
        return {"type": "audio_url", "audio_url": {"url": url}}
    image_url: PublicJsonObject = {"url": url}
    role_name = _IMAGE_ROLE_NAMES.get(role)
    if role_name is not None:
        image_url["role"] = role_name
    return {"type": "image_url", "image_url": image_url}


def _parse_images_outcome(payload: JsonObject) -> MediaOutcome:
    """图片是同步接口，一次返回即终态。"""

    response = ArkImagesResponse.model_validate(normalize_public_json_object(payload))
    if response.error is not None and (response.error.code or response.error.message):
        return Failed(
            MediaError(
                response.error.code or "UPSTREAM_TASK_FAILED",
                response.error.message or "Volcengine ARK 图片生成失败",
            )
        )
    outputs: list[MediaOutput] = []
    warnings: list[str] = []
    failed = 0
    # 组图模式下每一项独立成败：成功项收进输出，失败项计数并转成 warning，
    # 不能因为其中一张被审核拦下就把整单判失败。
    for item in response.data:
        if item.url is not None:
            outputs.append(MediaOutput(kind="media", url=item.url))
            continue
        failed += 1
        code = item.error.code if item.error is not None else None
        message = item.error.message if item.error is not None else None
        warnings.append(f"{code or 'OUTPUT_FAILED'}: {message or '图片结果项失败'}")
    if not outputs and failed == 0:
        return Failed(MediaError("NO_MEDIA_OUTPUT", "Volcengine ARK 未返回任何图片"))
    return Completed(
        outputs=tuple(outputs),
        usage=response.usage,
        warnings=tuple(warnings),
        failed_output_count=failed,
    )


def _task_status(payload: JsonObject) -> str:
    status = payload.get("status")
    return status.strip().lower() if isinstance(status, str) else ""


def _parse_task_outcome(
    payload: JsonObject,
    *,
    profile_name: str,
    capability: MediaCapability,
    fallback_task_id: str | None = None,
) -> MediaOutcome:
    response = ArkTaskResponse.model_validate(normalize_public_json_object(payload))
    task_id = response.id or fallback_task_id
    status = (response.status or "").strip().lower()

    if status in {"", "queued"} and task_id is not None:
        return Accepted(_handle(profile_name, capability, task_id))
    if status == "running" and task_id is not None:
        return Running(_handle(profile_name, capability, task_id))
    if status == "cancelled":
        return Canceled()
    if status == "expired":
        return Failed(MediaError("UPSTREAM_TASK_EXPIRED", "Volcengine ARK 视频任务已超时"))
    if status == "failed":
        code = response.error.code if response.error is not None else None
        message = response.error.message if response.error is not None else None
        return Failed(MediaError(code or "UPSTREAM_TASK_FAILED", message or "Volcengine ARK 视频任务失败"))
    if status != "succeeded":
        return Failed(MediaError("UPSTREAM_PROTOCOL_ERROR", f"未知 Volcengine ARK 任务状态: {status or '(缺失)'}"))

    content = response.content
    outputs: list[MediaOutput] = []
    if content is not None and content.video_url is not None:
        outputs.append(MediaOutput(kind="media", url=content.video_url, media_type="video/mp4"))
    if content is not None and content.last_frame_url is not None:
        outputs.append(MediaOutput(kind="media", url=content.last_frame_url))
    if not outputs:
        return Failed(MediaError("NO_MEDIA_OUTPUT", "Volcengine ARK 任务成功但未返回产物"))
    return Completed(outputs=tuple(outputs), usage=response.usage)


def _handle(profile_name: str, capability: MediaCapability, task_id: str) -> VersionedOpaqueHandle:
    return VersionedOpaqueHandle(
        driver_key=_DRIVER_KEY,
        payload_version=_PAYLOAD_VERSION,
        payload={"profile_name": profile_name, "capability": capability.value, "task_id": task_id},
    )


def _media_error(error: ClientHttpError, *, submission: bool = False, code: str | None = None) -> MediaError:
    if isinstance(error, ArkClientError):
        stable_code = code or error.code
        request_id = error.request_id
    else:
        stable_code = code or ("EXECUTION_UNCERTAIN" if submission and error.uncertain else "UPSTREAM_UNAVAILABLE")
        request_id = None
    if isinstance(error, ClientTimeoutError) and submission:
        stable_code = "EXECUTION_UNCERTAIN"
    return MediaError(
        stable_code,
        str(error),
        retryable=error.retryable,
        uncertain=submission and error.uncertain,
        request_id=request_id,
    )


def _protocol_family(value: str) -> ArkMediaProtocolFamily:
    match value:
        case "ark_images_generations":
            return "ark_images_generations"
        case "ark_content_generation_tasks":
            return "ark_content_generation_tasks"
        case _:
            raise ValueError(f"未知 Volcengine ARK media operation type: {value}")


def _to_client_json_object(value: PublicJsonObject) -> JsonObject:
    return {key: _to_client_json(item) for key, item in value.items()}


def _to_client_json(value: PublicJsonValue) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _to_client_json(item) for key, item in value.items()}
    return [_to_client_json(item) for item in value]
