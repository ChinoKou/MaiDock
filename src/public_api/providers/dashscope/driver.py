from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit

import mimetypes

from ....clients.common import (
    ClientHttpError,
    ClientTimeoutError,
    JsonObject,
    JsonValue,
    RetryPolicy,
    SseJsonEvent,
)
from ....clients.dashscope import DashScopeClient, DashScopeClientError, DashScopeSession
from ....clients.families import JsonResource, JsonResourceRequest
from ...domain import (
    Accepted,
    Canceled,
    Completed,
    Failed,
    MediaCapability,
    MediaError,
    MediaOutcome,
    MediaOutput,
    MediaRequest,
    MaterializedArtifact,
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
    DashScopeMediaProfile,
    MediaProtocolFamily,
    ResolvedMediaRequest,
    media_capabilities,
    resolve_media_request,
)
from .wire import (
    DashScopePreparedPayload,
    DashScopeRemoteHandlePayload,
    DashScopeResponse,
    DashScopeTaskOutput,
    DashScopeUploadPolicyResponse,
)

_DRIVER_KEY = "dashscope.media.v1"
_PAYLOAD_VERSION = 1
_SUBMISSION_RETRY = RetryPolicy(max_retries=0, uncertain_on_timeout=True)
_ASYNC_FAMILIES: frozenset[MediaProtocolFamily] = frozenset(
    {
        "dashscope_image_generation",
        "dashscope_text2image_synthesis",
        "dashscope_image2image_synthesis",
        "dashscope_video_generation",
    }
)


class DashScopePublicDriver:
    """Public API 合约到五个 DashScope 精确协议资源的供应商执行器。"""

    def __init__(
        self,
        *,
        client: DashScopeClient,
        profiles: tuple[DashScopeMediaProfile, ...] = (),
    ) -> None:
        self.client = client
        self._profiles: dict[str, DashScopeMediaProfile] = {}
        for profile in profiles:
            if profile.name in self._profiles:
                raise ValueError(f"DashScope 媒体 profile 重复: {profile.name}")
            self._profiles[profile.name] = profile

    @property
    def driver_key(self) -> str:
        return _DRIVER_KEY

    def capabilities(self) -> tuple[ModelCapability, ...]:
        return media_capabilities()

    def prepare(self, profile_name: str, request: MediaRequest) -> PreparedMediaOperation:
        profile = self._require_profile(profile_name)
        resolved = resolve_media_request(request, profile)
        body = build_media_body(resolved)
        headers: dict[str, str] = {}
        if resolved.family in _ASYNC_FAMILIES:
            headers["X-DashScope-Async"] = "enable"
        if _contains_oss(body):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        use_sse = (
            resolved.family == "dashscope_multimodal_generation"
            and resolved.model == "wan2.6-image"
            and resolved.request.mode == "text_to_image"
            and resolved.parameters.get("enable_interleave") is True
        )
        if use_sse:
            headers.update({"Accept": "text/event-stream", "X-DashScope-SSE": "enable"})
            parameters = normalize_public_json_object(body["parameters"])
            parameters["stream"] = True
            body["parameters"] = parameters
        payload = DashScopePreparedPayload(
            model=resolved.model,
            mode=resolved.request.mode,
            body=body,
            headers=headers,
            use_sse=use_sse,
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
            payload = DashScopePreparedPayload.model_validate(operation.payload)
            async with self.client.session(profile.connection) as session:
                resource = _resource_for_family(session, family)
                request = JsonResourceRequest(
                    body=_to_client_json_object(payload.body),
                    headers=payload.headers,
                )
                if payload.use_sse:
                    return await _collect_wan26_sse(
                        resource.stream(request, retry=_SUBMISSION_RETRY),
                        profile_name=profile.name,
                        capability=operation.capability,
                    )
                payload = await resource.create(request, retry=_SUBMISSION_RETRY)
            return _parse_outcome(
                payload,
                profile_name=profile.name,
                capability=operation.capability,
            )
        except ClientHttpError as exc:
            return Failed(_media_error(exc, submission=True))

    async def poll(self, handle: VersionedOpaqueHandle) -> MediaOutcome:
        try:
            profile, task_id, capability = self._resolve_handle(handle)
            async with self.client.session(profile.connection) as session:
                payload = await session.tasks.retrieve(task_id, retry=session.safe_retry)
            return _parse_outcome(
                payload,
                profile_name=profile.name,
                capability=capability,
                fallback_task_id=task_id,
            )
        except ClientHttpError as exc:
            return Failed(_media_error(exc))

    async def cancel(self, handle: VersionedOpaqueHandle) -> MediaOutcome:
        try:
            profile, task_id, capability = self._resolve_handle(handle)
            async with self.client.session(profile.connection) as session:
                payload = await session.tasks.cancel(task_id, retry=session.safe_retry)
            outcome = _parse_outcome(
                payload,
                profile_name=profile.name,
                capability=capability,
                fallback_task_id=task_id,
            )
            if isinstance(outcome, (Accepted, Running)):
                return Canceled(request_id=outcome.request_id)
            return outcome
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
        profile = self._require_profile(profile_name)
        try:
            async with self.client.session(profile.connection) as session:
                policy_payload = await session.uploads.create_policy(model, retry=session.safe_retry)
                output = DashScopeUploadPolicyResponse.model_validate(
                    normalize_public_json_object(policy_payload)
                ).output
                upload_dir = output.upload_dir.rstrip("/")
                object_key = f"{upload_dir}/{path.name}"
                upload_url = output.upload_host
                parsed_upload_url = urlsplit(upload_url)
                if parsed_upload_url.scheme.lower() != "https" or not parsed_upload_url.netloc:
                    raise PublicDriverOperationError(
                        MediaError("UPSTREAM_PROTOCOL_ERROR", "DashScope OSS upload_host 不是 HTTPS URL")
                    )
                form_data = {
                    "OSSAccessKeyId": output.oss_access_key_id,
                    "Signature": output.signature,
                    "policy": output.policy,
                    "key": object_key,
                    "x-oss-object-acl": output.x_oss_object_acl,
                    "x-oss-forbid-overwrite": output.x_oss_forbid_overwrite,
                    "success_action_status": "200",
                    "x-oss-content-type": media_type,
                }
                await session.uploads.upload(
                    upload_url=upload_url,
                    form_data=form_data,
                    path=path,
                    media_type=mimetypes.guess_type(path.name)[0] or media_type or "application/octet-stream",
                    retry=session.safe_retry,
                )
            return f"oss://{object_key}"
        except ClientHttpError as exc:
            raise PublicDriverOperationError(_media_error(exc, code="OSS_UPLOAD_FAILED")) from exc

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

    def _require_profile(self, profile_name: str) -> DashScopeMediaProfile:
        try:
            return self._profiles[profile_name]
        except KeyError as exc:
            raise KeyError(f"未知 DashScope 媒体 profile: {profile_name}") from exc

    def _validate_prepared(self, operation: PreparedMediaOperation) -> None:
        if operation.driver_key != self.driver_key:
            raise ValueError("prepared operation 不属于 DashScope Public Driver")
        if operation.payload_version != _PAYLOAD_VERSION:
            raise ValueError(f"不支持的 DashScope prepared payload 版本: {operation.payload_version}")

    def _resolve_handle(
        self,
        handle: VersionedOpaqueHandle,
    ) -> tuple[DashScopeMediaProfile, str, MediaCapability]:
        if handle.driver_key != self.driver_key or handle.payload_version != _PAYLOAD_VERSION:
            raise ValueError("远端句柄不属于当前 DashScope Public Driver 版本")
        payload = DashScopeRemoteHandlePayload.model_validate(handle.payload)
        profile_name = payload.profile_name
        task_id = payload.task_id
        capability = MediaCapability(payload.capability)
        return self._require_profile(profile_name), task_id, capability


def build_media_body(request: ResolvedMediaRequest) -> PublicJsonObject:
    prompt = request.request.prompt
    negative_prompt = request.request.negative_prompt
    if request.family in {"dashscope_multimodal_generation", "dashscope_image_generation"}:
        content: list[PublicJsonValue] = []
        if prompt:
            content.append({"text": prompt})
        for item in request.request.inputs:
            content.append({"image": item.source.value})
        parameters = dict(request.parameters)
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt
        return {
            "model": request.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": parameters,
        }
    if request.family == "dashscope_text2image_synthesis":
        input_body: PublicJsonObject = {"prompt": prompt}
        if negative_prompt:
            input_body["negative_prompt"] = negative_prompt
        return {"model": request.model, "input": input_body, "parameters": dict(request.parameters)}
    if request.family == "dashscope_image2image_synthesis":
        input_body = {
            "prompt": prompt,
            "images": [item.source.value for item in request.request.inputs],
        }
        if negative_prompt:
            input_body["negative_prompt"] = negative_prompt
        return {"model": request.model, "input": input_body, "parameters": dict(request.parameters)}
    video_input: PublicJsonObject = {"prompt": prompt}
    if negative_prompt:
        video_input["negative_prompt"] = negative_prompt
    if request.model.startswith("wan2.7-") and request.request.mode != "text_to_video":
        media: list[PublicJsonValue] = []
        for item in request.request.inputs:
            element: PublicJsonObject = {"type": item.role.value, "url": item.source.value}
            if item.reference_voice is not None:
                element["reference_voice"] = item.reference_voice.value
            media.append(element)
        if media:
            video_input["media"] = media
    elif request.model.startswith(("wan2.6-r2v", "wan2.5-r2v")):
        video_input["reference_urls"] = [item.source.value for item in request.request.inputs]
    else:
        media = []
        for item in request.request.inputs:
            if item.role.value == "first_frame":
                video_input["img_url"] = item.source.value
            elif item.role.value == "driving_audio":
                video_input["audio_url"] = item.source.value
            else:
                media.append({"type": item.role.value, "url": item.source.value})
        if media:
            video_input["media"] = media
    return {"model": request.model, "input": video_input, "parameters": dict(request.parameters)}


def _resource_for_family(session: DashScopeSession, family: MediaProtocolFamily) -> JsonResource:
    match family:
        case "dashscope_multimodal_generation":
            return session.multimodal_generation
        case "dashscope_image_generation":
            return session.image_generation
        case "dashscope_text2image_synthesis":
            return session.text2image_synthesis
        case "dashscope_image2image_synthesis":
            return session.image2image_synthesis
        case "dashscope_video_generation":
            return session.video_generation


async def _collect_wan26_sse(
    events: AsyncIterator[SseJsonEvent],
    *,
    profile_name: str,
    capability: MediaCapability,
) -> MediaOutcome:
    outputs: list[MediaOutput] = []
    usage: PublicJsonObject = {}
    request_id: str | None = None
    async for event in events:
        response = DashScopeResponse.model_validate(normalize_public_json_object(event.data))
        request_id = response.request_id or response.requestId or request_id
        usage.update(response.usage)
        if response.output is not None:
            _append_choice_outputs(outputs, response.output)
    if not outputs:
        return Failed(MediaError("UPSTREAM_PROTOCOL_ERROR", "DashScope SSE 未返回媒体输出", request_id=request_id))
    del profile_name, capability
    return Completed(outputs=tuple(outputs), usage=usage, request_id=request_id)


def _parse_outcome(
    payload: JsonObject,
    *,
    profile_name: str,
    capability: MediaCapability,
    fallback_task_id: str | None = None,
) -> MediaOutcome:
    response = DashScopeResponse.model_validate(normalize_public_json_object(payload))
    output = response.output
    if output is None:
        return Failed(MediaError("UPSTREAM_PROTOCOL_ERROR", "DashScope output 不是对象"))
    request_id = response.request_id or response.requestId
    task_id = output.task_id or fallback_task_id
    raw_status = (output.task_status or "").upper()
    if task_id is not None and raw_status in {"", "PENDING"}:
        return Accepted(_handle(profile_name, capability, task_id), request_id=request_id)
    if task_id is not None and raw_status == "RUNNING":
        return Running(_handle(profile_name, capability, task_id), request_id=request_id)
    if raw_status == "CANCELED":
        return Canceled(request_id=request_id)
    if raw_status == "FAILED":
        code = output.code or "UPSTREAM_TASK_FAILED"
        message = output.message or "DashScope 媒体任务失败"
        return Failed(MediaError(code, message, request_id=request_id))
    outputs, warnings, failed_count = _extract_outputs(output)
    if not outputs and task_id is not None and raw_status not in {"SUCCEEDED"}:
        return Running(_handle(profile_name, capability, task_id), request_id=request_id)
    return Completed(
        outputs=tuple(outputs),
        usage=response.usage,
        warnings=tuple(warnings),
        failed_output_count=failed_count,
        request_id=request_id,
    )


def _extract_outputs(output: DashScopeTaskOutput) -> tuple[list[MediaOutput], list[str], int]:
    results: list[MediaOutput] = []
    warnings: list[str] = []
    failed = 0
    _append_choice_outputs(results, output)
    for item in output.results:
        url = item.url or item.actual_url
        if url is not None:
            results.append(MediaOutput(kind="media", url=url))
        else:
            failed += 1
            warnings.append(f"{item.code or 'OUTPUT_FAILED'}: {item.message or '媒体结果项失败'}")
    if output.video_url is not None:
        results.append(MediaOutput(kind="media", url=output.video_url, media_type="video/mp4"))
    return results, warnings, failed


def _append_choice_outputs(results: list[MediaOutput], output: DashScopeTaskOutput) -> None:
    for choice in output.choices:
        if choice.message is None:
            continue
        for item in choice.message.content:
            if item.text is not None:
                if results and results[-1].kind == "text":
                    previous = results[-1]
                    results[-1] = MediaOutput(kind="text", text=f"{previous.text or ''}{item.text}")
                else:
                    results.append(MediaOutput(kind="text", text=item.text))
            url = item.image or item.video or item.url
            if url is not None:
                results.append(MediaOutput(kind="media", url=url))


def _handle(
    profile_name: str,
    capability: MediaCapability,
    task_id: str,
) -> VersionedOpaqueHandle:
    return VersionedOpaqueHandle(
        driver_key=_DRIVER_KEY,
        payload_version=_PAYLOAD_VERSION,
        payload={"profile_name": profile_name, "capability": capability.value, "task_id": task_id},
    )


def _media_error(
    error: ClientHttpError,
    *,
    submission: bool = False,
    code: str | None = None,
) -> MediaError:
    if isinstance(error, DashScopeClientError):
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


def _protocol_family(value: str) -> MediaProtocolFamily:
    match value:
        case "dashscope_multimodal_generation":
            return "dashscope_multimodal_generation"
        case "dashscope_image_generation":
            return "dashscope_image_generation"
        case "dashscope_text2image_synthesis":
            return "dashscope_text2image_synthesis"
        case "dashscope_image2image_synthesis":
            return "dashscope_image2image_synthesis"
        case "dashscope_video_generation":
            return "dashscope_video_generation"
        case _:
            raise ValueError(f"未知 DashScope media operation type: {value}")


def _contains_oss(value: PublicJsonValue) -> bool:
    if isinstance(value, str):
        return value.startswith("oss://")
    if isinstance(value, dict):
        return any(_contains_oss(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_oss(item) for item in value)
    return False


def _to_client_json_object(value: PublicJsonObject) -> JsonObject:
    return {key: _to_client_json(item) for key, item in value.items()}


def _to_client_json(value: PublicJsonValue) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _to_client_json(item) for key, item in value.items()}
    return [_to_client_json(item) for item in value]
