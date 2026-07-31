import base64
import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from src.clients.ark import ArkClient, ArkConnection
from src.clients.common import NO_RETRY, HttpConnection, RetryPolicy
from src.public_api.domain import (
    Accepted,
    Canceled,
    Completed,
    Failed,
    MediaCapability,
    MediaInput,
    MediaInputRole,
    MediaRequest,
    MediaSource,
    PublicDriverOperationError,
    PublicJsonValue,
    Running,
    VersionedOpaqueHandle,
)
from src.public_api.providers.volcengine_ark.driver import ArkPublicDriver
from src.public_api.providers.volcengine_ark.registry import ArkMediaProfile

BASE_URL = "https://ark.example/api/v3"
IMAGE_URL = "https://cdn.example/in.png"
PROFILE = "default"


def _profile(name: str = PROFILE) -> ArkMediaProfile:
    return ArkMediaProfile(
        name=name,
        connection=ArkConnection(
            http=HttpConnection(base_url=BASE_URL, default_headers=(("Authorization", "Bearer ark-key"),)),
            retry=NO_RETRY,
            responses_path="responses",
            embeddings_path="embeddings/multimodal",
            audio_transcriptions_path="responses",
            tokenization_path="tokenization",
            safe_retry=RetryPolicy(max_retries=1),
        ),
    )


def _driver(handler: httpx.MockTransport) -> ArkPublicDriver:
    return ArkPublicDriver(client=ArkClient(transport=handler), profiles=(_profile(),))


def _handle(task_id: str = "cgt-1") -> VersionedOpaqueHandle:
    """直接构造句柄：poll/cancel 的用例不必先跑一次 submit，
    否则同一个 handler 要同时扮演"创建"和"查询"两种响应。"""

    return VersionedOpaqueHandle(
        driver_key="volcengine_ark.media.v1",
        payload_version=1,
        payload={
            "profile_name": PROFILE,
            "capability": MediaCapability.VIDEO_GENERATION.value,
            "task_id": task_id,
        },
    )


def _image_request(
    *,
    model: str = "doubao-seedream-4-0",
    mode: str = "text_to_image",
    parameters: Mapping[str, PublicJsonValue] | None = None,
    inputs: tuple[MediaInput, ...] = (),
) -> MediaRequest:
    return MediaRequest(
        capability=MediaCapability.IMAGE_GENERATION,
        mode=mode,
        prompt="一只猫",
        model=model,
        inputs=inputs,
        parameters=dict(parameters or {}),
    )


def _video_request(
    *,
    model: str = "doubao-seedance-2-0",
    mode: str = "text_to_video",
    parameters: Mapping[str, PublicJsonValue] | None = None,
    inputs: tuple[MediaInput, ...] = (),
) -> MediaRequest:
    return MediaRequest(
        capability=MediaCapability.VIDEO_GENERATION,
        mode=mode,
        prompt="小猫打哈欠",
        model=model,
        inputs=inputs,
        parameters=dict(parameters or {}),
    )


@pytest.mark.asyncio
async def test_image_submit_targets_exact_endpoint_with_bearer() -> None:
    seen: list[tuple[str, str, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("Authorization")))
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}], "usage": {"total_tokens": 3}})

    driver = _driver(httpx.MockTransport(handler))
    outcome = await driver.submit(driver.prepare(PROFILE, _image_request()))

    assert seen == [("POST", "/api/v3/images/generations", "Bearer ark-key")]
    assert isinstance(outcome, Completed)
    assert [item.url for item in outcome.outputs] == ["https://cdn.example/a.png"]
    assert outcome.usage == {"total_tokens": 3}


@pytest.mark.asyncio
async def test_image_partial_failure_becomes_warnings_not_whole_failure() -> None:
    """组图里单张被拦下不能把整单判失败，要计数并转成 warning。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "data": [
                    {"url": "https://cdn.example/ok1.png"},
                    {"error": {"code": "ContentFiltered", "message": "命中审核"}},
                    {"url": "https://cdn.example/ok2.png"},
                ]
            },
        )

    driver = _driver(httpx.MockTransport(handler))
    outcome = await driver.submit(
        driver.prepare(PROFILE, _image_request(parameters={"sequential_image_generation": "auto", "max_images": 3}))
    )

    assert isinstance(outcome, Completed)
    assert len(outcome.outputs) == 2
    assert outcome.failed_output_count == 1
    assert outcome.warnings == ("ContentFiltered: 命中审核",)


@pytest.mark.asyncio
async def test_image_2xx_top_level_error_is_whole_request_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"error": {"code": "SensitiveContent", "message": "命中审核"}})

    driver = _driver(httpx.MockTransport(handler))
    outcome = await driver.submit(driver.prepare(PROFILE, _image_request()))

    assert isinstance(outcome, Failed)
    assert outcome.error.code == "SensitiveContent"


@pytest.mark.asyncio
async def test_image_body_nests_max_images_and_locks_response_format() -> None:
    captured: list[Mapping[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured.append(body)
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/a.png"}]})

    driver = _driver(httpx.MockTransport(handler))
    await driver.submit(
        driver.prepare(PROFILE, _image_request(parameters={"sequential_image_generation": "auto", "max_images": 4}))
    )

    body = captured[0]
    assert body["sequential_image_generation_options"] == {"max_images": 4}
    assert "max_images" not in body
    assert body["response_format"] == "url"
    assert body["model"] == "doubao-seedream-4-0"


@pytest.mark.asyncio
async def test_video_submit_returns_accepted_handle() -> None:
    seen: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})

    driver = _driver(httpx.MockTransport(handler))
    outcome = await driver.submit(driver.prepare(PROFILE, _video_request()))

    assert seen == [("POST", "/api/v3/contents/generations/tasks")]
    assert isinstance(outcome, Accepted)
    assert outcome.remote_handle.payload["task_id"] == "cgt-1"
    assert outcome.remote_handle.driver_key == "volcengine_ark.media.v1"


@pytest.mark.asyncio
async def test_video_body_uses_top_level_parameters_not_prompt_suffix() -> None:
    """文档的"新方式"是顶层字段强校验；`--key value` 后缀是弱校验的旧方式。"""

    captured: list[Mapping[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured.append(body)
        return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})

    driver = _driver(httpx.MockTransport(handler))
    await driver.submit(
        driver.prepare(PROFILE, _video_request(parameters={"resolution": "720p", "ratio": "16:9", "duration": 5}))
    )

    body = captured[0]
    assert body["resolution"] == "720p"
    assert body["ratio"] == "16:9"
    assert body["duration"] == 5
    assert body["content"] == [{"type": "text", "text": "小猫打哈欠"}]
    assert "--" not in json.dumps(body["content"], ensure_ascii=False)


@pytest.mark.asyncio
async def test_video_body_maps_input_roles_to_content_items() -> None:
    captured: list[Mapping[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body, dict)
        captured.append(body)
        return httpx.Response(200, json={"id": "cgt-1", "status": "queued"})

    driver = _driver(httpx.MockTransport(handler))
    await driver.submit(
        driver.prepare(
            PROFILE,
            _video_request(
                mode="first_last_frame_to_video",
                inputs=(
                    MediaInput(role=MediaInputRole.FIRST_FRAME, source=MediaSource(IMAGE_URL)),
                    MediaInput(role=MediaInputRole.LAST_FRAME, source=MediaSource("https://cdn.example/last.png")),
                ),
            ),
        )
    )

    assert captured[0]["content"] == [
        {"type": "text", "text": "小猫打哈欠"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL, "role": "first_frame"}},
        {"type": "image_url", "image_url": {"url": "https://cdn.example/last.png", "role": "last_frame"}},
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("queued", Accepted),
        ("running", Running),
        ("cancelled", Canceled),
        ("failed", Failed),
        ("expired", Failed),
    ],
)
@pytest.mark.asyncio
async def test_poll_maps_every_documented_status(status: str, expected: type[object]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"id": "cgt-1", "status": status})

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.poll(_handle())

    assert isinstance(outcome, expected)


@pytest.mark.asyncio
async def test_poll_succeeded_returns_video_and_last_frame() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "cgt-1",
                "status": "succeeded",
                "content": {
                    "video_url": "https://cdn.example/out.mp4",
                    "last_frame_url": "https://cdn.example/last.png",
                },
                "usage": {"total_tokens": 12},
            },
        )

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.poll(_handle())

    assert isinstance(outcome, Completed)
    assert [item.url for item in outcome.outputs] == [
        "https://cdn.example/out.mp4",
        "https://cdn.example/last.png",
    ]
    assert outcome.outputs[0].media_type == "video/mp4"


@pytest.mark.asyncio
async def test_poll_expired_uses_dedicated_error_code() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"id": "cgt-1", "status": "expired"})

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.poll(_handle())

    assert isinstance(outcome, Failed)
    assert outcome.error.code == "UPSTREAM_TASK_EXPIRED"


@pytest.mark.asyncio
async def test_poll_failed_surfaces_nested_error_code() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"id": "cgt-1", "status": "failed", "error": {"code": "InternalError", "message": "上游异常"}},
        )

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.poll(_handle())

    assert isinstance(outcome, Failed)
    assert outcome.error.code == "InternalError"


@pytest.mark.asyncio
async def test_cancel_deletes_only_when_queued() -> None:
    """queued 时 DELETE 才是取消；这里断言完整的请求序列。"""

    seen: list[tuple[str, str]] = []
    statuses = iter(["queued", "cancelled"])

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(200)
        return httpx.Response(200, json={"id": "cgt-1", "status": next(statuses)})

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.cancel(_handle())

    assert seen == [
        ("GET", "/api/v3/contents/generations/tasks/cgt-1"),
        ("DELETE", "/api/v3/contents/generations/tasks/cgt-1"),
        ("GET", "/api/v3/contents/generations/tasks/cgt-1"),
    ]
    assert isinstance(outcome, Canceled)


@pytest.mark.asyncio
async def test_cancel_never_deletes_a_running_task() -> None:
    """running 不支持 DELETE，只能继续轮询——绝不能发出 DELETE。"""

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json={"id": "cgt-1", "status": "running"})

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.cancel(_handle())

    assert "DELETE" not in seen
    assert isinstance(outcome, Running)


@pytest.mark.parametrize("status", ["succeeded", "failed", "expired", "cancelled"])
@pytest.mark.asyncio
async def test_cancel_never_deletes_a_terminal_task(status: str) -> None:
    """终态上 DELETE 会不可逆地删除远端记录，任何情况下都不能发。"""

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(
            200,
            json={
                "id": "cgt-1",
                "status": status,
                "content": {"video_url": "https://cdn.example/out.mp4"},
            },
        )

    driver = _driver(httpx.MockTransport(handler))

    await driver.cancel(_handle())

    assert "DELETE" not in seen


@pytest.mark.asyncio
async def test_cancel_reports_real_status_when_delete_races() -> None:
    """retrieve 到 DELETE 之间任务转成 running 时，ARK 按文档拒绝 DELETE（running 不支持取消）。

    拒绝不是作业失败：必须落到确认查询并返回远端真实状态，让引擎继续轮询——
    否则一个仍在跑（仍在计费）的远端任务会被本地记成终态 FAILED，产物再也拿不到。
    """

    statuses = iter(["queued", "running"])
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(
                400,
                json={"error": {"code": "InvalidParameter", "message": "The task is not in queued status"}},
            )
        return httpx.Response(200, json={"id": "cgt-1", "status": next(statuses)})

    driver = _driver(httpx.MockTransport(handler))

    outcome = await driver.cancel(_handle())

    assert methods == ["GET", "DELETE", "GET"]
    assert isinstance(outcome, Running)


@pytest.mark.asyncio
async def test_submit_timeout_sends_one_request_and_reports_uncertain() -> None:
    """提交零重试：超时只发一次，并以 EXECUTION_UNCERTAIN 收尾，避免重复计费。"""

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    driver = _driver(httpx.MockTransport(handler))
    outcome = await driver.submit(driver.prepare(PROFILE, _video_request()))

    assert attempts == 1
    assert isinstance(outcome, Failed)
    assert outcome.error.code == "EXECUTION_UNCERTAIN"
    assert outcome.error.uncertain is True


@pytest.mark.asyncio
async def test_upload_file_inlines_image_as_data_url(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n-fake"
    path = tmp_path / "in.png"
    path.write_bytes(payload)

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise AssertionError("ARK 内联上传不应发起任何网络请求")

    driver = _driver(httpx.MockTransport(handler))
    result = await driver.upload_file(PROFILE, model="doubao-seedream-4-0", path=path, media_type="image/png")

    assert result == f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


@pytest.mark.asyncio
async def test_upload_file_rejects_video(tmp_path: Path) -> None:
    path = tmp_path / "in.mp4"
    path.write_bytes(b"x")

    driver = _driver(httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    with pytest.raises(PublicDriverOperationError) as excinfo:
        await driver.upload_file(PROFILE, model="doubao-seedance-2-0", path=path, media_type="video/mp4")

    assert excinfo.value.error.code == "UPLOAD_UNSUPPORTED"


@pytest.mark.asyncio
async def test_upload_file_enforces_inline_size_limits(tmp_path: Path) -> None:
    path = tmp_path / "big.wav"
    path.write_bytes(b"0" * (15 * 1024 * 1024 + 1))

    driver = _driver(httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    with pytest.raises(PublicDriverOperationError) as excinfo:
        await driver.upload_file(PROFILE, model="doubao-seedance-2-0", path=path, media_type="audio/wav")

    assert excinfo.value.error.code == "UPLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_materialize_downloads_artifact(tmp_path: Path) -> None:
    body = b"maidock-ark-video"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cdn.example"
        return httpx.Response(200, content=body, headers={"Content-Type": "video/mp4"})

    driver = _driver(httpx.MockTransport(handler))
    destination = tmp_path / "nested" / "out.mp4"

    artifact = await driver.materialize(
        PROFILE,
        url="https://cdn.example/out.mp4",
        destination=destination,
        max_bytes=1024,
    )

    assert destination.read_bytes() == body
    assert artifact.media_type == "video/mp4"
    assert artifact.size == len(body)


@pytest.mark.asyncio
async def test_prepared_operation_from_another_driver_is_rejected() -> None:
    driver = _driver(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    operation = driver.prepare(PROFILE, _video_request())
    foreign = type(operation)(
        driver_key="dashscope.media.v1",
        payload_version=operation.payload_version,
        profile_name=operation.profile_name,
        capability=operation.capability,
        operation_type=operation.operation_type,
        payload=operation.payload,
    )

    with pytest.raises(ValueError, match="不属于 Volcengine ARK"):
        await driver.submit(foreign)


def test_driver_rejects_duplicate_profiles() -> None:
    with pytest.raises(ValueError, match="profile 重复"):
        ArkPublicDriver(client=ArkClient(), profiles=(_profile(), _profile()))
