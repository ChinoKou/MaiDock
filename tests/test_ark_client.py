from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from src.clients.ark import (
    ARK_CONTENT_GENERATION_TASKS_PATH,
    ARK_IMAGES_GENERATIONS_PATH,
    ArkClient,
    ArkClientError,
    ArkConnection,
    ArkSession,
    ark_error_factory,
)
from src.clients.common import NO_RETRY, ClientHttpError, HttpConnection, JsonObject, RetryPolicy
from src.clients.families import JsonResourceRequest

BASE_URL = "https://ark.example/api/v3"


def _connection(*, retry: RetryPolicy = NO_RETRY, safe_retry: RetryPolicy = NO_RETRY) -> ArkConnection:
    return ArkConnection(
        http=HttpConnection(
            base_url=BASE_URL,
            default_headers=(("Authorization", "Bearer ark-key"),),
        ),
        retry=retry,
        responses_path="responses",
        embeddings_path="embeddings/multimodal",
        audio_transcriptions_path="responses",
        tokenization_path="tokenization",
        safe_retry=safe_retry,
    )


@asynccontextmanager
async def _session(
    handler: httpx.AsyncBaseTransport,
    **kwargs: RetryPolicy,
) -> AsyncIterator[ArkSession]:
    client = ArkClient(transport=handler)
    try:
        async with client.session(_connection(**kwargs)) as session:
            yield session
    finally:
        await client.aclose()


def test_ark_connection_media_paths_default_without_touching_existing_call_sites() -> None:
    """媒体路径与 safe_retry 都是追加的默认值字段，既有 Host 构造点无需改动。"""

    connection = ArkConnection(
        http=HttpConnection(base_url=BASE_URL),
        retry=RetryPolicy(),
        responses_path="responses",
        embeddings_path="embeddings/multimodal",
        audio_transcriptions_path="responses",
        tokenization_path="tokenization",
    )

    assert connection.images_generations_path == ARK_IMAGES_GENERATIONS_PATH == "images/generations"
    assert connection.content_generation_tasks_path == ARK_CONTENT_GENERATION_TASKS_PATH
    assert connection.content_generation_tasks_path == "contents/generations/tasks"
    assert connection.safe_retry == NO_RETRY


def test_ark_error_factory_ignores_successful_payload() -> None:
    assert ark_error_factory({"data": [{"url": "https://cdn.example/a.png"}]}, 200, None) is None


def test_ark_error_factory_detects_top_level_error_on_2xx() -> None:
    """ARK 会用 200 + 顶层 error 表达整单失败，不看状态码也必须判出来。"""

    error = ark_error_factory(
        {"error": {"code": "InvalidParameter", "message": "size 不合法"}},
        200,
        None,
    )

    assert isinstance(error, ArkClientError)
    assert error.code == "InvalidParameter"
    assert error.upstream_message == "size 不合法"
    assert error.status_code == 200
    assert error.retryable is False
    assert "status=200" in str(error)
    assert "code=InvalidParameter" in str(error)


def test_ark_error_factory_ignores_per_item_error() -> None:
    """data[] 里的逐项 error 是部分成功，由 driver 转 warnings，不算整单失败。"""

    payload: JsonObject = {
        "data": [
            {"url": "https://cdn.example/ok.png"},
            {"error": {"code": "ContentFiltered", "message": "命中审核"}},
        ]
    }

    assert ark_error_factory(payload, 200, None) is None


def test_ark_error_factory_falls_back_when_body_has_no_error() -> None:
    error = ark_error_factory({}, 404, None)

    assert isinstance(error, ArkClientError)
    assert error.code == "UPSTREAM_ERROR"
    assert error.upstream_message == "上游未提供错误消息"
    assert error.retryable is False


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (403, False), (408, True), (409, True), (429, True), (500, True), (503, True)],
)
def test_ark_error_factory_marks_transient_statuses_retryable(status_code: int, retryable: bool) -> None:
    error = ark_error_factory({"error": {"code": "X", "message": "y"}}, status_code, None)

    assert isinstance(error, ArkClientError)
    assert error.retryable is retryable


def test_ark_error_factory_reports_request_id_when_present() -> None:
    error = ark_error_factory(
        {"error": {"code": "X", "message": "y"}, "request_id": "req-1"},
        500,
        None,
    )

    assert isinstance(error, ArkClientError)
    assert error.request_id == "req-1"
    assert "request_id=req-1" in str(error)


@pytest.mark.asyncio
async def test_images_generations_posts_to_exact_path_with_auth() -> None:
    seen: list[tuple[str, str, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("Authorization")))
        return httpx.Response(200, json={"model": "seedream", "data": [{"url": "https://cdn.example/a.png"}]})

    async with _session(httpx.MockTransport(handler)) as session:
        payload = await session.images_generations.create(
            JsonResourceRequest(body={"model": "seedream", "prompt": "cat"}),
            retry=session.retry,
        )

    assert seen == [("POST", "/api/v3/images/generations", "Bearer ark-key")]
    assert payload["data"] == [{"url": "https://cdn.example/a.png"}]


@pytest.mark.asyncio
async def test_images_generations_raises_ark_error_on_2xx_top_level_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"error": {"code": "SensitiveContent", "message": "命中审核"}})

    with pytest.raises(ArkClientError) as excinfo:
        async with _session(httpx.MockTransport(handler)) as session:
            await session.images_generations.create(
                JsonResourceRequest(body={"model": "seedream", "prompt": "x"}),
                retry=session.retry,
            )

    assert excinfo.value.code == "SensitiveContent"


@pytest.mark.asyncio
async def test_content_generation_tasks_uses_expected_methods_and_paths() -> None:
    seen: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"id": "task-1", "status": "queued"})

    async with _session(httpx.MockTransport(handler)) as session:
        created = await session.content_generation_tasks.create(
            {"model": "seedance", "content": []},
            retry=session.retry,
        )
        await session.content_generation_tasks.retrieve("task-1", retry=session.safe_retry)
        await session.content_generation_tasks.delete("task-1", retry=NO_RETRY)

    assert seen == [
        ("POST", "/api/v3/contents/generations/tasks"),
        ("GET", "/api/v3/contents/generations/tasks/task-1"),
        ("DELETE", "/api/v3/contents/generations/tasks/task-1"),
    ]
    assert created["id"] == "task-1"


@pytest.mark.asyncio
async def test_task_delete_accepts_empty_response_body() -> None:
    """文档写明"本接口无返回参数"：空体是正常成功，不能当协议错误。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        return httpx.Response(200)

    async with _session(httpx.MockTransport(handler)) as session:
        assert await session.content_generation_tasks.delete("task-1", retry=NO_RETRY) is None


@pytest.mark.asyncio
async def test_task_delete_still_surfaces_error_body_when_present() -> None:
    """空体要放过，但真带了错误体时仍然要走 ark_error_factory。"""

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"error": {"code": "InvalidStatus", "message": "running 不可取消"}})

    with pytest.raises(ArkClientError) as excinfo:
        async with _session(httpx.MockTransport(handler)) as session:
            await session.content_generation_tasks.delete("task-1", retry=NO_RETRY)

    assert excinfo.value.code == "InvalidStatus"


@pytest.mark.asyncio
async def test_task_delete_falls_back_to_status_error_without_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(409)

    with pytest.raises(ClientHttpError) as excinfo:
        async with _session(httpx.MockTransport(handler)) as session:
            await session.content_generation_tasks.delete("task-1", retry=NO_RETRY)

    assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_content_generation_tasks_surface_ark_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, json={"error": {"code": "TaskNotFound", "message": "任务不存在"}})

    with pytest.raises(ArkClientError) as excinfo:
        async with _session(httpx.MockTransport(handler)) as session:
            await session.content_generation_tasks.retrieve("missing", retry=session.safe_retry)

    assert excinfo.value.code == "TaskNotFound"
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_artifacts_download_writes_file_and_reports_digest(tmp_path: Path) -> None:
    body = b"maidock-ark-artifact"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cdn.example"
        return httpx.Response(200, content=body, headers={"Content-Type": "video/mp4"})

    destination = tmp_path / "out.mp4"
    async with _session(httpx.MockTransport(handler)) as session:
        artifact = await session.artifacts.download(
            "https://cdn.example/out.mp4",
            destination,
            max_bytes=1024,
            retry=session.safe_retry,
        )

    assert destination.read_bytes() == body
    assert artifact.size == len(body)
    assert artifact.media_type == "video/mp4"
    assert artifact.path == destination


@pytest.mark.asyncio
async def test_artifacts_download_rejects_non_https(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise AssertionError("非 HTTPS 应在发起请求前就被拒绝")

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ValueError, match="HTTPS"):
            await session.artifacts.download(
                "http://cdn.example/out.mp4",
                tmp_path / "out.mp4",
                max_bytes=1024,
                retry=session.safe_retry,
            )


@pytest.mark.asyncio
async def test_session_exposes_separate_submit_and_safe_retry_policies() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={})

    submit_retry = RetryPolicy(max_retries=0, uncertain_on_timeout=True)
    safe_retry = RetryPolicy(max_retries=3)

    async with _session(httpx.MockTransport(handler), retry=submit_retry, safe_retry=safe_retry) as session:
        assert session.retry == submit_retry
        assert session.safe_retry == safe_retry


@pytest.mark.asyncio
async def test_existing_text_resources_keep_their_paths() -> None:
    """媒体资源是新增的，四个既有文本资源的路径与主题不受影响。"""

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    async with _session(httpx.MockTransport(handler)) as session:
        for resource in (session.responses, session.embeddings, session.tokenization):
            await resource.create(JsonResourceRequest(body={}), retry=session.retry)

    assert seen == ["/api/v3/responses", "/api/v3/embeddings/multimodal", "/api/v3/tokenization"]
