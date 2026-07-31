from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import json

import httpx
import pytest

from src.clients.common import (
    ClientConnectionError,
    ClientHttpError,
    ClientProtocolError,
    ClientTimeoutError,
    HttpConnection,
    HttpSession,
    RetryPolicy,
    SharedHttpClient,
)
from tests.support.http import TrackingByteStream


@asynccontextmanager
async def _session(
    handler: httpx.AsyncBaseTransport,
    *,
    connection: HttpConnection | None = None,
) -> AsyncIterator[HttpSession]:
    client = SharedHttpClient(transport=handler)
    async with client.session(
        connection
        or HttpConnection(
            base_url="https://example.com/api/v1",
            default_headers=(("Authorization", "Bearer secret"),),
            default_query=(("base", "1"),),
        )
    ) as session:
        yield session
    await client.aclose()


@pytest.mark.asyncio
async def test_request_json_sends_connection_and_request_values() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with _session(httpx.MockTransport(handler)) as session:
        payload = await session.request_json(
            "POST",
            "resource",
            subject="测试资源",
            json_body={"hello": "world"},
            headers={"X-Trace": "trace-id"},
            query={"debug": True, "nested": {"items": [1, False]}, "skip": None},
        )

    assert payload == {"ok": True}
    request = requests[0]
    assert request.url.path == "/api/v1/resource"
    assert request.url.params["base"] == "1"
    assert request.url.params["debug"] == "True"
    assert request.url.params["nested"] == '{"items":[1,false]}'
    assert "skip" not in request.url.params
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["x-trace"] == "trace-id"
    assert json.loads(request.content) == {"hello": "world"}


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["not-json", "[]"])
async def test_request_json_rejects_invalid_success_payload(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text=body)

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientProtocolError, match="测试资源"):
            await session.request_json("POST", "resource", subject="测试资源", json_body={})


@pytest.mark.asyncio
async def test_request_json_uses_native_error_factory() -> None:
    expected = ClientHttpError("供应商原生错误", status_code=460)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"code": "BadRequest"})

    def error_factory(payload: dict, status: int | None, event: str | None) -> ClientHttpError:
        assert payload == {"code": "BadRequest"}
        assert status == 400
        assert event is None
        return expected

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            await session.request_json(
                "POST",
                "resource",
                subject="测试资源",
                json_body={},
                error_factory=error_factory,
            )

    assert captured.value is expected


@pytest.mark.asyncio
async def test_request_json_error_includes_upstream_detail() -> None:
    """兜底 ClientHttpError 必须带上上游 error.code/error.message，否则裸 400 没法排查。"""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": "The parameter messages.partial is required",
                }
            },
        )

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            await session.request_json("POST", "resource", subject="测试资源", json_body={})

    assert str(captured.value) == (
        "测试资源 请求失败: HTTP 400: InvalidParameter: The parameter messages.partial is required"
    )
    assert captured.value.status_code == 400


@pytest.mark.asyncio
async def test_request_json_error_detail_reads_top_level_fields_and_truncates() -> None:
    long_message = "坏" * 400

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"code": "TooLong", "message": long_message})

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            await session.request_json("POST", "resource", subject="测试资源", json_body={})

    message = str(captured.value)
    assert message.startswith("测试资源 请求失败: HTTP 400: TooLong: 坏")
    assert message.endswith("…")
    assert len(message) < 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected_suffix"),
    [
        ({"error": "invalid_api_key"}, ": invalid_api_key"),  # 网关常见：error 直接是字符串
        ({"error": {"code": 400, "message": ""}}, ": 400"),  # 数字 code + 空 message
        ({"error": {}, "message": "顶层兜底"}, ": 顶层兜底"),  # error object 为空退回顶层
        ({"error": True}, ""),  # bool 不是可展示详情，保持裸状态码
    ],
)
async def test_request_json_error_detail_covers_non_object_error_shapes(body: dict, expected_suffix: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json=body)

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            await session.request_json("POST", "resource", subject="测试资源", json_body={})

    assert str(captured.value) == f"测试资源 请求失败: HTTP 400{expected_suffix}"


@pytest.mark.asyncio
async def test_request_json_error_with_non_json_body_keeps_status_message() -> None:
    """错误路径的响应体尽力解析：网关的 HTML 错误页不该把 HTTP 失败盖成协议错误。"""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, text="<html>bad request</html>")

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            await session.request_json("POST", "resource", subject="测试资源", json_body={})

    assert not isinstance(captured.value, ClientProtocolError)
    assert str(captured.value) == "测试资源 请求失败: HTTP 400"


@pytest.mark.asyncio
async def test_request_json_retries_retryable_status_with_non_json_body() -> None:
    """可重试的 5xx 即便错误体不是 JSON 也必须先走重试，而不是先解析后炸协议错误。"""

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(502, text="<html>bad gateway</html>")
        return httpx.Response(200, json={"ok": True})

    async with _session(httpx.MockTransport(handler)) as session:
        payload = await session.request_json(
            "POST",
            "resource",
            subject="测试资源",
            json_body={},
            retry=RetryPolicy(max_retries=1),
        )

    assert payload == {"ok": True}
    assert attempts == 2


@pytest.mark.asyncio
async def test_sse_open_error_includes_upstream_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={"error": {"code": "AuthenticationError", "message": "bad key"}})

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientHttpError) as captured:
            async for _ in session.stream_sse_json("stream", subject="测试资源", json_body={}):
                pass

    assert str(captured.value) == "测试资源 SSE 请求失败: HTTP 401: AuthenticationError: bad key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "retry_header", "expected_attempts"),
    [(429, None, 2), (400, "true", 2), (500, "false", 1)],
)
async def test_request_json_honors_resource_policy_and_retry_header(
    status_code: int,
    retry_header: str | None,
    expected_attempts: int,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            headers = {"x-should-retry": retry_header} if retry_header is not None else None
            return httpx.Response(status_code, headers=headers, json={"error": "retry"})
        return httpx.Response(200, json={"ok": True})

    async with _session(httpx.MockTransport(handler)) as session:
        if expected_attempts == 1:
            with pytest.raises(ClientHttpError):
                await session.request_json(
                    "POST",
                    "resource",
                    subject="测试资源",
                    json_body={},
                    retry=RetryPolicy(max_retries=1),
                )
        else:
            assert await session.request_json(
                "POST",
                "resource",
                subject="测试资源",
                json_body={},
                retry=RetryPolicy(max_retries=1),
            ) == {"ok": True}

    assert attempts == expected_attempts


@pytest.mark.asyncio
async def test_request_timeout_reports_uncertain_without_implicit_retry() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientTimeoutError) as captured:
            await session.request_json(
                "POST",
                "submit",
                subject="生成提交",
                json_body={},
                retry=RetryPolicy(uncertain_on_timeout=True),
            )

    assert captured.value.uncertain is True
    assert attempts == 1


@pytest.mark.asyncio
async def test_multipart_owns_content_type_and_retries_safe_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": "retry"})
        return httpx.Response(200, json={"ok": True})

    async with _session(httpx.MockTransport(handler)) as session:
        response = await session.request_multipart(
            "upload",
            subject="上传",
            form_data={"model": "asr"},
            files={"file": ("audio.wav", b"audio")},
            retry=RetryPolicy(max_retries=1),
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert requests[1].headers["content-type"].startswith("multipart/form-data; boundary=")


@pytest.mark.asyncio
async def test_sse_parses_protocol_fields_and_stops_at_done() -> None:
    stream = TrackingByteStream(
        [
            b': heartbeat\n\nevent: delta\nstatus: 206\ndata: {"index":1}\n\n'
            b'data: {"index":2}\n\ndata: [DONE]\n\ndata: {"index":3}\n\n'
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"stream": True}
        return httpx.Response(200, stream=stream)

    async with _session(httpx.MockTransport(handler)) as session:
        events = [
            event
            async for event in session.stream_sse_json(
                "stream",
                subject="流式资源",
                json_body={"stream": True},
            )
        ]

    assert [(event.event, event.status, event.data) for event in events] == [
        ("delta", 206, {"index": 1}),
        (None, None, {"index": 2}),
    ]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_sse_retries_before_first_event_but_not_after_emission() -> None:
    attempts = 0
    streams: list[TrackingByteStream] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            stream = TrackingByteStream(
                [],
                error=httpx.ConnectError("disconnect", request=request),
                error_after_chunks=0,
            )
        else:
            stream = TrackingByteStream(
                [b'data: {"index":1}\n\n'],
                error=httpx.ConnectError("disconnect", request=request),
                error_after_chunks=1,
            )
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    emitted: list[dict] = []
    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientConnectionError):
            async for event in session.stream_sse_json(
                "stream",
                subject="流式资源",
                json_body={},
                retry=RetryPolicy(max_retries=2),
            ):
                emitted.append(event.data)

    assert attempts == 2
    assert emitted == [{"index": 1}]
    assert [stream.close_calls for stream in streams] == [1, 1]


@pytest.mark.asyncio
async def test_download_rejects_oversize_and_removes_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"
    stream = TrackingByteStream([b"123", b"456"])

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, headers={"content-type": "application/octet-stream"}, stream=stream)

    async with _session(httpx.MockTransport(handler)) as session:
        with pytest.raises(ClientProtocolError, match="最大字节数"):
            await session.download(
                "https://cdn.example/artifact.bin",
                destination,
                subject="下载",
                max_bytes=5,
                retry=RetryPolicy(max_retries=1),
            )

    assert not destination.exists()
    assert not (tmp_path / ".artifact.bin.part").exists()
