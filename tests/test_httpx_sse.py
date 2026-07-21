import json
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import cast

import httpx
import pytest

from src.providers.common import httpx as httpx_common
from src.providers.common.httpx import (
    HttpxClientConfig,
    HttpxJsonErrorFactory,
    HttpxProviderError,
    HttpxProviderParseError,
    SseJsonEvent,
    create_async_client,
    stream_sse_json,
)
from tests.support.http import TrackingByteStream

type AsyncHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
type StreamErrorFactory = Callable[[], httpx.HTTPError]


def _client(handler: AsyncHandler) -> httpx.AsyncClient:
    return create_async_client(
        HttpxClientConfig(
            base_url="https://example.com/api/v1",
            default_query={"base": "1"},
        ),
        transport=httpx.MockTransport(handler),
    )


async def _collect_events(
    client: httpx.AsyncClient,
    *,
    max_retries: int = 0,
    retry_interval: float = 0.0,
    error_factory: HttpxJsonErrorFactory | None = None,
) -> list[SseJsonEvent]:
    return [
        event
        async for event in stream_sse_json(
            client,
            "stream",
            json_body={"stream": True},
            provider_label="TestProvider",
            max_retries=max_retries,
            retry_interval=retry_interval,
            error_factory=error_factory,
        )
    ]


def _read_timeout() -> httpx.HTTPError:
    return httpx.ReadTimeout("stream timed out")


def _connect_error() -> httpx.HTTPError:
    return httpx.ConnectError("stream disconnected")


@pytest.mark.asyncio
async def test_stream_sse_json_sends_request_and_parses_protocol_fields() -> None:
    requests: list[httpx.Request] = []
    stream = TrackingByteStream(
        [
            b": heartbeat\r\n"
            b"\r\n"
            b"event:first\r\n"
            b"status:\r\n"
            b'data: {"first":true}\r\n'
            b"\r\n"
            b"event: second\n"
            b"status: invalid\n"
            b'data: {"second":true}\n'
            b"\n"
            b"event: final\n"
            b"status: 201\n"
            b'data: {"message":"hel"\n'
            b'data: ,"index":1}'
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=stream, headers={"Content-Type": "text/event-stream"})

    async with _client(handler) as client:
        events = [
            event
            async for event in stream_sse_json(
                client,
                "stream",
                json_body={"stream": True, "model": "test-model"},
                headers={"X-Trace": "trace-id"},
                query={"debug": True, "metadata": {"source": "test"}, "skip": None},
                provider_label="TestProvider",
            )
        ]

    assert [(event.event, event.status, event.data) for event in events] == [
        ("first", None, {"first": True}),
        ("second", None, {"second": True}),
        ("final", 201, {"message": "hel", "index": 1}),
    ]
    assert stream.close_calls == 1
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/v1/stream"
    assert request.url.params["base"] == "1"
    assert request.url.params["debug"] == "True"
    assert request.url.params["metadata"] == '{"source":"test"}'
    assert "skip" not in request.url.params
    assert request.headers["X-Trace"] == "trace-id"
    assert json.loads(request.content.decode()) == {"stream": True, "model": "test-model"}


@pytest.mark.asyncio
async def test_stream_sse_json_stops_at_done_marker() -> None:
    stream = TrackingByteStream([b'data: {"index":1}\n\ndata: [DONE]\n\ndata: {"index":2}\n\n'])

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        events = await _collect_events(client)

    assert [event.data for event in events] == [{"index": 1}]
    assert stream.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ["not-json", "[]"])
async def test_stream_sse_json_rejects_invalid_event_payload_and_closes_stream(data: str) -> None:
    stream = TrackingByteStream([f"data: {data}\n\n".encode()])

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderParseError, match="TestProvider"):
            await _collect_events(client)

    assert stream.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminated", [True, False])
async def test_stream_sse_json_event_error_factory_receives_protocol_context_and_closes_stream(
    terminated: bool,
) -> None:
    suffix = "\n\n" if terminated else ""
    stream = TrackingByteStream([f'event: provider.error\nstatus: 460\ndata: {{"code":"bad"}}{suffix}'.encode()])
    expected_error = HttpxProviderError("provider-specific", status_code=460)
    calls: list[tuple[dict, int | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        calls.append((payload, status, event))
        return expected_error

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await _collect_events(client, error_factory=error_factory)

    assert captured.value is expected_error
    assert calls == [({"code": "bad"}, 460, "provider.error")]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_http_error_factory_precedes_generic_error() -> None:
    stream = TrackingByteStream([b'{"code":"provider_error"}'])
    expected_error = HttpxProviderError("provider-specific", status_code=460)
    calls: list[tuple[dict, int | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, stream=stream)

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        calls.append((payload, status, event))
        return expected_error

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await _collect_events(client, error_factory=error_factory)

    assert captured.value is expected_error
    assert calls == [({"code": "provider_error"}, 400, None)]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_http_error_factory_can_decline_or_skip_payload() -> None:
    json_stream = TrackingByteStream([b'{"code":"generic_error"}'])
    text_stream = TrackingByteStream([b"not-json"])
    streams = [json_stream, text_stream]
    calls: list[tuple[dict, int | None, str | None]] = []
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        stream = streams[attempts]
        attempts += 1
        return httpx.Response(400, stream=stream)

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        calls.append((payload, status, event))
        return None

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as json_error:
            await _collect_events(client, error_factory=error_factory)
        with pytest.raises(HttpxProviderError) as text_error:
            await _collect_events(client, error_factory=error_factory)

    assert json_error.value.status_code == 400
    assert text_error.value.status_code == 400
    assert calls == [({"code": "generic_error"}, 400, None)]
    assert json_stream.close_calls == 1
    assert text_stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_reports_non_json_http_error_and_closes_stream() -> None:
    stream = TrackingByteStream([b"upstream unavailable"])

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(502, stream=stream)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await _collect_events(client)

    assert captured.value.status_code == 502
    assert "upstream unavailable" in str(captured.value)
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_event_error_factory_can_decline_terminated_and_final_events() -> None:
    stream = TrackingByteStream([b'id: ignored\nevent: first\ndata: {"index":1}\n\nevent: final\ndata: {"index":2}'])
    calls: list[tuple[dict, int | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        calls.append((payload, status, event))
        return None

    async with _client(handler) as client:
        events = await _collect_events(client, error_factory=error_factory)

    assert [event.data for event in events] == [{"index": 1}, {"index": 2}]
    assert calls == [
        ({"index": 1}, None, "first"),
        ({"index": 2}, None, "final"),
    ]
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_retries_status_and_closes_each_response() -> None:
    attempts = 0
    first_stream = TrackingByteStream([b'{"error":"rate limited"}'])
    second_stream = TrackingByteStream([b'data: {"ok":true}\n\n'])

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, stream=first_stream)
        return httpx.Response(200, stream=second_stream)

    async with _client(handler) as client:
        events = await _collect_events(client, max_retries=1)

    assert attempts == 2
    assert [event.data for event in events] == [{"ok": True}]
    assert first_stream.close_calls == 1
    assert second_stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_retry_header_can_disable_status_retry() -> None:
    attempts = 0
    stream = TrackingByteStream([b'{"error":"stop"}'])

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        return httpx.Response(500, stream=stream, headers={"x-should-retry": "false"})

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await _collect_events(client, max_retries=1)

    assert attempts == 1
    assert captured.value.status_code == 500
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_waits_configured_interval_before_status_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "retry"})
        return httpx.Response(200, text='data: {"ok":true}\n\n')

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        events = await _collect_events(client, max_retries=1, retry_interval=0.25)

    assert attempts == 2
    assert delays == [0.25]
    assert [event.data for event in events] == [{"ok": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_stream_sse_json_retries_stream_failure_before_first_event(
    error_factory: StreamErrorFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    streams: list[TrackingByteStream] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            stream = TrackingByteStream([], error=error_factory(), error_after_chunks=0)
        else:
            stream = TrackingByteStream([b'data: {"ok":true}\n\n'])
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        events = await _collect_events(client, max_retries=1, retry_interval=0.5)

    assert attempts == 2
    assert delays == [0.5]
    assert [event.data for event in events] == [{"ok": True}]
    assert [stream.close_calls for stream in streams] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_stream_sse_json_wraps_stream_failure_after_retry_exhaustion(
    error_factory: StreamErrorFactory,
) -> None:
    attempts = 0
    streams: list[TrackingByteStream] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        stream = TrackingByteStream([], error=error_factory(), error_after_chunks=0)
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await _collect_events(client, max_retries=1)

    assert attempts == 2
    assert isinstance(captured.value.__cause__, type(error_factory()))
    assert [stream.close_calls for stream in streams] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_stream_sse_json_does_not_retry_after_emitting_event(
    error_factory: StreamErrorFactory,
) -> None:
    attempts = 0
    stream = TrackingByteStream(
        [b'data: {"index":1}\n\n'],
        error=error_factory(),
        error_after_chunks=1,
    )
    emitted: list[SseJsonEvent] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            async for event in stream_sse_json(
                client,
                "stream",
                json_body={},
                provider_label="TestProvider",
                max_retries=2,
            ):
                emitted.append(event)

    assert attempts == 1
    assert [event.data for event in emitted] == [{"index": 1}]
    assert isinstance(captured.value.__cause__, type(error_factory()))
    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_stream_sse_json_closes_response_when_consumer_stops_early() -> None:
    stream = TrackingByteStream([b'data: {"index":1}\n\ndata: {"index":2}\n\n'])

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        iterator = cast(
            AsyncGenerator[SseJsonEvent, None],
            stream_sse_json(client, "stream", json_body={}, provider_label="TestProvider"),
        )
        first_event = await anext(iterator)
        await iterator.aclose()

    assert first_event.data == {"index": 1}
    assert stream.close_calls == 1
