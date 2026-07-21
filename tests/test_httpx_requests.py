import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from src.providers.common import httpx as httpx_common
from src.providers.common.httpx import (
    HttpxClientConfig,
    HttpxProviderError,
    HttpxProviderParseError,
    create_async_client,
    post_json,
    post_multipart,
)

type AsyncHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
type HttpErrorFactory = Callable[[httpx.Request], httpx.HTTPError]


def _client(handler: AsyncHandler) -> httpx.AsyncClient:
    return create_async_client(
        HttpxClientConfig(
            base_url="https://example.com/api/v1",
            default_query={"base": "1"},
        ),
        transport=httpx.MockTransport(handler),
    )


def _read_timeout(request: httpx.Request) -> httpx.HTTPError:
    return httpx.ReadTimeout("read timed out", request=request)


def _connect_error(request: httpx.Request) -> httpx.HTTPError:
    return httpx.ConnectError("connection failed", request=request)


@pytest.mark.asyncio
async def test_post_json_sends_body_headers_and_serialized_query() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        payload = await post_json(
            client,
            "services/test",
            json_body={"hello": "world"},
            headers={"X-Trace": "trace-id"},
            query={"debug": True, "nested": {"items": [1, False]}, "skip": None},
            provider_label="TestProvider",
        )

    assert payload == {"ok": True}
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/api/v1/services/test"
    assert request.url.params["base"] == "1"
    assert request.url.params["debug"] == "True"
    assert request.url.params["nested"] == '{"items":[1,false]}'
    assert "skip" not in request.url.params
    assert request.headers["X-Trace"] == "trace-id"
    assert json.loads(request.content.decode("utf-8")) == {"hello": "world"}


@pytest.mark.asyncio
async def test_post_json_rejects_success_with_invalid_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text="not-json")

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderParseError, match="TestProvider"):
            await post_json(client, "response", json_body={}, provider_label="TestProvider")


@pytest.mark.asyncio
async def test_post_json_rejects_success_with_non_object_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=["not", "an", "object"])

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderParseError, match="TestProvider"):
            await post_json(client, "response", json_body={}, provider_label="TestProvider")


@pytest.mark.asyncio
async def test_post_json_preserves_status_and_sanitizes_json_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"error": {"message": "bad request", "api_key": "secret-key"}})

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_json(client, "response", json_body={}, provider_label="TestProvider")

    assert captured.value.status_code == 400
    assert "bad request" in str(captured.value)
    assert "secret-key" not in str(captured.value)


@pytest.mark.asyncio
async def test_post_json_reports_non_json_and_non_object_error_responses() -> None:
    responses = [
        httpx.Response(502, text="upstream unavailable"),
        httpx.Response(503, json=["unexpected"]),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return responses.pop(0)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as text_error:
            await post_json(client, "response", json_body={}, provider_label="TestProvider")
        with pytest.raises(HttpxProviderError) as list_error:
            await post_json(client, "response", json_body={}, provider_label="TestProvider")

    assert text_error.value.status_code == 502
    assert "upstream unavailable" in str(text_error.value)
    assert list_error.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 400])
async def test_post_json_error_factory_precedes_generic_status_handling(status_code: int) -> None:
    expected_error = HttpxProviderError("provider-specific", status_code=460)
    factory_calls: list[tuple[dict, int | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status_code, json={"code": "provider_error"})

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        factory_calls.append((payload, status, event))
        return expected_error

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_json(
                client,
                "response",
                json_body={},
                provider_label="TestProvider",
                error_factory=error_factory,
            )

    assert captured.value is expected_error
    assert factory_calls == [({"code": "provider_error"}, status_code, None)]


@pytest.mark.asyncio
async def test_post_json_error_factory_can_decline_error_payload() -> None:
    factory_calls: list[tuple[dict, int | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, json={"code": "generic_error"})

    def error_factory(payload: dict, status: int | None, event: str | None) -> HttpxProviderError | None:
        factory_calls.append((payload, status, event))
        return None

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_json(
                client,
                "response",
                json_body={},
                provider_label="TestProvider",
                error_factory=error_factory,
            )

    assert captured.value.status_code == 400
    assert factory_calls == [({"code": "generic_error"}, 400, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
async def test_post_json_retries_retryable_statuses(status_code: int) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(status_code, json={"error": "retry"})
        return httpx.Response(200, json={"attempt": attempts})

    async with _client(handler) as client:
        payload = await post_json(
            client,
            "response",
            json_body={},
            provider_label="TestProvider",
            max_retries=1,
        )

    assert attempts == 2
    assert payload == {"attempt": 2}


@pytest.mark.asyncio
async def test_post_json_retry_header_can_force_retry_for_ordinary_client_error() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(400, json={"error": "retry"}, headers={"x-should-retry": " TRUE "})
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        payload = await post_json(
            client,
            "response",
            json_body={},
            provider_label="TestProvider",
            max_retries=1,
        )

    assert attempts == 2
    assert payload == {"ok": True}


@pytest.mark.asyncio
async def test_post_json_retry_header_can_disable_retry_for_server_error() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        return httpx.Response(500, json={"error": "stop"}, headers={"x-should-retry": "false"})

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_json(
                client,
                "response",
                json_body={},
                provider_label="TestProvider",
                max_retries=1,
            )

    assert attempts == 1
    assert captured.value.status_code == 500


@pytest.mark.asyncio
async def test_post_json_invalid_retry_header_uses_status_default() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(500, json={"error": "retry"}, headers={"x-should-retry": "sometimes"})
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        payload = await post_json(
            client,
            "response",
            json_body={},
            provider_label="TestProvider",
            max_retries=1,
        )

    assert attempts == 2
    assert payload == {"ok": True}


@pytest.mark.asyncio
async def test_post_json_clamps_negative_retry_count_to_zero() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        return httpx.Response(429, json={"error": "rate limited"})

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError):
            await post_json(
                client,
                "response",
                json_body={},
                provider_label="TestProvider",
                max_retries=-3,
            )

    assert attempts == 1


@pytest.mark.asyncio
async def test_post_json_waits_configured_interval_before_status_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        await post_json(
            client,
            "response",
            json_body={},
            provider_label="TestProvider",
            max_retries=1,
            retry_interval=0.25,
        )

    assert delays == [0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_post_json_retries_transport_error_before_success(
    error_factory: HttpErrorFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_factory(request)
        return httpx.Response(200, json={"ok": True})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        payload = await post_json(
            client,
            "response",
            json_body={},
            provider_label="TestProvider",
            max_retries=1,
            retry_interval=0.5,
        )

    assert attempts == 2
    assert delays == [0.5]
    assert payload == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_post_json_wraps_transport_error_after_retry_exhaustion(error_factory: HttpErrorFactory) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise error_factory(request)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_json(
                client,
                "response",
                json_body={},
                provider_label="TestProvider",
                max_retries=1,
            )

    assert attempts == 2
    assert isinstance(captured.value.__cause__, type(error_factory(httpx.Request("POST", "https://example.com"))))


@pytest.mark.asyncio
async def test_post_multipart_replaces_json_content_type_and_preserves_request_fields() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await post_multipart(
            client,
            "audio/transcriptions",
            form_data={"model": "whisper-1"},
            files={"file": ("sample.wav", b"RIFF-test")},
            headers={"CONTENT-TYPE": "application/json", "X-Trace": "trace-id"},
            query={"language": "zh", "metadata": {"source": "test"}},
            provider_label="TestProvider",
        )

    assert response.status_code == 200
    assert response.is_closed
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert request.headers["X-Trace"] == "trace-id"
    assert request.url.params["base"] == "1"
    assert request.url.params["language"] == "zh"
    assert request.url.params["metadata"] == '{"source":"test"}'
    assert b'name="model"' in request.content
    assert b"whisper-1" in request.content
    assert b'filename="sample.wav"' in request.content
    assert b"RIFF-test" in request.content


@pytest.mark.asyncio
async def test_post_multipart_retries_retryable_status() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        response = await post_multipart(
            client,
            "audio/transcriptions",
            form_data={"model": "whisper-1"},
            files={"file": ("sample.wav", b"RIFF-test")},
            provider_label="TestProvider",
            max_retries=1,
        )

    assert attempts == 2
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_post_multipart_waits_configured_interval_before_status_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        await post_multipart(
            client,
            "audio/transcriptions",
            form_data={},
            files={"file": ("sample.wav", b"RIFF-test")},
            provider_label="TestProvider",
            max_retries=1,
            retry_interval=0.25,
        )

    assert attempts == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_post_multipart_retry_header_can_disable_retry() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        del request
        attempts += 1
        return httpx.Response(500, text="stop", headers={"x-should-retry": "false"})

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_multipart(
                client,
                "audio/transcriptions",
                form_data={},
                files={"file": ("sample.wav", b"RIFF-test")},
                provider_label="TestProvider",
                max_retries=1,
            )

    assert attempts == 1
    assert captured.value.status_code == 500
    assert "stop" in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_post_multipart_retries_transport_error_before_success(
    error_factory: HttpErrorFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_factory(request)
        return httpx.Response(200, json={"ok": True})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(httpx_common.asyncio, "sleep", record_sleep)
    async with _client(handler) as client:
        response = await post_multipart(
            client,
            "audio/transcriptions",
            form_data={},
            files={"file": ("sample.wav", b"RIFF-test")},
            provider_label="TestProvider",
            max_retries=1,
            retry_interval=0.5,
        )

    assert attempts == 2
    assert delays == [0.5]
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("error_factory", [_read_timeout, _connect_error])
async def test_post_multipart_wraps_transport_error_after_retry_exhaustion(
    error_factory: HttpErrorFactory,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise error_factory(request)

    async with _client(handler) as client:
        with pytest.raises(HttpxProviderError) as captured:
            await post_multipart(
                client,
                "audio/transcriptions",
                form_data={},
                files={"file": ("sample.wav", b"RIFF-test")},
                provider_label="TestProvider",
                max_retries=1,
            )

    assert attempts == 2
    assert isinstance(captured.value.__cause__, type(error_factory(httpx.Request("POST", "https://example.com"))))
