from pathlib import Path

import json

import httpx
import pytest

from src.clients.common import HttpConnection, RetryPolicy
from src.clients.dashscope import DashScopeClient, DashScopeConnection, DashScopePaths
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
    Running,
)
from src.public_api.providers.dashscope import DashScopeMediaProfile, DashScopePublicDriver


def _connection(*, safe_retries: int = 1) -> DashScopeConnection:
    return DashScopeConnection(
        http=HttpConnection(
            base_url="https://dashscope.example/api/v1",
            default_headers=(
                ("Authorization", "Bearer secret"),
                ("X-DashScope-WorkSpace", "workspace"),
            ),
        ),
        retry=RetryPolicy(),
        safe_retry=RetryPolicy(max_retries=safe_retries),
        paths=DashScopePaths(
            text_generation="services/aigc/text-generation/generation",
            multimodal_generation="services/aigc/multimodal-generation/generation",
            embeddings="services/embeddings/text-embedding/text-embedding",
            image_generation="services/aigc/image-generation/generation",
            text2image_synthesis="services/aigc/text2image/image-synthesis",
            image2image_synthesis="services/aigc/image2image/image-synthesis",
            video_generation="services/aigc/video-generation/video-synthesis",
        ),
    )


def _adapter(
    handler: httpx.AsyncBaseTransport,
    *,
    safe_retries: int = 1,
) -> DashScopePublicDriver:
    profile = DashScopeMediaProfile(
        name="main",
        connection=_connection(safe_retries=safe_retries),
        default_image_model="wan2.7-image",
        default_video_model="wan2.7-t2v",
    )
    return DashScopePublicDriver(
        client=DashScopeClient(transport=handler),
        profiles=(profile,),
    )


def _source(role: str, url: str) -> MediaInput:
    return MediaInput(role=MediaInputRole(role), source=MediaSource(url))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_request", "expected_path", "is_async"),
    [
        (
            MediaRequest(
                capability=MediaCapability.IMAGE_GENERATION,
                model="qwen-image-2.0",
                mode="text_to_image",
                prompt="生成",
            ),
            "/api/v1/services/aigc/multimodal-generation/generation",
            False,
        ),
        (
            MediaRequest(
                capability=MediaCapability.IMAGE_GENERATION,
                model="wan2.7-image",
                mode="text_to_image",
                prompt="生成",
            ),
            "/api/v1/services/aigc/image-generation/generation",
            True,
        ),
        (
            MediaRequest(
                capability=MediaCapability.IMAGE_GENERATION,
                model="wan2.5-t2i-preview",
                mode="text_to_image",
                prompt="生成",
            ),
            "/api/v1/services/aigc/text2image/image-synthesis",
            True,
        ),
        (
            MediaRequest(
                capability=MediaCapability.IMAGE_GENERATION,
                model="wan2.5-i2i-preview",
                mode="image_edit",
                prompt="编辑",
                inputs=(_source("source_image", "https://example.com/source.png"),),
            ),
            "/api/v1/services/aigc/image2image/image-synthesis",
            True,
        ),
        (
            MediaRequest(
                capability=MediaCapability.VIDEO_GENERATION,
                model="wan2.7-t2v",
                mode="text_to_video",
                prompt="生成",
            ),
            "/api/v1/services/aigc/video-generation/video-synthesis",
            True,
        ),
    ],
)
async def test_five_protocols_use_exact_resource(
    media_request: MediaRequest,
    expected_path: str,
    is_async: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handler(upstream: httpx.Request) -> httpx.Response:
        requests.append(upstream)
        if is_async:
            return httpx.Response(200, json={"request_id": "req", "output": {"task_id": "task"}})
        return httpx.Response(
            200,
            json={
                "request_id": "req",
                "output": {"choices": [{"message": {"content": [{"image": "https://out/image.png"}]}}]},
            },
        )

    adapter = _adapter(httpx.MockTransport(handler))
    operation = adapter.prepare("main", media_request)
    outcome = await adapter.submit(operation)

    assert requests[0].url.path == expected_path
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert (requests[0].headers.get("x-dashscope-async") == "enable") is is_async
    assert isinstance(outcome, Accepted if is_async else Completed)
    await adapter.client.aclose()


@pytest.mark.asyncio
async def test_explicit_family_is_locked_without_fallback() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"code": "InvalidParameter", "message": "failed"})

    adapter = _adapter(httpx.MockTransport(handler))
    operation = adapter.prepare(
        "main",
        MediaRequest(
            capability=MediaCapability.IMAGE_GENERATION,
            model="wan2.7-image",
            mode="text_to_image",
            prompt="生成",
            protocol_family="dashscope_multimodal_generation",
        ),
    )
    outcome = await adapter.submit(operation)

    assert isinstance(outcome, Failed)
    assert calls == 1
    await adapter.client.aclose()


@pytest.mark.asyncio
async def test_wan26_sse_preserves_text_media_order_and_usage() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = "\n\n".join(
            (
                'data: {"request_id":"req","output":{"choices":[{"message":{"content":[{"text":"步"}]}}]},"usage":{"image_count":0}}',
                'data: {"request_id":"req","output":{"choices":[{"message":{"content":[{"text":"骤"}]}}]},"usage":{"image_count":0}}',
                'data: {"request_id":"req","output":{"choices":[{"message":{"content":[{"image":"https://out/a.png"}]}}]},"usage":{"image_count":1}}',
                'data: {"request_id":"req","output":{"choices":[{"message":{"content":[{"text":"完成"}]}}]},"usage":{"image_count":1}}',
                "data: [DONE]",
            )
        )
        return httpx.Response(200, text=f"{body}\n\n", headers={"content-type": "text/event-stream"})

    adapter = _adapter(httpx.MockTransport(handler))
    operation = adapter.prepare(
        "main",
        MediaRequest(
            capability=MediaCapability.IMAGE_GENERATION,
            model="wan2.6-image",
            mode="text_to_image",
            prompt="教程",
            protocol_family="dashscope_multimodal_generation",
            parameters={"enable_interleave": True},
        ),
    )
    outcome = await adapter.submit(operation)

    assert isinstance(outcome, Completed)
    assert [(item.kind, item.text, item.url) for item in outcome.outputs] == [
        ("text", "步骤", None),
        ("media", None, "https://out/a.png"),
        ("text", "完成", None),
    ]
    assert outcome.usage == {"image_count": 1}
    assert requests[0].headers["x-dashscope-sse"] == "enable"
    assert json.loads(requests[0].content)["parameters"]["stream"] is True
    await adapter.client.aclose()


def test_qwen_output_constraints_and_parameter_precedence() -> None:
    profile = DashScopeMediaProfile(
        name="main",
        connection=_connection(),
        image_default_parameters={"n": 2, "watermark": False},
        image_override_parameters={"watermark": True},
    )
    adapter = DashScopePublicDriver(client=DashScopeClient(), profiles=(profile,))
    operation = adapter.prepare(
        "main",
        MediaRequest(
            capability=MediaCapability.IMAGE_GENERATION,
            model="qwen-image-2.0",
            mode="text_to_image",
            prompt="生成",
            parameters={"n": 6, "watermark": False},
        ),
    )
    assert operation.payload["body"]["parameters"] == {"n": 6, "watermark": True}  # type: ignore[index]

    with pytest.raises(ValueError, match="只允许"):
        adapter.prepare(
            "main",
            MediaRequest(
                capability=MediaCapability.IMAGE_GENERATION,
                model="qwen-image-edit",
                mode="image_edit",
                prompt="编辑",
                inputs=(_source("source_image", "https://example.com/source.png"),),
                parameters={"n": 2},
            ),
        )


@pytest.mark.asyncio
async def test_task_query_retries_and_cancel_normalizes_outcome() -> None:
    calls = 0
    task_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls, task_calls
        calls += 1
        if "/tasks/" not in request.url.path:
            return httpx.Response(200, json={"output": {"task_id": "task", "task_status": "PENDING"}})
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"request_id": "cancel", "output": {"task_status": "CANCELED"}})
        task_calls += 1
        if task_calls == 1:
            return httpx.Response(503, json={"code": "Unavailable", "message": "retry"})
        return httpx.Response(
            200,
            json={"request_id": "poll", "output": {"task_id": "task", "task_status": "RUNNING"}},
        )

    adapter = _adapter(httpx.MockTransport(handler))
    operation = adapter.prepare(
        "main",
        MediaRequest(capability=MediaCapability.VIDEO_GENERATION, mode="text_to_video", prompt="生成"),
    )
    submitted = await adapter.submit(operation)
    assert isinstance(submitted, Running | Accepted)
    handle = submitted.remote_handle

    running = await adapter.poll(handle)
    canceled = await adapter.cancel(handle)
    assert isinstance(running, Running)
    assert isinstance(canceled, Canceled)
    assert calls == 4
    await adapter.client.aclose()


@pytest.mark.asyncio
async def test_submission_timeout_is_uncertain_and_never_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    adapter = _adapter(httpx.MockTransport(handler), safe_retries=5)
    operation = adapter.prepare(
        "main",
        MediaRequest(capability=MediaCapability.IMAGE_GENERATION, mode="text_to_image", prompt="生成"),
    )
    outcome = await adapter.submit(operation)

    assert isinstance(outcome, Failed)
    assert outcome.error.code == "EXECUTION_UNCERTAIN"
    assert outcome.error.uncertain is True
    assert calls == 1
    await adapter.client.aclose()


@pytest.mark.asyncio
async def test_oss_upload_and_streaming_artifact_download(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/uploads"):
            return httpx.Response(
                200,
                json={
                    "output": {
                        "upload_dir": "dashscope-instant/test",
                        "oss_access_key_id": "access",
                        "signature": "signature",
                        "policy": "policy",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                        "upload_host": "https://oss.example.com",
                    }
                },
            )
        if request.url.host == "oss.example.com":
            return httpx.Response(200)
        return httpx.Response(200, content=b"12345", headers={"content-type": "image/png"})

    adapter = _adapter(httpx.MockTransport(handler))
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    oss_url = await adapter.upload_file("main", model="wan2.7-image", path=source, media_type="image/png")
    assert oss_url == "oss://dashscope-instant/test/source.png"
    assert b"OSSAccessKeyId" in requests[1].content

    destination = tmp_path / "artifacts" / "result.png"
    artifact = await adapter.materialize(
        "main",
        url="https://artifacts.example/result.png",
        destination=destination,
        max_bytes=5,
    )
    assert destination.read_bytes() == b"12345"
    assert artifact.size == 5
    assert len(artifact.sha256) == 64
    await adapter.client.aclose()
