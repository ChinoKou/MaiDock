from collections.abc import Mapping

import pytest

from src.clients.ark import ArkConnection
from src.clients.common import NO_RETRY, HttpConnection
from src.public_api.domain import (
    MediaCapability,
    MediaInput,
    MediaInputRole,
    MediaRequest,
    MediaSource,
    PublicJsonValue,
)
from src.public_api.providers.volcengine_ark.registry import (
    ARK_MEDIA_MODEL_REGISTRY,
    ArkMediaProfile,
    ArkProtocolRoute,
    media_capabilities,
    resolve_media_request,
)

IMAGE_URL = "https://cdn.example/in.png"


def _profile(
    *,
    default_image_model: str | None = None,
    default_video_model: str | None = None,
    image_override_parameters: Mapping[str, PublicJsonValue] | None = None,
    protocol_routes: tuple[ArkProtocolRoute, ...] = (),
) -> ArkMediaProfile:
    connection = ArkConnection(
        http=HttpConnection(base_url="https://ark.example/api/v3"),
        retry=NO_RETRY,
        responses_path="responses",
        embeddings_path="embeddings/multimodal",
        audio_transcriptions_path="responses",
        tokenization_path="tokenization",
    )
    return ArkMediaProfile(
        name="default",
        connection=connection,
        default_image_model=default_image_model,
        default_video_model=default_video_model,
        image_override_parameters=dict(image_override_parameters or {}),
        protocol_routes=protocol_routes,
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


def _reference(role: MediaInputRole, url: str = IMAGE_URL) -> MediaInput:
    return MediaInput(role=role, source=MediaSource(url))


def test_registry_covers_documented_model_ids() -> None:
    """模型 ID 逐字段核对自本地 provider_docs，改动目录时这条会先失败。"""

    for model in (
        "doubao-seedream-5-0-pro",
        "doubao-seedream-5-0-pro-260628",
        "doubao-seedream-5-0-lite",
        "doubao-seedream-5-0-lite-260128",
        "doubao-seedream-4-5",
        "doubao-seedream-4-5-251128",
        "doubao-seedream-4-0",
        "doubao-seedream-4-0-250828",
        "doubao-seedance-2-0",
        "doubao-seedance-2-0-260128",
        "doubao-seedance-2-0-fast",
        "doubao-seedance-2-0-mini",
        "doubao-seedance-1-5-pro",
        "doubao-seedance-1-5-pro-251215",
        "doubao-seedance-1-0-pro",
        "doubao-seedance-1-0-pro-250528",
        "doubao-seedance-1-0-pro-fast",
    ):
        assert model in ARK_MEDIA_MODEL_REGISTRY, model


def test_image_request_locks_response_format_to_url() -> None:
    """b64_json 会把整张图塞进 JSON 响应，与"先拿 URL 再 materialize"的流程冲突。"""

    resolved = resolve_media_request(_image_request(), _profile())

    assert resolved.parameters["response_format"] == "url"
    assert resolved.family == "ark_images_generations"


def test_image_request_rejects_profile_override_of_response_format() -> None:
    resolved = resolve_media_request(
        _image_request(parameters={"response_format": "b64_json"}),
        _profile(image_override_parameters={"response_format": "b64_json"}),
    )

    assert resolved.parameters["response_format"] == "url"


def test_seedream_5_pro_rejects_group_image_parameters() -> None:
    """5.0 pro 只出单图，文档未列组图参数，因此必须报未知参数而不是静默透传。"""

    with pytest.raises(ValueError, match="不支持参数"):
        resolve_media_request(
            _image_request(model="doubao-seedream-5-0-pro", parameters={"sequential_image_generation": "auto"}),
            _profile(),
        )


def test_seedream_4_accepts_group_image_parameters() -> None:
    resolved = resolve_media_request(
        _image_request(parameters={"sequential_image_generation": "auto", "max_images": 4}),
        _profile(),
    )

    assert resolved.parameters["sequential_image_generation"] == "auto"
    assert resolved.parameters["max_images"] == 4


def test_seedream_4_rejects_out_of_range_max_images() -> None:
    with pytest.raises(ValueError, match="max_images"):
        resolve_media_request(_image_request(parameters={"max_images": 16}), _profile())


def test_reference_images_plus_max_images_must_not_exceed_fifteen() -> None:
    """文档明写"输入的参考图数量 + 最终生成的图片数量 ≤ 15 张"。"""

    inputs = (
        _reference(MediaInputRole.SOURCE_IMAGE),
        *(_reference(MediaInputRole.REFERENCE_IMAGE) for _ in range(4)),
    )

    with pytest.raises(ValueError, match="之和不能超过 15"):
        resolve_media_request(
            _image_request(mode="image_edit", inputs=inputs, parameters={"max_images": 11}),
            _profile(),
        )


def test_seedream_5_pro_limits_reference_images_to_ten() -> None:
    inputs = (
        _reference(MediaInputRole.SOURCE_IMAGE),
        *(_reference(MediaInputRole.REFERENCE_IMAGE) for _ in range(10)),
    )

    with pytest.raises(ValueError, match="最多允许 10 张参考图"):
        resolve_media_request(
            _image_request(model="doubao-seedream-5-0-pro", mode="image_edit", inputs=inputs),
            _profile(),
        )


def test_output_format_is_rejected_on_seedream_4() -> None:
    """output_format 文档只标注 5.0 系列支持。"""

    with pytest.raises(ValueError, match="不支持参数"):
        resolve_media_request(_image_request(parameters={"output_format": "png"}), _profile())

    resolved = resolve_media_request(
        _image_request(model="doubao-seedream-5-0-pro", parameters={"output_format": "png"}),
        _profile(),
    )
    assert resolved.parameters["output_format"] == "png"


def test_video_resolution_4k_only_on_full_seedance_2() -> None:
    resolved = resolve_media_request(_video_request(parameters={"resolution": "4k"}), _profile())
    assert resolved.parameters["resolution"] == "4k"

    with pytest.raises(ValueError, match="resolution"):
        resolve_media_request(
            _video_request(model="doubao-seedance-2-0-fast", parameters={"resolution": "4k"}),
            _profile(),
        )


def test_video_1080p_rejected_on_fast_and_mini() -> None:
    for model in ("doubao-seedance-2-0-fast", "doubao-seedance-2-0-mini"):
        with pytest.raises(ValueError, match="resolution"):
            resolve_media_request(_video_request(model=model, parameters={"resolution": "1080p"}), _profile())


def test_video_duration_accepts_model_auto_sentinel() -> None:
    """-1 表示由模型自选时长，是文档明列的合法值，不能按下界剔掉。"""

    resolved = resolve_media_request(_video_request(parameters={"duration": -1}), _profile())
    assert resolved.parameters["duration"] == -1


def test_seedance_1_0_rejects_negative_duration() -> None:
    """1.0 系列没有 -1 智能时长，区间是 [2, 12]。"""

    with pytest.raises(ValueError, match="duration"):
        resolve_media_request(
            _video_request(model="doubao-seedance-1-0-pro", parameters={"duration": -1}),
            _profile(),
        )


def test_seed_and_frames_rejected_on_seedance_2() -> None:
    """文档写明 Seedance 2.0 系列暂不支持 seed 与 frames。"""

    unsupported: tuple[Mapping[str, PublicJsonValue], ...] = ({"seed": 11}, {"frames": 57})
    for parameters in unsupported:
        with pytest.raises(ValueError, match="不支持参数"):
            resolve_media_request(_video_request(parameters=parameters), _profile())


def test_frames_must_match_documented_step() -> None:
    """frames 只接受 [29, 289] 内满足 25+4n 的值。"""

    resolved = resolve_media_request(
        _video_request(model="doubao-seedance-1-0-pro", parameters={"frames": 57}),
        _profile(),
    )
    assert resolved.parameters["frames"] == 57

    with pytest.raises(ValueError, match="frames"):
        resolve_media_request(
            _video_request(model="doubao-seedance-1-0-pro", parameters={"frames": 58}),
            _profile(),
        )


def test_reference_to_video_only_on_seedance_2() -> None:
    inputs = (_reference(MediaInputRole.REFERENCE_IMAGE),)
    resolved = resolve_media_request(_video_request(mode="reference_to_video", inputs=inputs), _profile())
    assert resolved.family == "ark_content_generation_tasks"

    with pytest.raises(ValueError, match="不支持 mode"):
        resolve_media_request(
            _video_request(model="doubao-seedance-1-5-pro", mode="reference_to_video", inputs=inputs),
            _profile(),
        )


def test_first_last_frame_requires_both_roles() -> None:
    with pytest.raises(ValueError, match="缺少输入角色"):
        resolve_media_request(
            _video_request(mode="first_last_frame_to_video", inputs=(_reference(MediaInputRole.FIRST_FRAME),)),
            _profile(),
        )

    resolved = resolve_media_request(
        _video_request(
            mode="first_last_frame_to_video",
            inputs=(_reference(MediaInputRole.FIRST_FRAME), _reference(MediaInputRole.LAST_FRAME)),
        ),
        _profile(),
    )
    assert resolved.model == "doubao-seedance-2-0"


def test_unknown_endpoint_model_requires_explicit_family() -> None:
    """ep- 开头的接入点 ID 不在目录里，必须显式声明协议簇或配置 profile 路由。"""

    with pytest.raises(ValueError, match="必须显式提供 protocol_family"):
        resolve_media_request(_video_request(model="ep-20260101-abcde"), _profile())


def test_unknown_endpoint_model_accepts_profile_route() -> None:
    profile = _profile(
        protocol_routes=(
            ArkProtocolRoute(
                capability=MediaCapability.VIDEO_GENERATION,
                model="ep-20260101-abcde",
                protocol_family="ark_content_generation_tasks",
            ),
        )
    )

    resolved = resolve_media_request(_video_request(model="ep-20260101-abcde"), profile)

    assert resolved.family == "ark_content_generation_tasks"
    assert resolved.definition is None


def test_family_and_capability_must_agree() -> None:
    request = MediaRequest(
        capability=MediaCapability.VIDEO_GENERATION,
        mode="text_to_video",
        prompt="x",
        model="doubao-seedance-2-0",
        protocol_family="ark_images_generations",
    )

    with pytest.raises(ValueError, match="不支持能力"):
        resolve_media_request(request, _profile())


def test_profile_default_model_is_used_when_request_omits_it() -> None:
    profile = _profile(default_video_model="doubao-seedance-1-5-pro")
    request = MediaRequest(capability=MediaCapability.VIDEO_GENERATION, mode="text_to_video", prompt="x")

    resolved = resolve_media_request(request, profile)

    assert resolved.model == "doubao-seedance-1-5-pro"


def test_media_capabilities_reports_every_registry_entry() -> None:
    capabilities = media_capabilities()

    assert len(capabilities) == len(ARK_MEDIA_MODEL_REGISTRY)
    by_model = {item.model: item for item in capabilities}
    assert by_model["doubao-seedream-5-0-pro"].max_outputs == 1
    assert by_model["doubao-seedream-4-0"].max_outputs == 15
    assert by_model["doubao-seedance-2-0"].protocol_families == ("ark_content_generation_tasks",)
