"""Volcengine ARK 媒体模型目录。

模型 ID、参数取值与模式支持全部逐字段核对自本仓库的
`docs/provider_docs/volcengine_ark/API参考/5.视频生成 API/` 与 `6.图片生成 API/`。
文档没写的东西这里就不写——宁可让 ARK 自己强校验并报错，也不猜一个约束出来。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ....clients.ark import ArkConnection
from ...domain import (
    MediaCapability,
    MediaInputRole,
    MediaRequest,
    ModelCapability,
    PublicJsonObject,
)
from ..common import ModeConstraint, ParameterConstraint

type ArkMediaProtocolFamily = Literal[
    "ark_images_generations",
    "ark_content_generation_tasks",
]


@dataclass(frozen=True, slots=True)
class ArkProtocolRoute:
    capability: MediaCapability
    model: str
    protocol_family: ArkMediaProtocolFamily
    mode: str = ""


@dataclass(frozen=True, slots=True)
class ArkMediaProfile:
    name: str
    connection: ArkConnection
    default_image_model: str | None = None
    default_video_model: str | None = None
    image_default_parameters: PublicJsonObject = field(default_factory=dict)
    image_override_parameters: PublicJsonObject = field(default_factory=dict)
    video_default_parameters: PublicJsonObject = field(default_factory=dict)
    video_override_parameters: PublicJsonObject = field(default_factory=dict)
    protocol_routes: tuple[ArkProtocolRoute, ...] = ()


@dataclass(frozen=True, slots=True)
class ArkMediaModelDefinition:
    model: str
    capability: MediaCapability
    modes: Mapping[str, ModeConstraint]
    families: frozenset[ArkMediaProtocolFamily]
    default_family: ArkMediaProtocolFamily
    parameters: Mapping[str, ParameterConstraint]
    max_outputs: int
    # 参考图上限：Seedream 5.0 pro 10 张、其余 14 张、Seedance 2.0 系列 9 张。
    max_reference_images: int = 0


@dataclass(frozen=True, slots=True)
class ArkResolvedMediaRequest:
    profile: ArkMediaProfile
    request: MediaRequest
    model: str
    family: ArkMediaProtocolFamily
    parameters: PublicJsonObject
    definition: ArkMediaModelDefinition | None


_BOOL = ParameterConstraint((bool,))
# size 同时接受档位串（2K）和宽高像素串（2048x2048），两种形态不可混用但都是字符串；
# 具体档位逐模型不同，且像素值是连续区间，没法用 choices 表达，交给 ARK 强校验。
_SIZE = ParameterConstraint((str,))
_RESPONSE_FORMAT = ParameterConstraint((str,), choices=frozenset({"url"}))
_SEQUENTIAL = ParameterConstraint((str,), choices=frozenset({"auto", "disabled"}))
_MAX_IMAGES = ParameterConstraint((int,), minimum=1, maximum=15)
_OUTPUT_FORMAT = ParameterConstraint((str,), choices=frozenset({"png", "jpeg"}))

# 图片：Seedream 5.0 pro 只出单图，没有组图相关参数；output_format 仅 5.0 系列支持。
_IMAGE_BASE: dict[str, ParameterConstraint] = {
    "size": _SIZE,
    "response_format": _RESPONSE_FORMAT,
    "watermark": _BOOL,
}
_SEEDREAM_5_PRO = {**_IMAGE_BASE, "output_format": _OUTPUT_FORMAT}
_SEEDREAM_5_LITE = {
    **_IMAGE_BASE,
    "output_format": _OUTPUT_FORMAT,
    "sequential_image_generation": _SEQUENTIAL,
    "max_images": _MAX_IMAGES,
}
_SEEDREAM_4 = {
    **_IMAGE_BASE,
    "sequential_image_generation": _SEQUENTIAL,
    "max_images": _MAX_IMAGES,
}

_RESOLUTION_2_0 = ParameterConstraint((str,), choices=frozenset({"480p", "720p", "1080p", "4k"}))
_RESOLUTION_NO_1080 = ParameterConstraint((str,), choices=frozenset({"480p", "720p"}))
_RESOLUTION_STANDARD = ParameterConstraint((str,), choices=frozenset({"480p", "720p", "1080p"}))
_RATIO = ParameterConstraint(
    (str,),
    choices=frozenset({"16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"}),
)
# duration 允许 -1（由模型自选时长），所以下界不是 2 而是 -1。
_DURATION_2_0 = ParameterConstraint((int,), choices=frozenset({-1, *range(4, 16)}))
_DURATION_1_5 = ParameterConstraint((int,), choices=frozenset({-1, *range(4, 13)}))
_DURATION_1_0 = ParameterConstraint((int,), minimum=2, maximum=12)
# frames 只接受 [29, 289] 里满足 25+4n 的值；Seedance 2.0 系列与 1.5 Pro 不支持。
_FRAMES = ParameterConstraint((int,), choices=frozenset(value for value in range(29, 290) if (value - 25) % 4 == 0))
_SEED = ParameterConstraint((int,), minimum=-1, maximum=4_294_967_295)
_SERVICE_TIER = ParameterConstraint((str,), choices=frozenset({"default", "priority"}))
_EXECUTION_EXPIRES_AFTER = ParameterConstraint((int,), minimum=1)

_VIDEO_BASE: dict[str, ParameterConstraint] = {
    "ratio": _RATIO,
    "camera_fixed": _BOOL,
    "watermark": _BOOL,
    "generate_audio": _BOOL,
    "return_last_frame": _BOOL,
    "service_tier": _SERVICE_TIER,
    "execution_expires_after": _EXECUTION_EXPIRES_AFTER,
}
# Seedance 2.0 系列不支持 seed 与 frames，且额外支持 draft（样片）。
_SEEDANCE_2_0 = {**_VIDEO_BASE, "resolution": _RESOLUTION_2_0, "duration": _DURATION_2_0, "draft": _BOOL}
_SEEDANCE_2_0_SMALL = {
    **_VIDEO_BASE,
    "resolution": _RESOLUTION_NO_1080,
    "duration": _DURATION_2_0,
    "draft": _BOOL,
}
_SEEDANCE_1_5 = {**_VIDEO_BASE, "resolution": _RESOLUTION_STANDARD, "duration": _DURATION_1_5, "seed": _SEED}
_SEEDANCE_1_0 = {
    **_VIDEO_BASE,
    "resolution": _RESOLUTION_STANDARD,
    "duration": _DURATION_1_0,
    "frames": _FRAMES,
    "seed": _SEED,
}

_TEXT_TO_IMAGE = ModeConstraint(prompt_required=True)
_IMAGE_EDIT = ModeConstraint(
    required_roles=frozenset({MediaInputRole.SOURCE_IMAGE}),
    allowed_roles=frozenset({MediaInputRole.SOURCE_IMAGE, MediaInputRole.REFERENCE_IMAGE}),
    prompt_required=True,
)
_TEXT_TO_VIDEO = ModeConstraint(prompt_required=True)
_FIRST_FRAME = ModeConstraint(
    required_roles=frozenset({MediaInputRole.FIRST_FRAME}),
    allowed_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.DRIVING_AUDIO}),
)
_FIRST_LAST_FRAME = ModeConstraint(
    required_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.LAST_FRAME}),
    allowed_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.LAST_FRAME, MediaInputRole.DRIVING_AUDIO}),
)
# 多模态参考生视频：参考图 / 参考视频 / 驱动音频可组合，与首帧、首尾帧互斥。
_REFERENCE = ModeConstraint(
    allowed_roles=frozenset(
        {
            MediaInputRole.REFERENCE_IMAGE,
            MediaInputRole.REFERENCE_VIDEO,
            MediaInputRole.VIDEO,
            MediaInputRole.DRIVING_AUDIO,
        }
    ),
    prompt_required=True,
)
_MODE_CONSTRAINTS: dict[str, ModeConstraint] = {
    "text_to_image": _TEXT_TO_IMAGE,
    "image_edit": _IMAGE_EDIT,
    "text_to_video": _TEXT_TO_VIDEO,
    "first_frame_to_video": _FIRST_FRAME,
    "first_last_frame_to_video": _FIRST_LAST_FRAME,
    "reference_to_video": _REFERENCE,
}

_FAMILY_CAPABILITY: dict[ArkMediaProtocolFamily, MediaCapability] = {
    "ark_images_generations": MediaCapability.IMAGE_GENERATION,
    "ark_content_generation_tasks": MediaCapability.VIDEO_GENERATION,
}
_FAMILY_MODES: dict[ArkMediaProtocolFamily, frozenset[str]] = {
    "ark_images_generations": frozenset({"text_to_image", "image_edit"}),
    "ark_content_generation_tasks": frozenset(
        {"text_to_video", "first_frame_to_video", "first_last_frame_to_video", "reference_to_video"}
    ),
}


def _image_definition(
    model: str,
    *,
    parameters: Mapping[str, ParameterConstraint],
    max_outputs: int,
    max_reference_images: int,
) -> ArkMediaModelDefinition:
    return ArkMediaModelDefinition(
        model=model,
        capability=MediaCapability.IMAGE_GENERATION,
        modes={mode: _MODE_CONSTRAINTS[mode] for mode in ("text_to_image", "image_edit")},
        families=frozenset({"ark_images_generations"}),
        default_family="ark_images_generations",
        parameters=parameters,
        max_outputs=max_outputs,
        max_reference_images=max_reference_images,
    )


def _video_definition(
    model: str,
    *,
    modes: tuple[str, ...],
    parameters: Mapping[str, ParameterConstraint],
    max_reference_images: int,
) -> ArkMediaModelDefinition:
    return ArkMediaModelDefinition(
        model=model,
        capability=MediaCapability.VIDEO_GENERATION,
        modes={mode: _MODE_CONSTRAINTS[mode] for mode in modes},
        families=frozenset({"ark_content_generation_tasks"}),
        default_family="ark_content_generation_tasks",
        parameters=parameters,
        max_outputs=1,
        max_reference_images=max_reference_images,
    )


def _build_registry() -> dict[str, ArkMediaModelDefinition]:
    registry: dict[str, ArkMediaModelDefinition] = {}

    # Seedream 5.0 pro：只生成单图，不支持组图参数，最多 10 张参考图。
    for model in ("doubao-seedream-5-0-pro", "doubao-seedream-5-0-pro-260628"):
        registry[model] = _image_definition(
            model,
            parameters=_SEEDREAM_5_PRO,
            max_outputs=1,
            max_reference_images=10,
        )
    # Seedream 5.0 lite：支持组图，最多 14 张参考图，且"参考图数 + 生成图数 ≤ 15"。
    for model in ("doubao-seedream-5-0-lite", "doubao-seedream-5-0-lite-260128"):
        registry[model] = _image_definition(
            model,
            parameters=_SEEDREAM_5_LITE,
            max_outputs=15,
            max_reference_images=14,
        )
    # Seedream 4.5 / 4.0：组图能力同上，但不支持 output_format。
    for model in (
        "doubao-seedream-4-5",
        "doubao-seedream-4-5-251128",
        "doubao-seedream-4-0",
        "doubao-seedream-4-0-250828",
    ):
        registry[model] = _image_definition(
            model,
            parameters=_SEEDREAM_4,
            max_outputs=15,
            max_reference_images=14,
        )

    seedance_2_modes = ("text_to_video", "first_frame_to_video", "first_last_frame_to_video", "reference_to_video")
    for model in ("doubao-seedance-2-0", "doubao-seedance-2-0-260128"):
        registry[model] = _video_definition(
            model,
            modes=seedance_2_modes,
            parameters=_SEEDANCE_2_0,
            max_reference_images=9,
        )
    # Seedance 2.0 Fast / Mini 不支持 1080p 与 4k。
    for model in (
        "doubao-seedance-2-0-fast",
        "doubao-seedance-2-0-fast-260128",
        "doubao-seedance-2-0-mini",
        "doubao-seedance-2-0-mini-260615",
    ):
        registry[model] = _video_definition(
            model,
            modes=seedance_2_modes,
            parameters=_SEEDANCE_2_0_SMALL,
            max_reference_images=9,
        )
    # Seedance 1.5 Pro：支持首尾帧，不支持参考图生视频。
    for model in ("doubao-seedance-1-5-pro", "doubao-seedance-1-5-pro-251215"):
        registry[model] = _video_definition(
            model,
            modes=("text_to_video", "first_frame_to_video", "first_last_frame_to_video"),
            parameters=_SEEDANCE_1_5,
            max_reference_images=0,
        )
    for model in (
        "doubao-seedance-1-0-pro",
        "doubao-seedance-1-0-pro-250528",
        "doubao-seedance-1-0-pro-fast",
        "doubao-seedance-1-0-pro-fast-251015",
    ):
        registry[model] = _video_definition(
            model,
            modes=("text_to_video", "first_frame_to_video", "first_last_frame_to_video"),
            parameters=_SEEDANCE_1_0,
            max_reference_images=0,
        )
    return registry


ARK_MEDIA_MODEL_REGISTRY: dict[str, ArkMediaModelDefinition] = _build_registry()


def resolve_media_request(request: MediaRequest, profile: ArkMediaProfile) -> ArkResolvedMediaRequest:
    model = request.model or (
        profile.default_image_model
        if request.capability is MediaCapability.IMAGE_GENERATION
        else profile.default_video_model
    )
    if not model:
        raise ValueError("请求和 profile 均未提供媒体模型")
    definition = ARK_MEDIA_MODEL_REGISTRY.get(model)
    family = _resolve_family(request, profile, model, definition)
    _validate_family(request, family, definition)
    _validate_inputs(request, definition)
    parameters = _resolve_parameters(request, profile, definition)
    return ArkResolvedMediaRequest(
        profile=profile,
        request=request,
        model=model,
        family=family,
        parameters=parameters,
        definition=definition,
    )


def _resolve_family(
    request: MediaRequest,
    profile: ArkMediaProfile,
    model: str,
    definition: ArkMediaModelDefinition | None,
) -> ArkMediaProtocolFamily:
    if request.protocol_family is not None:
        if request.protocol_family not in _FAMILY_CAPABILITY:
            raise ValueError(f"未知 Volcengine ARK 媒体协议簇 {request.protocol_family}")
        return request.protocol_family
    for route in profile.protocol_routes:
        if route.capability == request.capability and route.model == model and route.mode == request.mode:
            return route.protocol_family
    for route in profile.protocol_routes:
        if route.capability == request.capability and route.model == model and not route.mode:
            return route.protocol_family
    # ep- 开头的接入点 ID 不在目录里，必须显式声明协议簇或配置 profile 路由。
    if definition is None:
        raise ValueError(f"未知媒体模型 {model} 必须显式提供 protocol_family 或配置 profile 路由")
    return definition.default_family


def _validate_family(
    request: MediaRequest,
    family: ArkMediaProtocolFamily,
    definition: ArkMediaModelDefinition | None,
) -> None:
    if _FAMILY_CAPABILITY[family] != request.capability:
        raise ValueError(f"协议簇 {family} 不支持能力 {request.capability}")
    if request.mode not in _FAMILY_MODES[family]:
        raise ValueError(f"协议簇 {family} 不支持 mode {request.mode}")
    if definition is None:
        return
    if definition.capability != request.capability:
        raise ValueError(f"模型 {definition.model} 不支持能力 {request.capability}")
    if request.mode not in definition.modes:
        raise ValueError(f"模型 {definition.model} 不支持 mode {request.mode}")
    if family not in definition.families:
        raise ValueError(f"模型 {definition.model} 不支持协议簇 {family}")


def _validate_inputs(request: MediaRequest, definition: ArkMediaModelDefinition | None) -> None:
    constraint = _MODE_CONSTRAINTS.get(request.mode) if definition is None else definition.modes[request.mode]
    if constraint is None:
        raise ValueError(f"未知媒体 mode {request.mode}")
    if constraint.prompt_required and not request.prompt.strip():
        raise ValueError(f"mode {request.mode} 必须提供 prompt")
    roles = [item.role for item in request.inputs]
    missing = constraint.required_roles.difference(roles)
    if missing:
        raise ValueError(f"mode {request.mode} 缺少输入角色: {', '.join(sorted(missing))}")
    unsupported = set(roles).difference(constraint.allowed_roles)
    if unsupported:
        raise ValueError(f"mode {request.mode} 不支持输入角色: {', '.join(sorted(unsupported))}")
    for role in set(roles):
        if roles.count(role) > 1 and role not in {MediaInputRole.REFERENCE_IMAGE, MediaInputRole.REFERENCE_VIDEO}:
            raise ValueError(f"输入角色 {role} 不允许重复")
    if definition is None:
        return
    reference_images = roles.count(MediaInputRole.REFERENCE_IMAGE) + roles.count(MediaInputRole.SOURCE_IMAGE)
    if reference_images > definition.max_reference_images:
        raise ValueError(f"模型 {definition.model} 最多允许 {definition.max_reference_images} 张参考图")


def _resolve_parameters(
    request: MediaRequest,
    profile: ArkMediaProfile,
    definition: ArkMediaModelDefinition | None,
) -> PublicJsonObject:
    if request.capability is MediaCapability.IMAGE_GENERATION:
        defaults = profile.image_default_parameters
        overrides = profile.image_override_parameters
        # response_format 锁定 url：b64_json 会把整张图塞进 JSON 响应，
        # 与引擎"产物先拿 URL、再由 materialize 落盘"的流程冲突。
        merged: PublicJsonObject = {"response_format": "url"}
    else:
        defaults = profile.video_default_parameters
        overrides = profile.video_override_parameters
        merged = {}
    merged.update(defaults)
    merged.update(request.parameters)
    merged.update(overrides)
    if request.capability is MediaCapability.IMAGE_GENERATION:
        merged["response_format"] = "url"
    if definition is not None:
        unknown = set(merged).difference(definition.parameters)
        if unknown:
            raise ValueError(f"模型 {definition.model} 不支持参数: {', '.join(sorted(unknown))}")
        for name, value in merged.items():
            definition.parameters[name].validate(name, value)
        _validate_image_output_count(request, merged, definition)
    return merged


def _validate_image_output_count(
    request: MediaRequest,
    parameters: PublicJsonObject,
    definition: ArkMediaModelDefinition,
) -> None:
    """校验组图数量：受 max_images、模型上限与"参考图 + 生成图 ≤ 15"三重约束。"""

    if definition.capability is not MediaCapability.IMAGE_GENERATION:
        return
    max_images = parameters.get("max_images")
    if not isinstance(max_images, int) or isinstance(max_images, bool):
        return
    if max_images > definition.max_outputs:
        raise ValueError(f"模型 {definition.model} 最多生成 {definition.max_outputs} 个输出")
    reference_images = sum(
        1 for item in request.inputs if item.role in {MediaInputRole.REFERENCE_IMAGE, MediaInputRole.SOURCE_IMAGE}
    )
    if reference_images + max_images > 15:
        raise ValueError(f"参考图数量({reference_images}) 与 max_images({max_images}) 之和不能超过 15")


def media_capabilities() -> tuple[ModelCapability, ...]:
    return tuple(
        ModelCapability(
            model=definition.model,
            capability=definition.capability,
            modes=tuple(definition.modes),
            protocol_families=tuple(sorted(definition.families)),
            max_outputs=definition.max_outputs,
        )
        for definition in ARK_MEDIA_MODEL_REGISTRY.values()
    )
