from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ....clients.dashscope import DashScopeConnection
from ...domain import (
    MediaCapability,
    MediaInputRole,
    MediaRequest,
    ModelCapability,
    PublicJsonObject,
)
from ..common import ModeConstraint, ParameterConstraint

type MediaProtocolFamily = Literal[
    "dashscope_multimodal_generation",
    "dashscope_image_generation",
    "dashscope_text2image_synthesis",
    "dashscope_image2image_synthesis",
    "dashscope_video_generation",
]


@dataclass(frozen=True, slots=True)
class DashScopeProtocolRoute:
    capability: MediaCapability
    model: str
    protocol_family: MediaProtocolFamily
    mode: str = ""


@dataclass(frozen=True, slots=True)
class DashScopeMediaProfile:
    name: str
    connection: DashScopeConnection
    default_image_model: str | None = None
    default_video_model: str | None = None
    image_default_parameters: PublicJsonObject = field(default_factory=dict)
    image_override_parameters: PublicJsonObject = field(default_factory=dict)
    video_default_parameters: PublicJsonObject = field(default_factory=dict)
    video_override_parameters: PublicJsonObject = field(default_factory=dict)
    protocol_routes: tuple[DashScopeProtocolRoute, ...] = ()


@dataclass(frozen=True, slots=True)
class MediaModelDefinition:
    model: str
    capability: MediaCapability
    modes: Mapping[str, ModeConstraint]
    families: frozenset[MediaProtocolFamily]
    default_family: MediaProtocolFamily
    parameters: Mapping[str, ParameterConstraint]
    max_outputs: int


@dataclass(frozen=True, slots=True)
class ResolvedMediaRequest:
    profile: DashScopeMediaProfile
    request: MediaRequest
    model: str
    family: MediaProtocolFamily
    parameters: PublicJsonObject
    definition: MediaModelDefinition | None


_BOOL = ParameterConstraint((bool,))
_SEED = ParameterConstraint((int,), minimum=0, maximum=2_147_483_647)
_IMAGE_COMMON: dict[str, ParameterConstraint] = {
    "size": ParameterConstraint((str,)),
    "n": ParameterConstraint((int,), minimum=1, maximum=4),
    "seed": _SEED,
    "prompt_extend": _BOOL,
    "watermark": _BOOL,
}
_QWEN_SINGLE = {**_IMAGE_COMMON, "n": ParameterConstraint((int,), choices=frozenset({1}))}
_QWEN_MULTI = {**_IMAGE_COMMON, "n": ParameterConstraint((int,), minimum=1, maximum=6)}
_WAN27_IMAGE = {
    **_IMAGE_COMMON,
    "n": ParameterConstraint((int,), minimum=1, maximum=12),
    "thinking_mode": _BOOL,
    "bbox_list": ParameterConstraint((list,)),
    "enable_sequential": _BOOL,
    "color_palette": ParameterConstraint((list,)),
}
_WAN26_IMAGE = {
    **_IMAGE_COMMON,
    "enable_interleave": _BOOL,
    "max_images": ParameterConstraint((int,), minimum=1, maximum=5),
}
_VIDEO_COMMON: dict[str, ParameterConstraint] = {
    "prompt_extend": _BOOL,
    "watermark": _BOOL,
    "seed": _SEED,
}
_RESOLUTION_27 = ParameterConstraint((str,), choices=frozenset({"720P", "1080P"}))
_RESOLUTION_26 = ParameterConstraint((str,), choices=frozenset({"720P", "1080P"}))
_RESOLUTION_25 = ParameterConstraint((str,), choices=frozenset({"480P", "720P", "1080P"}))
_RATIO = ParameterConstraint((str,), choices=frozenset({"16:9", "9:16", "1:1", "4:3", "3:4"}))
_SIZE_720_1080 = ParameterConstraint(
    (str,),
    choices=frozenset(
        {
            "1280*720",
            "720*1280",
            "960*960",
            "1088*832",
            "832*1088",
            "1920*1080",
            "1080*1920",
            "1440*1440",
            "1632*1248",
            "1248*1632",
        }
    ),
)
_SIZE_480_720_1080 = ParameterConstraint(
    (str,),
    choices=frozenset({"832*480", "480*832", "624*624", *_SIZE_720_1080.choices}),
)
_SHOT_TYPE = ParameterConstraint((str,), choices=frozenset({"single", "multi"}))

_TEXT_TO_IMAGE = ModeConstraint(prompt_required=True)
_IMAGE_EDIT = ModeConstraint(
    required_roles=frozenset({MediaInputRole.SOURCE_IMAGE}),
    allowed_roles=frozenset({MediaInputRole.SOURCE_IMAGE, MediaInputRole.REFERENCE_IMAGE}),
    prompt_required=True,
)
_TEXT_TO_VIDEO = ModeConstraint(
    allowed_roles=frozenset({MediaInputRole.DRIVING_AUDIO}),
    prompt_required=True,
)
_FIRST_FRAME = ModeConstraint(
    required_roles=frozenset({MediaInputRole.FIRST_FRAME}),
    allowed_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.DRIVING_AUDIO}),
)
_FIRST_LAST_FRAME = ModeConstraint(
    required_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.LAST_FRAME}),
    allowed_roles=frozenset({MediaInputRole.FIRST_FRAME, MediaInputRole.LAST_FRAME, MediaInputRole.DRIVING_AUDIO}),
)
_CONTINUATION = ModeConstraint(
    required_roles=frozenset({MediaInputRole.FIRST_CLIP}),
    allowed_roles=frozenset({MediaInputRole.FIRST_CLIP, MediaInputRole.DRIVING_AUDIO}),
)
_REFERENCE = ModeConstraint(
    allowed_roles=frozenset({MediaInputRole.REFERENCE_IMAGE, MediaInputRole.REFERENCE_VIDEO}),
    prompt_required=True,
)
_VIDEO_EDIT = ModeConstraint(
    required_roles=frozenset({MediaInputRole.VIDEO}),
    allowed_roles=frozenset({MediaInputRole.VIDEO, MediaInputRole.REFERENCE_IMAGE}),
    prompt_required=True,
)
_MODE_CONSTRAINTS: dict[str, ModeConstraint] = {
    "text_to_image": _TEXT_TO_IMAGE,
    "image_edit": _IMAGE_EDIT,
    "text_to_video": _TEXT_TO_VIDEO,
    "first_frame_to_video": _FIRST_FRAME,
    "first_last_frame_to_video": _FIRST_LAST_FRAME,
    "video_continuation": _CONTINUATION,
    "reference_to_video": _REFERENCE,
    "video_edit": _VIDEO_EDIT,
}


def _image_definition(
    model: str,
    *,
    modes: tuple[str, ...],
    families: tuple[MediaProtocolFamily, ...],
    default_family: MediaProtocolFamily,
    parameters: Mapping[str, ParameterConstraint],
    max_outputs: int,
) -> MediaModelDefinition:
    return MediaModelDefinition(
        model=model,
        capability=MediaCapability.IMAGE_GENERATION,
        modes={mode: _MODE_CONSTRAINTS[mode] for mode in modes},
        families=frozenset(families),
        default_family=default_family,
        parameters=parameters,
        max_outputs=max_outputs,
    )


def _video_definition(
    model: str,
    *,
    modes: tuple[str, ...],
    parameters: Mapping[str, ParameterConstraint],
) -> MediaModelDefinition:
    return MediaModelDefinition(
        model=model,
        capability=MediaCapability.VIDEO_GENERATION,
        modes={mode: _MODE_CONSTRAINTS[mode] for mode in modes},
        families=frozenset({"dashscope_video_generation"}),
        default_family="dashscope_video_generation",
        parameters=parameters,
        max_outputs=1,
    )


def _build_registry() -> dict[str, MediaModelDefinition]:
    registry: dict[str, MediaModelDefinition] = {}
    qwen_generation = (
        "qwen-image-max",
        "qwen-image-max-2025-12-30",
        "qwen-image-plus",
        "qwen-image-plus-2026-01-09",
        "qwen-image",
        "z-image-turbo",
    )
    qwen_generation_and_edit = (
        "qwen-image-2.0-pro",
        "qwen-image-2.0-pro-2026-06-22",
        "qwen-image-2.0-pro-2026-04-22",
        "qwen-image-2.0-pro-2026-03-03",
        "qwen-image-2.0",
        "qwen-image-2.0-2026-03-03",
    )
    qwen_edit = (
        "qwen-image-edit-max",
        "qwen-image-edit-max-2026-01-16",
        "qwen-image-edit-plus",
        "qwen-image-edit-plus-2025-12-15",
        "qwen-image-edit-plus-2025-10-30",
        "qwen-image-edit",
    )
    for model in qwen_generation:
        extra_family: tuple[MediaProtocolFamily, ...] = ()
        if model.startswith(("qwen-image-plus", "qwen-image")) and not model.startswith("qwen-image-max"):
            extra_family = ("dashscope_text2image_synthesis",)
        registry[model] = _image_definition(
            model,
            modes=("text_to_image",),
            families=("dashscope_multimodal_generation", *extra_family),
            default_family="dashscope_multimodal_generation",
            parameters=_QWEN_SINGLE,
            max_outputs=1,
        )
    for model in qwen_generation_and_edit:
        registry[model] = _image_definition(
            model,
            modes=("text_to_image", "image_edit"),
            families=("dashscope_multimodal_generation",),
            default_family="dashscope_multimodal_generation",
            parameters=_QWEN_MULTI,
            max_outputs=6,
        )
    for model in qwen_edit:
        single = model == "qwen-image-edit"
        registry[model] = _image_definition(
            model,
            modes=("image_edit",),
            families=("dashscope_multimodal_generation",),
            default_family="dashscope_multimodal_generation",
            parameters=_QWEN_SINGLE if single else _QWEN_MULTI,
            max_outputs=1 if single else 6,
        )
    for model in ("wan2.7-image-pro", "wan2.7-image", "wan2.6-image"):
        registry[model] = _image_definition(
            model,
            modes=("text_to_image", "image_edit"),
            families=("dashscope_image_generation", "dashscope_multimodal_generation"),
            default_family="dashscope_image_generation",
            parameters=_WAN27_IMAGE if model.startswith("wan2.7-") else _WAN26_IMAGE,
            max_outputs=12 if model.startswith("wan2.7-") else 5,
        )
    registry["wan2.6-t2i"] = _image_definition(
        "wan2.6-t2i",
        modes=("text_to_image",),
        families=("dashscope_image_generation",),
        default_family="dashscope_image_generation",
        parameters=_IMAGE_COMMON,
        max_outputs=4,
    )
    registry["wan2.5-t2i-preview"] = _image_definition(
        "wan2.5-t2i-preview",
        modes=("text_to_image",),
        families=("dashscope_text2image_synthesis",),
        default_family="dashscope_text2image_synthesis",
        parameters=_IMAGE_COMMON,
        max_outputs=4,
    )
    registry["wan2.5-i2i-preview"] = _image_definition(
        "wan2.5-i2i-preview",
        modes=("image_edit",),
        families=("dashscope_image2image_synthesis",),
        default_family="dashscope_image2image_synthesis",
        parameters=_IMAGE_COMMON,
        max_outputs=4,
    )

    duration_2_15 = ParameterConstraint((int,), minimum=2, maximum=15)
    duration_2_10 = ParameterConstraint((int,), minimum=2, maximum=10)
    duration_5_10_15 = ParameterConstraint((int,), choices=frozenset({5, 10, 15}))
    duration_5_10 = ParameterConstraint((int,), choices=frozenset({5, 10}))
    parameters_27_i2v = {**_VIDEO_COMMON, "resolution": _RESOLUTION_27, "duration": duration_2_15}
    for model in ("wan2.7-i2v", "wan2.7-i2v-2026-04-25"):
        registry[model] = _video_definition(
            model,
            modes=("first_frame_to_video", "first_last_frame_to_video", "video_continuation"),
            parameters=parameters_27_i2v,
        )
    parameters_27_t2v = {
        **_VIDEO_COMMON,
        "resolution": _RESOLUTION_27,
        "ratio": _RATIO,
        "duration": duration_2_15,
    }
    for model in ("wan2.7-t2v", "wan2.7-t2v-2026-06-12", "wan2.7-t2v-2026-04-25"):
        registry[model] = _video_definition(model, modes=("text_to_video",), parameters=parameters_27_t2v)
    parameters_27_r2v = {
        **_VIDEO_COMMON,
        "resolution": _RESOLUTION_27,
        "ratio": _RATIO,
        "duration": duration_2_10,
    }
    for model in ("wan2.7-r2v", "wan2.7-r2v-2026-06-12"):
        registry[model] = _video_definition(model, modes=("reference_to_video",), parameters=parameters_27_r2v)
    registry["wan2.7-videoedit"] = _video_definition(
        "wan2.7-videoedit",
        modes=("video_edit",),
        parameters={**_VIDEO_COMMON, "resolution": _RESOLUTION_27},
    )
    i2v_26 = {
        **_VIDEO_COMMON,
        "resolution": _RESOLUTION_26,
        "duration": duration_2_15,
        "shot_type": _SHOT_TYPE,
    }
    registry["wan2.6-i2v"] = _video_definition("wan2.6-i2v", modes=("first_frame_to_video",), parameters=i2v_26)
    registry["wan2.6-i2v-flash"] = _video_definition(
        "wan2.6-i2v-flash",
        modes=("first_frame_to_video",),
        parameters={**i2v_26, "audio": _BOOL},
    )
    registry["wan2.6-i2v-us"] = _video_definition(
        "wan2.6-i2v-us",
        modes=("first_frame_to_video",),
        parameters={**i2v_26, "duration": duration_5_10_15},
    )
    t2v_26 = {
        **_VIDEO_COMMON,
        "size": _SIZE_720_1080,
        "duration": duration_2_15,
        "shot_type": _SHOT_TYPE,
    }
    registry["wan2.6-t2v"] = _video_definition("wan2.6-t2v", modes=("text_to_video",), parameters=t2v_26)
    registry["wan2.6-t2v-us"] = _video_definition(
        "wan2.6-t2v-us",
        modes=("text_to_video",),
        parameters={**t2v_26, "duration": duration_5_10_15},
    )
    r2v_26 = {
        **_VIDEO_COMMON,
        "size": _SIZE_720_1080,
        "duration": duration_2_10,
        "shot_type": _SHOT_TYPE,
    }
    registry["wan2.6-r2v"] = _video_definition("wan2.6-r2v", modes=("reference_to_video",), parameters=r2v_26)
    registry["wan2.6-r2v-flash"] = _video_definition(
        "wan2.6-r2v-flash",
        modes=("reference_to_video",),
        parameters={**r2v_26, "audio": _BOOL},
    )
    registry["wan2.5-i2v-preview"] = _video_definition(
        "wan2.5-i2v-preview",
        modes=("first_frame_to_video",),
        parameters={**_VIDEO_COMMON, "resolution": _RESOLUTION_25, "duration": duration_5_10},
    )
    registry["wan2.5-t2v-preview"] = _video_definition(
        "wan2.5-t2v-preview",
        modes=("text_to_video",),
        parameters={**_VIDEO_COMMON, "size": _SIZE_480_720_1080, "duration": duration_5_10},
    )
    return registry


MEDIA_MODEL_REGISTRY = _build_registry()

_FAMILY_CAPABILITY: dict[MediaProtocolFamily, MediaCapability] = {
    "dashscope_multimodal_generation": MediaCapability.IMAGE_GENERATION,
    "dashscope_image_generation": MediaCapability.IMAGE_GENERATION,
    "dashscope_text2image_synthesis": MediaCapability.IMAGE_GENERATION,
    "dashscope_image2image_synthesis": MediaCapability.IMAGE_GENERATION,
    "dashscope_video_generation": MediaCapability.VIDEO_GENERATION,
}
_FAMILY_MODES: dict[MediaProtocolFamily, frozenset[str]] = {
    "dashscope_multimodal_generation": frozenset({"text_to_image", "image_edit"}),
    "dashscope_image_generation": frozenset({"text_to_image", "image_edit"}),
    "dashscope_text2image_synthesis": frozenset({"text_to_image"}),
    "dashscope_image2image_synthesis": frozenset({"image_edit"}),
    "dashscope_video_generation": frozenset(
        {
            "text_to_video",
            "first_frame_to_video",
            "first_last_frame_to_video",
            "video_continuation",
            "reference_to_video",
            "video_edit",
        }
    ),
}


def resolve_media_request(request: MediaRequest, profile: DashScopeMediaProfile) -> ResolvedMediaRequest:
    model = request.model or (
        profile.default_image_model
        if request.capability is MediaCapability.IMAGE_GENERATION
        else profile.default_video_model
    )
    if not model:
        raise ValueError("请求和 profile 均未提供媒体模型")
    definition = MEDIA_MODEL_REGISTRY.get(model)
    family = _resolve_family(request, profile, model, definition)
    _validate_family(request, family, definition)
    _validate_inputs(request, definition)
    parameters = _resolve_parameters(request, profile, definition)
    return ResolvedMediaRequest(
        profile=profile,
        request=request,
        model=model,
        family=family,
        parameters=parameters,
        definition=definition,
    )


def _resolve_family(
    request: MediaRequest,
    profile: DashScopeMediaProfile,
    model: str,
    definition: MediaModelDefinition | None,
) -> MediaProtocolFamily:
    if request.protocol_family is not None:
        if request.protocol_family not in _FAMILY_CAPABILITY:
            raise ValueError(f"未知 DashScope 媒体协议簇 {request.protocol_family}")
        return request.protocol_family
    for route in profile.protocol_routes:
        if route.capability == request.capability and route.model == model and route.mode == request.mode:
            return route.protocol_family
    for route in profile.protocol_routes:
        if route.capability == request.capability and route.model == model and not route.mode:
            return route.protocol_family
    if definition is None:
        raise ValueError(f"未知媒体模型 {model} 必须显式提供 protocol_family 或配置 profile 路由")
    return definition.default_family


def _validate_family(
    request: MediaRequest,
    family: MediaProtocolFamily,
    definition: MediaModelDefinition | None,
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


def _validate_inputs(request: MediaRequest, definition: MediaModelDefinition | None) -> None:
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
        if roles.count(role) > 1 and role not in {
            MediaInputRole.REFERENCE_IMAGE,
            MediaInputRole.REFERENCE_VIDEO,
        }:
            raise ValueError(f"输入角色 {role} 不允许重复")
    if request.model == "wan2.5-i2i-preview" and len(request.inputs) > 3:
        raise ValueError("wan2.5-i2i-preview 最多允许 3 张输入图片")
    for item in request.inputs:
        if item.reference_voice is not None and not (request.model or "").startswith("wan2.7-r2v"):
            raise ValueError("reference_voice 仅支持 wan2.7-r2v 模型")


def _resolve_parameters(
    request: MediaRequest,
    profile: DashScopeMediaProfile,
    definition: MediaModelDefinition | None,
) -> PublicJsonObject:
    if request.capability is MediaCapability.IMAGE_GENERATION:
        defaults = profile.image_default_parameters
        overrides = profile.image_override_parameters
        merged: PublicJsonObject = {"n": 1}
    else:
        defaults = profile.video_default_parameters
        overrides = profile.video_override_parameters
        merged = {}
    merged.update(defaults)
    merged.update(request.parameters)
    merged.update(overrides)
    if definition is not None:
        unknown = set(merged).difference(definition.parameters)
        if unknown:
            raise ValueError(f"模型 {definition.model} 不支持参数: {', '.join(sorted(unknown))}")
        for name, value in merged.items():
            definition.parameters[name].validate(name, value)
        output_count = merged.get("n", 1)
        if isinstance(output_count, int) and output_count > definition.max_outputs:
            raise ValueError(f"模型 {definition.model} 最多生成 {definition.max_outputs} 个输出")
    return merged


def media_capabilities() -> tuple[ModelCapability, ...]:
    return tuple(
        ModelCapability(
            model=definition.model,
            capability=definition.capability,
            modes=tuple(definition.modes),
            protocol_families=tuple(sorted(definition.families)),
            max_outputs=definition.max_outputs,
        )
        for definition in MEDIA_MODEL_REGISTRY.values()
    )
