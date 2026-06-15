import base64
import binascii
import io
import json
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from PIL import Image as PILImage

from ..version import DEFAULT_USER_AGENT
from .diagnostics import sanitize_for_log
from .parameter_policy import ParameterPolicyRegistry
from .parsing import ReasoningParseMode, ToolArgumentParseMode, arguments_to_json
from ..schemas import (
    ApiProviderSnapshot,
    AudioTranscriptionRequestSnapshot,
    BaseProviderRequestSnapshot,
    GenericUsageSnapshot,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
    ModelInfoSnapshot,
    ObjectFields,
    ProviderUsage,
)

SUPPORTED_IMAGE_FORMATS = {"jpeg", "png", "webp"}
InvalidImagePolicy = Literal["placeholder", "skip", "error"]
MAIDOCK_USER_AGENT = DEFAULT_USER_AGENT
DEFAULT_MAX_IMAGE_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000
DEFAULT_MAX_IMAGE_DIMENSION = 8192
DEFAULT_MAX_IMAGE_FRAMES = 64


@dataclass(slots=True)
class ImageProcessingLimits:
    """图片处理资源上限。"""

    max_base64_chars: int = math.ceil(DEFAULT_MAX_IMAGE_BYTES / 3) * 4
    max_decoded_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION
    max_frames: int = DEFAULT_MAX_IMAGE_FRAMES


@dataclass(slots=True)
class ProviderRuntimeOptions:
    """插件运行时配置。"""

    include_raw_data: bool = False
    log_payload_summary: bool = True
    log_payload_debug: bool = False
    tool_argument_parse_mode: ToolArgumentParseMode = "auto"
    reasoning_parse_mode: ReasoningParseMode = "auto"
    invalid_image_policy: InvalidImagePolicy = "placeholder"
    openai_user_agent: str = MAIDOCK_USER_AGENT
    anthropic_user_agent: str = MAIDOCK_USER_AGENT
    dashscope_user_agent: str = MAIDOCK_USER_AGENT
    siliconflow_user_agent: str = MAIDOCK_USER_AGENT
    volcengine_user_agent: str = MAIDOCK_USER_AGENT
    dashscope_force_official_endpoint: bool = True
    dashscope_auto_detect_endpoint: bool = True
    siliconflow_force_official_endpoint: bool = True
    volcengine_force_official_endpoint: bool = True
    mimo_user_agent: str = MAIDOCK_USER_AGENT
    mimo_force_disable_thinking: bool = True
    mimo_audio_transcription_prompt: str = "请转写这段音频"
    openai_max_retries: int = 3
    anthropic_max_retries: int = 3
    dashscope_max_retries: int = 3
    siliconflow_max_retries: int = 3
    volcengine_max_retries: int = 3
    mimo_max_retries: int = 3
    openai_force_max_retries: bool = False
    anthropic_force_max_retries: bool = False
    dashscope_force_max_retries: bool = False
    siliconflow_force_max_retries: bool = False
    volcengine_force_max_retries: bool = False
    mimo_force_max_retries: bool = False
    openai_retry_interval: float = 5.0
    anthropic_retry_interval: float = 5.0
    dashscope_retry_interval: float = 5.0
    siliconflow_retry_interval: float = 5.0
    volcengine_retry_interval: float = 5.0
    mimo_retry_interval: float = 5.0
    openai_force_retry_interval: bool = False
    anthropic_force_retry_interval: bool = False
    dashscope_force_retry_interval: bool = False
    siliconflow_force_retry_interval: bool = False
    volcengine_force_retry_interval: bool = False
    mimo_force_retry_interval: bool = False
    image_limits: ImageProcessingLimits = field(default_factory=ImageProcessingLimits)
    parameter_policies: ParameterPolicyRegistry = field(default_factory=ParameterPolicyRegistry)


@dataclass(slots=True)
class OpenAICompatibleClientConfig:
    """OpenAI 兼容的原生 HTTP 客户端配置。"""

    api_key: str
    base_url: str
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict = field(default_factory=dict)


@dataclass(slots=True)
class RequestOverrides:
    """单次上游 HTTP 请求覆盖参数。"""

    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_query: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)
    direct_params: dict = field(default_factory=dict)


def read_model_identifier(model_info: ModelInfoSnapshot) -> str:
    """读取模型标识。"""

    model = model_info.model_identifier or model_info.name
    if model is None or not model.strip():
        raise ValueError("LLM Provider 请求缺少 model_info.model_identifier")
    return model.strip()


def read_api_key(api_provider: ApiProviderSnapshot, *, allow_empty: bool = False) -> str:
    """读取 API key。"""

    api_key = api_provider.api_key.strip()
    if not api_key and not allow_empty:
        raise ValueError("api_provider.api_key 为空，无法调用需要鉴权的上游 LLM API")
    return api_key


def read_timeout(api_provider: ApiProviderSnapshot) -> float | None:
    """读取 timeout。"""

    timeout = api_provider.timeout
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    return None


def resolve_max_retries(
    api_provider: ApiProviderSnapshot,
    *,
    config_value: int,
    force: bool,
    default: int,
) -> int:
    """按 force/fallback 逻辑计算有效的最大重试次数。"""

    if force:
        return config_value
    host_value = api_provider.max_retry
    if isinstance(host_value, int) and host_value >= 0:
        return host_value
    return config_value if config_value >= 0 else default


def resolve_retry_interval(
    api_provider: ApiProviderSnapshot,
    *,
    config_value: float,
    force: bool,
    default: float,
) -> float:
    """
    按 force/fallback 逻辑计算有效的重试间隔（秒）。

    非 force 模式下，0 视为无效值：Host 传入 0 或配置值为 0 时不会启用零间隔重试；
    若需零间隔（立刻重试），请将 force 设为 True 并将 config_value 设为 0。
    """

    if force:
        return config_value
    host_value = api_provider.retry_interval
    if isinstance(host_value, (int, float)) and host_value > 0:
        return float(host_value)
    return config_value if config_value > 0 else default


def read_max_retries(api_provider: ApiProviderSnapshot, default: int) -> int:
    """读取原生 HTTP 最大重试次数（向后兼容，委托 resolve_max_retries）。"""

    return resolve_max_retries(api_provider, config_value=default, force=False, default=default)


def merge_extra_params(request: BaseProviderRequestSnapshot) -> dict:
    """合并模型级和请求级 extra_params，请求级覆盖模型级。"""

    merged: dict = {}
    for source in (request.model_info.extra_params, request.extra_params):
        for key, value in source.fields.items():
            if value is not None:
                merged[key] = value
    return merged


def pop_json_object(payload: dict, key: str) -> dict:
    """从 payload 取出 object 字段。"""

    value = payload.pop(key, None)
    return ObjectFields.from_unknown(value).to_plain_dict()


def require_string_dict(value: Mapping[str, object], *, field_name: str) -> dict[str, str]:
    """校验字符串字典。"""

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise TypeError(f"{field_name}.{key} 必须是字符串，实际为 {type(item).__name__}")
        result[str(key)] = item
    return result


def require_string_mapping(value: ObjectFields, *, field_name: str) -> dict[str, str]:
    """校验 ObjectFields 为字符串字典。"""

    return require_string_dict(value.fields, field_name=field_name)


def normalize_base_url(base_url: str | None) -> str:
    """规范化 base_url。"""

    if not base_url or not base_url.strip():
        raise ValueError("api_provider.base_url 为空，无法构建 HTTP 客户端配置")
    normalized = base_url.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = "https://" + normalized
    parsed = urlsplit(normalized)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url 仅支持 http/https（{base_url!r}）")
    if not parsed.netloc:
        raise ValueError(f"base_url 缺少主机名（{base_url!r}）")
    return normalized.rstrip("/")


def _build_auth_header_value(prefix: str, api_key: str) -> str:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return api_key
    return f"{normalized_prefix} {api_key}"


def with_default_user_agent(headers: Mapping[str, str], user_agent: str | None = MAIDOCK_USER_AGENT) -> dict[str, str]:
    result = dict(headers)
    if not any(key.lower() == "user-agent" for key in result):
        normalized_user_agent = (user_agent or "").strip() or MAIDOCK_USER_AGENT
        result["User-Agent"] = normalized_user_agent
    return result


def build_openai_compatible_client_config(
    api_provider: ApiProviderSnapshot,
    *,
    user_agent: str | None = MAIDOCK_USER_AGENT,
) -> OpenAICompatibleClientConfig:
    """按 OpenAI-compatible 规则构造原生 HTTP 鉴权配置。"""

    default_headers = require_string_mapping(api_provider.default_headers, field_name="api_provider.default_headers")
    default_query = api_provider.default_query.to_plain_dict()
    auth_type = (api_provider.auth_type or "bearer").strip().lower()
    api_key = api_provider.api_key.strip()
    client_api_key = api_key

    if auth_type == "bearer":
        if api_provider.auth_header_name != "Authorization" or api_provider.auth_header_prefix.strip() != "Bearer":
            client_api_key = ""
            default_headers[api_provider.auth_header_name] = _build_auth_header_value(
                api_provider.auth_header_prefix,
                api_key,
            )
    elif auth_type == "header":
        client_api_key = ""
        default_headers[api_provider.auth_header_name] = _build_auth_header_value(
            api_provider.auth_header_prefix, api_key
        )
    elif auth_type == "query":
        client_api_key = ""
        default_query[api_provider.auth_query_name] = api_key
    elif auth_type == "none":
        client_api_key = ""
    else:
        raise ValueError(f"不支持的 auth_type: {api_provider.auth_type}")

    if auth_type != "none" and not api_key:
        raise ValueError("api_provider.api_key 为空，无法构建鉴权配置")

    return OpenAICompatibleClientConfig(
        api_key=client_api_key,
        base_url=normalize_base_url(api_provider.base_url),
        default_headers=with_default_user_agent(default_headers, user_agent),
        default_query=default_query,
    )


def split_request_overrides(
    extra_params: Mapping[str, object] | None,
    *,
    direct_body_keys: set[str] | None = None,
    reserved_body_keys: set[str] | None = None,
) -> RequestOverrides:
    """拆分 headers/query/body/direct params。"""

    raw_params = dict(extra_params or {})
    extra_headers = require_string_dict(pop_json_object(raw_params, "headers"), field_name="extra_params.headers")
    extra_query = pop_json_object(raw_params, "query")
    extra_body = pop_json_object(raw_params, "body")
    direct_params: dict = {}
    direct_keys = direct_body_keys or set()
    blocked_keys = reserved_body_keys or set()

    for key, value in raw_params.items():
        if key in direct_keys:
            direct_params[key] = value
            continue
        if key in blocked_keys:
            continue
        extra_body[key] = value

    return RequestOverrides(
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        direct_params=direct_params,
    )


def message_text(message: MessageSnapshot) -> str:
    """拼接消息中的文本片段。"""

    chunks = [part.text for part in message.parts if isinstance(part, MessagePartText) and part.text]
    return "\n".join(chunks)


def image_media_type(image_format: str | None) -> str:
    """把图片格式规整为 media type。"""

    fmt = str(image_format or "png").lower().lstrip(".")
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"jpeg", "png", "webp", "gif"}:
        fmt = "png"
    return f"image/{fmt}"


def _validate_base64_size(image_base64: str, limits: ImageProcessingLimits) -> bool:
    return limits.max_base64_chars <= 0 or len(image_base64) <= limits.max_base64_chars


def _validate_decoded_size(image_bytes: bytes, limits: ImageProcessingLimits) -> bool:
    return limits.max_decoded_bytes <= 0 or len(image_bytes) <= limits.max_decoded_bytes


def _validate_image_geometry(image: PILImage.Image, limits: ImageProcessingLimits) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效")
    if limits.max_dimension > 0 and (width > limits.max_dimension or height > limits.max_dimension):
        raise ValueError("图片单边尺寸超过限制")
    if limits.max_pixels > 0 and width * height > limits.max_pixels:
        raise ValueError("图片像素数量超过限制")
    frame_count = int(getattr(image, "n_frames", 1) or 1)
    if limits.max_frames > 0 and frame_count > limits.max_frames:
        raise ValueError("图片帧数超过限制")


def normalize_image_for_openai(
    part: MessagePartImage,
    logger: logging.Logger,
    limits: ImageProcessingLimits | None = None,
) -> tuple[str, str] | None:
    """将图片规整为 OpenAI Responses 接受的 data URL 片段。"""

    active_limits = limits or ImageProcessingLimits()
    if not part.image_base64:
        return None
    if not _validate_base64_size(part.image_base64, active_limits):
        logger.warning("图片 Base64 长度超过限制，已按配置处理该图片片段")
        return None
    try:
        image_bytes = base64.b64decode(part.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("图片 Base64 解码失败，已按配置处理该图片片段: %s", exc)
        return None
    if not _validate_decoded_size(image_bytes, active_limits):
        logger.warning("图片解码后大小超过限制，已按配置处理该图片片段")
        return None

    original_max_pixels = PILImage.MAX_IMAGE_PIXELS
    try:
        PILImage.MAX_IMAGE_PIXELS = active_limits.max_pixels if active_limits.max_pixels > 0 else original_max_pixels
        with warnings.catch_warnings():
            warnings.simplefilter("error", PILImage.DecompressionBombWarning)
            with PILImage.open(io.BytesIO(image_bytes)) as image:
                _validate_image_geometry(image, active_limits)
                image_format = (image.format or part.image_format or "png").lower()
                if image_format == "jpg":
                    image_format = "jpeg"
                if image_format in SUPPORTED_IMAGE_FORMATS:
                    return image_format, part.image_base64
                if image_format == "gif":
                    return _convert_gif_to_webp(image, active_limits)
                return _convert_static_image_to_png(image, active_limits)
    except Exception as exc:
        logger.warning("图片内容无法识别或超过处理限制，已按配置处理该图片片段: %s", exc)
        return None
    finally:
        PILImage.MAX_IMAGE_PIXELS = original_max_pixels


def _convert_gif_to_webp(image: PILImage.Image, limits: ImageProcessingLimits) -> tuple[str, str]:
    _validate_image_geometry(image, limits)
    frame_count = int(getattr(image, "n_frames", 1) or 1)
    frames: list[PILImage.Image] = []
    durations: list[int] = []
    for frame_index in range(frame_count):
        image.seek(frame_index)
        _validate_image_geometry(image, limits)
        frame = image.copy()
        if frame.mode not in {"RGB", "RGBA"}:
            frame = frame.convert("RGBA")
        frames.append(frame)
        durations.append(int(image.info.get("duration", 100) or 100))

    output_buffer = io.BytesIO()
    if frame_count > 1:
        frames[0].save(
            output_buffer,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=int(image.info.get("loop", 0) or 0),
            lossless=True,
        )
    else:
        frames[0].save(
            output_buffer,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=int(image.info.get("loop", 0) or 0),
        )
    return "webp", base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _convert_static_image_to_png(image: PILImage.Image, limits: ImageProcessingLimits) -> tuple[str, str]:
    _validate_image_geometry(image, limits)
    normalized_image = image.copy()
    if normalized_image.mode not in {"RGB", "RGBA"}:
        normalized_image = normalized_image.convert("RGBA")
    output_buffer = io.BytesIO()
    normalized_image.save(output_buffer, format="PNG")
    return "png", base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def image_data_url(
    part: MessagePartImage,
    logger: logging.Logger,
    invalid_policy: InvalidImagePolicy,
    limits: ImageProcessingLimits | None = None,
) -> str | None:
    """构造图片 data URL，非法图片按策略处理。"""

    normalized_image = normalize_image_for_openai(part, logger, limits)
    if normalized_image is None:
        if invalid_policy == "error":
            raise ValueError("图片数据无效，无法构建上游请求")
        return None
    image_format, image_base64 = normalized_image
    return f"data:image/{image_format};base64,{image_base64}"


def tool_arguments_to_json(value: ObjectFields | str | None, parse_mode: ToolArgumentParseMode) -> str:
    """把工具参数转为 JSON 字符串。"""

    return arguments_to_json(value, parse_mode)


def build_usage(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
) -> ProviderUsage:
    """构建统一 usage。"""

    normalized_prompt_tokens = int(prompt_tokens or 0)
    normalized_completion_tokens = int(completion_tokens or 0)
    normalized_total_tokens = (
        int(total_tokens) if total_tokens is not None else normalized_prompt_tokens + normalized_completion_tokens
    )
    return ProviderUsage(
        prompt_tokens=normalized_prompt_tokens,
        completion_tokens=normalized_completion_tokens,
        total_tokens=normalized_total_tokens,
        prompt_cache_hit_tokens=int(prompt_cache_hit_tokens or 0),
        prompt_cache_miss_tokens=int(prompt_cache_miss_tokens or 0),
    )


def build_usage_from_snapshot(usage: GenericUsageSnapshot) -> ProviderUsage:
    """从宽松 usage 模型构建 Host usage。"""

    prompt_tokens = usage.prompt_tokens or usage.input_tokens
    completion_tokens = usage.completion_tokens or usage.output_tokens
    total_tokens = usage.total_tokens or prompt_tokens + completion_tokens
    cache_hit_tokens = usage.prompt_cache_hit_tokens or usage.cache_read_input_tokens
    cache_miss_tokens = usage.prompt_cache_miss_tokens

    input_details = usage.input_tokens_details.to_plain_dict()
    prompt_details = usage.prompt_tokens_details.to_plain_dict()
    if cache_hit_tokens == 0:
        for details in (input_details, prompt_details):
            cached_tokens = details.get("cached_tokens") or details.get("cache_read_input_tokens")
            if isinstance(cached_tokens, (int, float)):
                cache_hit_tokens = int(cached_tokens)
                break
    if cache_miss_tokens == 0:
        cache_creation = usage.cache_creation_input_tokens
        if cache_creation:
            cache_miss_tokens = cache_creation
        elif cache_hit_tokens:
            cache_miss_tokens = max(prompt_tokens - cache_hit_tokens, 0)

    return build_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=cache_hit_tokens,
        prompt_cache_miss_tokens=cache_miss_tokens,
    )


def build_audio_file(
    audio_request: AudioTranscriptionRequestSnapshot,
) -> tuple[str, io.BytesIO]:
    """把 Host 的音频 base64 转为上游 multipart 文件对象。"""

    if not audio_request.audio_base64:
        raise ValueError("音频转写请求缺少 audio_base64")
    try:
        audio_bytes = base64.b64decode(audio_request.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 不是有效的 Base64 数据") from exc
    return "audio.wav", io.BytesIO(audio_bytes)


def log_request_summary(
    logger: logging.Logger,
    *,
    provider_label: str,
    model: str,
    messages: int | None = None,
    tools: int | None = None,
    extra: Mapping[str, object] | None = None,
    options: ProviderRuntimeOptions,
) -> None:
    """按配置记录请求摘要。"""

    if not options.log_payload_summary:
        return
    logger.info(
        "[MaiDock/%s] request model=%s messages=%s tools=%s",
        provider_label,
        model,
        messages,
        tools,
    )
    if options.log_payload_debug and extra is not None:
        logger.debug(
            "[MaiDock/%s] request payload=%s",
            provider_label,
            sanitize_for_log(dict(extra)),
        )


def log_response_summary(
    logger: logging.Logger,
    *,
    provider_label: str,
    content: str | None,
    tool_calls: Sequence[object],
    usage: ProviderUsage,
    options: ProviderRuntimeOptions,
) -> None:
    """按配置记录响应摘要。"""

    if not options.log_payload_summary:
        return
    logger.info(
        "[MaiDock/%s] response content_len=%s tool_calls=%s usage=%s",
        provider_label,
        len(content or ""),
        len(tool_calls),
        json.dumps(usage.model_dump(mode="json"), ensure_ascii=False),
    )
