import base64
import binascii
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image as PILImage

from .diagnostics import sanitize_for_log
from .parsing import ReasoningParseMode, ToolArgumentParseMode, arguments_to_json
from .schemas import (
    ApiProviderSnapshot,
    AudioTranscriptionRequestSnapshot,
    BaseProviderRequestSnapshot,
    GenericUsageSnapshot,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
    ModelInfoSnapshot,
    ObjectFields,
    OpenAITextConfig,
    OpenAITextFormatConfig,
    ProviderUsage,
    ResponseFormatSchemaSnapshot,
    ResponseRequestSnapshot,
)

SUPPORTED_IMAGE_FORMATS = {"jpeg", "png", "webp"}
InvalidImagePolicy = Literal["placeholder", "skip", "error"]
MAIDOCK_USER_AGENT = "MaiDock/1.0.0"


@dataclass(slots=True)
class ProviderRuntimeOptions:
    """插件运行时配置。"""

    include_raw_data: bool = False
    log_payload_summary: bool = True
    log_payload_debug: bool = False
    tool_argument_parse_mode: ToolArgumentParseMode = "auto"
    reasoning_parse_mode: ReasoningParseMode = "auto"
    strict_extra_params: bool = False
    invalid_image_policy: InvalidImagePolicy = "placeholder"


@dataclass(slots=True)
class OpenAICompatibleClientConfig:
    """OpenAI SDK 初始化配置。"""

    api_key: str
    base_url: str | None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_query: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RequestOverrides:
    """SDK 单次请求覆盖参数。"""

    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_query: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    direct_params: dict[str, Any] = field(default_factory=dict)


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


def read_max_retries(api_provider: ApiProviderSnapshot, default: int) -> int:
    """读取 SDK 最大重试次数。"""

    if isinstance(api_provider.max_retry, int) and api_provider.max_retry >= 0:
        return api_provider.max_retry
    return default


def merge_extra_params(request: BaseProviderRequestSnapshot) -> dict[str, Any]:
    """合并模型级和请求级 extra_params，请求级覆盖模型级。"""

    merged: dict[str, Any] = {}
    for source in (request.model_info.extra_params, request.extra_params):
        for key, value in source.fields.items():
            if value is not None:
                merged[key] = value
    return merged


def pop_json_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """从 payload 取出 object 字段。"""

    value = payload.pop(key, None)
    return ObjectFields.from_unknown(value).to_plain_dict()


def require_string_dict(value: Mapping[str, Any], *, field_name: str) -> dict[str, str]:
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


def normalize_base_url(base_url: str | None) -> str | None:
    """规范化 base_url。"""

    if base_url is None:
        return None
    normalized = base_url.strip()
    if not normalized:
        return None
    if "://" not in normalized:
        normalized = "http://" + normalized
    return normalized.rstrip("/")


def normalize_anthropic_base_url(base_url: str | None) -> str | None:
    """规范化 Anthropic SDK base_url，避免重复拼接 /v1。"""

    normalized = normalize_base_url(base_url)
    if normalized is None:
        return None
    if normalized.lower().endswith("/v1"):
        return normalized[:-3]
    return normalized


def _build_auth_header_value(prefix: str, api_key: str) -> str:
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        return api_key
    return f"{normalized_prefix} {api_key}"


def with_default_user_agent(headers: Mapping[str, str], user_agent: str = MAIDOCK_USER_AGENT) -> dict[str, str]:
    result = dict(headers)
    if not any(key.lower() == "user-agent" for key in result):
        result["User-Agent"] = user_agent
    return result


def build_openai_compatible_client_config(api_provider: ApiProviderSnapshot) -> OpenAICompatibleClientConfig:
    """按 OpenAI 兼容规则构造 SDK 鉴权配置。"""

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
        default_headers=with_default_user_agent(default_headers),
        default_query=default_query,
    )


def build_anthropic_client_config(api_provider: ApiProviderSnapshot) -> OpenAICompatibleClientConfig:
    """构造 Anthropic SDK 初始化配置。"""

    default_headers = require_string_mapping(api_provider.default_headers, field_name="api_provider.default_headers")
    default_query = api_provider.default_query.to_plain_dict()
    auth_type = (api_provider.auth_type or "bearer").strip().lower()
    api_key = api_provider.api_key.strip()
    client_api_key = api_key

    if auth_type == "bearer":
        client_api_key = api_key
    elif auth_type == "header":
        if api_provider.auth_header_name.lower() == "x-api-key":
            client_api_key = _build_auth_header_value(api_provider.auth_header_prefix, api_key)
        else:
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
        base_url=normalize_anthropic_base_url(api_provider.base_url),
        default_headers=with_default_user_agent(default_headers),
        default_query=default_query,
    )


def split_request_overrides(
    extra_params: Mapping[str, Any] | None,
    *,
    direct_body_keys: set[str] | None = None,
    reserved_body_keys: set[str] | None = None,
    strict_extra_params: bool = False,
) -> RequestOverrides:
    """拆分 headers/query/body/direct params。"""

    raw_params = dict(extra_params or {})
    extra_headers = require_string_dict(pop_json_object(raw_params, "headers"), field_name="extra_params.headers")
    extra_query = pop_json_object(raw_params, "query")
    extra_body = pop_json_object(raw_params, "body")
    direct_params: dict[str, Any] = {}
    direct_keys = direct_body_keys or set()
    blocked_keys = reserved_body_keys or set()

    unknown_keys: list[str] = []
    for key, value in raw_params.items():
        if key in direct_keys:
            direct_params[key] = value
            continue
        if key in blocked_keys:
            continue
        if strict_extra_params:
            unknown_keys.append(key)
            continue
        extra_body[key] = value

    if unknown_keys:
        raise ValueError(f"不支持这些 extra_params 字段: {', '.join(sorted(unknown_keys))}")

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


def normalize_image_for_openai(part: MessagePartImage, logger: logging.Logger) -> tuple[str, str] | None:
    """将图片规整为 OpenAI Responses 接受的 data URL 片段。"""

    if not part.image_base64:
        return None
    try:
        image_bytes = base64.b64decode(part.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("图片 Base64 解码失败，已按配置处理该图片片段: %s", exc)
        return None

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or part.image_format or "png").lower()
            if image_format == "jpg":
                image_format = "jpeg"
            if image_format in SUPPORTED_IMAGE_FORMATS:
                return image_format, part.image_base64
            if image_format == "gif":
                return _convert_gif_to_webp(image)
            return _convert_static_image_to_png(image)
    except Exception as exc:
        logger.warning("图片内容无法识别为有效图片，已按配置处理该图片片段: %s", exc)
        return None


def _convert_gif_to_webp(image: PILImage.Image) -> tuple[str, str]:
    frame_count = getattr(image, "n_frames", 1)
    frames: list[PILImage.Image] = []
    durations: list[int] = []
    for frame_index in range(frame_count):
        image.seek(frame_index)
        frame = image.copy()
        if frame.mode not in {"RGB", "RGBA"}:
            frame = frame.convert("RGBA")
        frames.append(frame)
        durations.append(int(image.info.get("duration", 100) or 100))

    output_buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {
        "format": "WEBP",
        "save_all": True,
        "append_images": frames[1:],
        "duration": durations,
        "loop": int(image.info.get("loop", 0) or 0),
    }
    if frame_count > 1:
        save_kwargs["lossless"] = True
    frames[0].save(output_buffer, **save_kwargs)
    return "webp", base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _convert_static_image_to_png(image: PILImage.Image) -> tuple[str, str]:
    normalized_image = image.copy()
    if normalized_image.mode not in {"RGB", "RGBA"}:
        normalized_image = normalized_image.convert("RGBA")
    output_buffer = io.BytesIO()
    normalized_image.save(output_buffer, format="PNG")
    return "png", base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def image_data_url(part: MessagePartImage, logger: logging.Logger, invalid_policy: InvalidImagePolicy) -> str | None:
    """构造图片 data URL，非法图片按策略处理。"""

    normalized_image = normalize_image_for_openai(part, logger)
    if normalized_image is None:
        if invalid_policy == "error":
            raise ValueError("图片数据无效，无法构建上游请求")
        return None
    image_format, image_base64 = normalized_image
    return f"data:image/{image_format};base64,{image_base64}"


def tool_arguments_to_json(value: ObjectFields | str | None, parse_mode: ToolArgumentParseMode) -> str:
    """把工具参数转为 JSON 字符串。"""

    return arguments_to_json(value, parse_mode)


def extract_response_format(request: ResponseRequestSnapshot) -> OpenAITextConfig | None:
    """将 Host response_format 转换为 OpenAI Responses text 配置。"""

    response_format = request.response_format
    if response_format is None or response_format.format_type in {None, "text"}:
        return None
    if response_format.format_type in {"json_object", "json_obj"}:
        return OpenAITextConfig(format=OpenAITextFormatConfig(type="json_object"))
    if response_format.format_type != "json_schema" or response_format.schema_ is None:
        return None

    schema_payload = response_format.schema_
    if isinstance(schema_payload, ResponseFormatSchemaSnapshot):
        schema = schema_payload.schema_ if schema_payload.schema_ is not None else ObjectFields()
        return OpenAITextConfig(
            format=OpenAITextFormatConfig(
                type="json_schema",
                name=schema_payload.name or "maibot_response",
                description=schema_payload.description or "Maibot structured response",
                schema=schema,
                strict=schema_payload.strict,
            )
        )
    return OpenAITextConfig(
        format=OpenAITextFormatConfig(
            type="json_schema",
            name="maibot_response",
            description="Maibot structured response",
            schema=schema_payload,
            strict=False,
        )
    )


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


def build_audio_file(audio_request: AudioTranscriptionRequestSnapshot) -> tuple[str, io.BytesIO]:
    """把 Host 的音频 base64 转为 SDK 文件对象。"""

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
    extra: Mapping[str, Any] | None = None,
    options: ProviderRuntimeOptions,
) -> None:
    """按配置记录请求摘要。"""

    if not options.log_payload_summary:
        return
    logger.info("[MaiDock/%s] request model=%s messages=%s tools=%s", provider_label, model, messages, tools)
    if options.log_payload_debug and extra is not None:
        logger.debug("[MaiDock/%s] request payload=%s", provider_label, sanitize_for_log(dict(extra)))


def log_response_summary(
    logger: logging.Logger,
    *,
    provider_label: str,
    content: str | None,
    tool_calls: list[Any],
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
