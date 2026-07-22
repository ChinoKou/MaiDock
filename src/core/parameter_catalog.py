from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .parameter_policy import CapabilityKey, ProviderPolicyKey

type ParameterValueKind = Literal["string", "integer", "number", "boolean", "json", "string_list"]
type ParameterFieldLocation = Literal["body", "headers", "query"]


@dataclass(frozen=True, slots=True)
class ParameterFieldDefinition:
    """Provider 目标字段定义，用于 UI 控制和参数转译。"""

    key: str
    label: str
    description: str
    target_path: tuple[str, ...]
    value_kind: ParameterValueKind = "json"
    source_aliases: tuple[str, ...] = ()
    config_key_name: str = ""
    order: int = 0

    @property
    def config_key(self) -> str:
        return safe_parameter_key(self.config_key_name or ".".join(self.target_path))

    @property
    def safe_key(self) -> str:
        return self.config_key

    @property
    def location(self) -> ParameterFieldLocation:
        root = self.target_path[0] if self.target_path else "body"
        if root == "headers":
            return "headers"
        if root == "query":
            return "query"
        return "body"

    @property
    def disable_paths(self) -> tuple[str, ...]:
        return (dotted_path(self.target_path),)

    @property
    def override_path(self) -> tuple[str, ...]:
        return self.target_path

    @property
    def accepted_source_keys(self) -> frozenset[str]:
        return frozenset((self.key, *self.source_aliases))


@dataclass(frozen=True, slots=True)
class CapabilityParameterCatalog:
    """UI 与请求构建器共享的 Provider/能力目标参数目录。"""

    provider: ProviderPolicyKey
    capability: CapabilityKey
    title: str
    fields: tuple[ParameterFieldDefinition, ...]
    direct_body_keys: frozenset[str] = frozenset()
    reserved_body_keys: frozenset[str] = frozenset()

    def field_by_safe_key(self, safe_key: str) -> ParameterFieldDefinition | None:
        normalized = safe_parameter_key(safe_key)
        for field in self.fields:
            if field.config_key == normalized:
                return field
            if safe_parameter_key(field.key) == normalized:
                return field
            for alias in field.source_aliases:
                if safe_parameter_key(alias) == normalized:
                    return field
        return None

    def source_alias_map(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for field in self.fields:
            for source_key in field.accepted_source_keys:
                aliases[source_key] = field.key
        return aliases


_PROVIDER_ORDER: tuple[ProviderPolicyKey, ...] = (
    "openai_responses",
    "anthropic_messages",
    "dashscope",
    "siliconflow",
    "volcengine_ark",
    "xiaomi_mimo",
)
_CAPABILITY_ORDER: tuple[CapabilityKey, ...] = (
    "response",
    "chat_completion",
    "embeddings",
    "audio_transcription",
    "image_generation",
)

PROVIDER_TITLES: dict[ProviderPolicyKey, str] = {
    "openai_responses": "OpenAI Responses",
    "anthropic_messages": "Anthropic Messages",
    "dashscope": "阿里云百炼 DashScope",
    "siliconflow": "SiliconFlow",
    "volcengine_ark": "Volcengine Ark",
    "xiaomi_mimo": "Xiaomi Mimo",
}

CAPABILITY_TITLES: dict[CapabilityKey, str] = {
    "response": "文本生成",
    "chat_completion": "文本生成",
    "embeddings": "Embeddings",
    "audio_transcription": "语音转录",
    "image_generation": "图像生成",
}


def safe_parameter_key(key: str) -> str:
    """将目标路径或上游参数名转换为稳定的配置键前缀。"""

    chars: list[str] = []
    previous_underscore = False
    for char in key.strip().lower():
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
            continue
        if not previous_underscore:
            chars.append("_")
            previous_underscore = True
    safe_key = "".join(chars).strip("_")
    return safe_key or "param"


def field_enabled_key(field: ParameterFieldDefinition) -> str:
    return f"{field.config_key}_enabled"


def field_override_enabled_key(field: ParameterFieldDefinition) -> str:
    return f"{field.config_key}_override_enabled"


def field_override_value_key(field: ParameterFieldDefinition) -> str:
    return f"{field.config_key}_override_value"


def dotted_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _path(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


def _field(
    key: str,
    target_path: str,
    *,
    value_kind: ParameterValueKind = "json",
    description: str = "",
    source_aliases: tuple[str, ...] = (),
    config_key_name: str = "",
    order: int = 0,
    ui_label: str = "",
) -> ParameterFieldDefinition:
    path = _path(target_path)
    clean = list(path)
    if clean and clean[0] in ("body", "headers", "query"):
        clean.pop(0)
    if clean and clean[0] == "parameters":
        clean.pop(0)
    label = ui_label or ".".join(clean)
    return ParameterFieldDefinition(
        key=key,
        label=label,
        description=description or f"Provider API 目标字段 {label}",
        target_path=path,
        value_kind=value_kind,
        source_aliases=source_aliases,
        config_key_name=config_key_name,
        order=order,
    )


def _fields(*fields: ParameterFieldDefinition) -> tuple[ParameterFieldDefinition, ...]:
    return tuple(sorted(fields, key=lambda item: item.order))


def _accepted_keys(fields: tuple[ParameterFieldDefinition, ...], *extra_keys: str) -> frozenset[str]:
    keys: set[str] = set(extra_keys)
    for field in fields:
        keys.update(field.accepted_source_keys)
    return frozenset(keys)


# ── OpenAI Responses ─────────────────────────────────────────

_RESPONSES_FIELDS = _fields(
    _field("temperature", "body.temperature", value_kind="number", order=10),
    _field(
        "max_tokens",
        "body.max_output_tokens",
        value_kind="integer",
        source_aliases=("max_output_tokens",),
        order=20,
    ),
    _field(
        "response_format",
        "body.text.format",
        description="Host response_format 转译到 Responses text.format",
        order=30,
    ),
    _field("top_p", "body.top_p", value_kind="number", order=40),
    _field("reasoning", "body.reasoning", order=50),
    _field("thinking", "body.thinking", order=60),
    _field("text", "body.text", order=70),
    _field("tool_choice", "body.tool_choice", order=80),
    _field(
        "parallel_tool_calls",
        "body.parallel_tool_calls",
        value_kind="boolean",
        ui_label="并行工具调用",
        order=90,
    ),
    _field("max_tool_calls", "body.max_tool_calls", value_kind="integer", order=100),
    _field("include", "body.include", value_kind="string_list", order=110),
    _field("instructions", "body.instructions", value_kind="string", order=120),
    _field("metadata", "body.metadata", order=130),
    _field("store", "body.store", value_kind="boolean", ui_label="存储响应", order=140),
    _field("truncation", "body.truncation", value_kind="string", order=150),
    _field("service_tier", "body.service_tier", value_kind="string", order=160),
    _field(
        "previous_response_id",
        "body.previous_response_id",
        value_kind="string",
        order=170,
    ),
    _field("user", "body.user", value_kind="string", order=180),
    _field("session", "body.session", order=190),
    _field("caching", "body.caching", order=200),
    _field("expire_at", "body.expire_at", value_kind="integer", order=210),
)
_RESPONSES_DIRECT_KEYS = _accepted_keys(_RESPONSES_FIELDS, "tools")
_RESPONSES_RESERVED_KEYS = frozenset({"input", "model", "stream"})

_OPENAI_EMBEDDING_FIELDS = _fields(
    _field("dimensions", "body.dimensions", value_kind="integer", order=10),
    _field("encoding_format", "body.encoding_format", value_kind="string", order=20),
    _field("user", "body.user", value_kind="string", order=30),
)
_OPENAI_EMBEDDING_DIRECT_KEYS = _accepted_keys(_OPENAI_EMBEDDING_FIELDS)
_OPENAI_EMBEDDING_RESERVED_KEYS = frozenset({"input", "model"})

_OPENAI_AUDIO_FIELDS = _fields(
    _field("language", "body.language", value_kind="string", order=10),
    _field("prompt", "body.prompt", value_kind="string", order=20),
    _field("response_format", "body.response_format", value_kind="string", order=30),
    _field("temperature", "body.temperature", value_kind="number", order=40),
    _field(
        "timestamp_granularities",
        "body.timestamp_granularities",
        value_kind="string_list",
        order=50,
    ),
    _field("chunking_strategy", "body.chunking_strategy", order=60),
    _field("include", "body.include", value_kind="string_list", order=70),
    _field("stream", "body.stream", value_kind="boolean", ui_label="流式输出", order=80),
)
_OPENAI_AUDIO_DIRECT_KEYS = _accepted_keys(_OPENAI_AUDIO_FIELDS)
_OPENAI_AUDIO_RESERVED_KEYS = frozenset({"file", "model"})

# ── Anthropic Messages ───────────────────────────────────────

_ANTHROPIC_FIELDS = _fields(
    _field("temperature", "body.temperature", value_kind="number", order=10),
    _field("max_tokens", "body.max_tokens", value_kind="integer", order=20),
    _field("top_p", "body.top_p", value_kind="number", order=30),
    _field("top_k", "body.top_k", value_kind="integer", order=40),
    _field("thinking", "body.thinking", order=50),
    _field("tool_choice", "body.tool_choice", order=60),
    _field("stop_sequences", "body.stop_sequences", value_kind="string_list", order=70),
    _field("metadata", "body.metadata", order=80),
    _field("service_tier", "body.service_tier", value_kind="string", order=90),
)
_ANTHROPIC_DIRECT_KEYS = _accepted_keys(_ANTHROPIC_FIELDS)
_ANTHROPIC_RESERVED_KEYS = frozenset({"messages", "model", "stream", "system", "tools"})

# ── 阿里云百炼 DashScope ─────────────────────────────────────

_DASHSCOPE_CHAT_FIELDS = _fields(
    _field("temperature", "body.parameters.temperature", value_kind="number", order=10),
    _field("max_tokens", "body.parameters.max_tokens", value_kind="integer", order=20),
    _field(
        "max_completion_tokens",
        "body.parameters.max_completion_tokens",
        value_kind="integer",
        order=30,
    ),
    _field("thinking_budget", "body.parameters.thinking_budget", value_kind="integer", order=40),
    _field("reasoning_effort", "body.parameters.reasoning_effort", value_kind="string", order=50),
    _field("response_format", "body.parameters.response_format", order=60),
    _field("result_format", "body.parameters.result_format", value_kind="string", order=70),
    _field("top_p", "body.parameters.top_p", value_kind="number", order=80),
    _field("top_k", "body.parameters.top_k", value_kind="integer", order=90),
    _field(
        "enable_thinking",
        "body.parameters.enable_thinking",
        value_kind="boolean",
        ui_label="启用思考链",
        order=100,
    ),
    _field(
        "enable_search",
        "body.parameters.enable_search",
        value_kind="boolean",
        ui_label="启用搜索",
        order=110,
    ),
    _field("search_options", "body.parameters.search_options", ui_label="搜索选项", order=120),
    _field(
        "incremental_output",
        "body.parameters.incremental_output",
        value_kind="boolean",
        ui_label="增量输出",
        order=130,
    ),
    _field(
        "stream",
        "body.parameters.stream",
        value_kind="boolean",
        ui_label="流式输出",
        order=140,
    ),
    _field(
        "tool_stream",
        "body.parameters.tool_stream",
        value_kind="boolean",
        ui_label="复杂工具参数流式输出",
        order=150,
    ),
    _field(
        "parallel_tool_calls",
        "body.parameters.parallel_tool_calls",
        value_kind="boolean",
        ui_label="并行工具调用",
        order=160,
    ),
    _field(
        "enable_code_interpreter",
        "body.parameters.enable_code_interpreter",
        value_kind="boolean",
        ui_label="启用代码解释器",
        order=170,
    ),
    _field(
        "vl_high_resolution_images",
        "body.parameters.vl_high_resolution_images",
        value_kind="boolean",
        ui_label="高分辨率图像",
        order=180,
    ),
    _field("seed", "body.parameters.seed", value_kind="integer", order=190),
    _field("stop", "body.parameters.stop", order=200),
    _field("n", "body.parameters.n", value_kind="integer", order=210),
    _field(
        "presence_penalty",
        "body.parameters.presence_penalty",
        value_kind="number",
        order=220,
    ),
    _field(
        "repetition_penalty",
        "body.parameters.repetition_penalty",
        value_kind="number",
        order=230,
    ),
    _field("tool_choice", "body.parameters.tool_choice", order=240),
    _field("tools", "body.parameters.tools", order=250),
    _field("plugins", "headers.X-DashScope-Plugin", order=260),
    _field(
        "customized_model_id",
        "body.input.customized_model_id",
        value_kind="string",
        order=270,
    ),
)
_DASHSCOPE_CHAT_DIRECT_KEYS = _accepted_keys(_DASHSCOPE_CHAT_FIELDS)
_DASHSCOPE_CHAT_RESERVED_KEYS = frozenset({"input", "model", "parameters"})

_DASHSCOPE_EMBEDDING_FIELDS = _fields(
    _field(
        "dimensions",
        "body.parameters.dimension",
        value_kind="integer",
        source_aliases=("dimension",),
        order=10,
    ),
    _field("output_type", "body.parameters.output_type", value_kind="string", order=30),
    _field("instruct", "body.parameters.instruct", value_kind="string", order=40),
    _field("text_type", "body.parameters.text_type", value_kind="string", order=50),
    _field(
        "auto_truncation",
        "body.parameters.auto_truncation",
        value_kind="boolean",
        ui_label="自动截断",
        order=60,
    ),
    _field(
        "enable_fusion",
        "body.parameters.enable_fusion",
        value_kind="boolean",
        ui_label="启用融合",
        order=70,
    ),
    _field("fps", "body.parameters.fps", value_kind="number", order=90),
    _field(
        "max_video_frames",
        "body.parameters.max_video_frames",
        value_kind="integer",
        order=100,
    ),
    _field("res_level", "body.parameters.res_level", value_kind="integer", order=110),
)
_DASHSCOPE_EMBEDDING_DIRECT_KEYS = _accepted_keys(_DASHSCOPE_EMBEDDING_FIELDS)
_DASHSCOPE_EMBEDDING_RESERVED_KEYS = frozenset({"input", "model", "parameters"})

_DASHSCOPE_AUDIO_FIELDS = _fields(
    _field(
        "language",
        "body.parameters.asr_options.language",
        value_kind="string",
        order=10,
    ),
    _field(
        "enable_itn",
        "body.parameters.asr_options.enable_itn",
        value_kind="boolean",
        ui_label="逆文本正则化",
        order=20,
    ),
    _field("format", "body.format", value_kind="string", order=30),
    _field("audio_format", "body.audio_format", value_kind="string", order=40),
)
_DASHSCOPE_AUDIO_DIRECT_KEYS = _accepted_keys(_DASHSCOPE_AUDIO_FIELDS)
_DASHSCOPE_AUDIO_RESERVED_KEYS = frozenset({"input", "model", "parameters"})

# ── SiliconFlow ──────────────────────────────────────────────

_SILICONFLOW_CHAT_FIELDS = _fields(
    _field("temperature", "body.temperature", value_kind="number", order=10),
    _field("max_tokens", "body.max_tokens", value_kind="integer", order=20),
    _field("response_format", "body.response_format", order=30),
    _field("top_p", "body.top_p", value_kind="number", order=40),
    _field("tool_choice", "body.tool_choice", order=50),
    _field("tools", "body.tools", order=60),
    _field("frequency_penalty", "body.frequency_penalty", value_kind="number", order=70),
    _field("presence_penalty", "body.presence_penalty", value_kind="number", order=80),
    _field("seed", "body.seed", value_kind="integer", order=90),
    _field("stop", "body.stop", order=100),
    _field("n", "body.n", value_kind="integer", order=110),
)
_SILICONFLOW_CHAT_DIRECT_KEYS = _accepted_keys(_SILICONFLOW_CHAT_FIELDS)
_SILICONFLOW_CHAT_RESERVED_KEYS = frozenset({"messages", "model", "stream"})

_SILICONFLOW_EMBEDDING_FIELDS = _fields(
    _field("dimensions", "body.dimensions", value_kind="integer", order=10),
    _field("encoding_format", "body.encoding_format", value_kind="string", order=20),
)
_SILICONFLOW_EMBEDDING_DIRECT_KEYS = _accepted_keys(_SILICONFLOW_EMBEDDING_FIELDS)
_SILICONFLOW_EMBEDDING_RESERVED_KEYS = frozenset({"input", "model"})

_SILICONFLOW_AUDIO_FIELDS = _fields(
    _field("language", "body.language", value_kind="string", order=10),
    _field("prompt", "body.prompt", value_kind="string", order=20),
    _field("response_format", "body.response_format", value_kind="string", order=30),
    _field("temperature", "body.temperature", value_kind="number", order=40),
    _field(
        "timestamp_granularities",
        "body.timestamp_granularities",
        value_kind="string_list",
        order=50,
    ),
    _field("chunking_strategy", "body.chunking_strategy", order=60),
    _field("include", "body.include", value_kind="string_list", order=70),
    _field("stream", "body.stream", value_kind="boolean", ui_label="流式输出", order=80),
)
_SILICONFLOW_AUDIO_DIRECT_KEYS = _accepted_keys(_SILICONFLOW_AUDIO_FIELDS)
_SILICONFLOW_AUDIO_RESERVED_KEYS = frozenset({"file", "model"})

# ── Volcengine Ark ───────────────────────────────────────────

_ARK_EMBEDDING_FIELDS = _fields(
    _field("dimensions", "body.dimensions", value_kind="integer", order=10),
    _field(
        "sparse_embedding",
        "body.sparse_embedding",
        value_kind="boolean",
        description='对象字段，开启 → {"type": "enabled"}，关闭 → {"type": "disabled"}',
        ui_label="稀疏向量 (type)",
        order=20,
    ),
    _field("encoding_format", "body.encoding_format", value_kind="string", order=30),
)
_ARK_EMBEDDING_DIRECT_KEYS = _accepted_keys(_ARK_EMBEDDING_FIELDS)
_ARK_EMBEDDING_RESERVED_KEYS = frozenset({"input", "model"})

_ARK_AUDIO_FIELDS = _fields(
    _field(
        "max_tokens",
        "body.max_output_tokens",
        value_kind="integer",
        source_aliases=("max_output_tokens",),
        order=10,
    ),
    _field("prompt", "body.prompt", value_kind="string", order=20),
    _field("format", "body.format", value_kind="string", order=30),
    _field("audio_format", "body.audio_format", value_kind="string", order=40),
)
_ARK_AUDIO_DIRECT_KEYS = _accepted_keys(_ARK_AUDIO_FIELDS)
_ARK_AUDIO_RESERVED_KEYS = frozenset({"input", "model", "stream"})

# ── Xiaomi Mimo ──────────────────────────────────────────────

_MIMO_CHAT_FIELDS = _fields(
    _field("temperature", "body.temperature", value_kind="number", order=10),
    _field(
        "max_tokens",
        "body.max_completion_tokens",
        value_kind="integer",
        source_aliases=("max_completion_tokens",),
        config_key_name="body_max_tokens",
        order=20,
    ),
    _field("response_format", "body.response_format", order=30),
    _field("top_p", "body.top_p", value_kind="number", order=40),
    _field("tool_choice", "body.tool_choice", order=50),
    _field("tools", "body.tools", order=60),
    _field("frequency_penalty", "body.frequency_penalty", value_kind="number", order=70),
    _field("presence_penalty", "body.presence_penalty", value_kind="number", order=80),
    _field("seed", "body.seed", value_kind="integer", order=90),
    _field("stop", "body.stop", order=100),
    _field("n", "body.n", value_kind="integer", order=110),
    _field("thinking", "body.thinking", order=120),
)
_MIMO_CHAT_DIRECT_KEYS = _accepted_keys(_MIMO_CHAT_FIELDS)
_MIMO_CHAT_RESERVED_KEYS = frozenset({"messages", "model", "stream"})

_MIMO_AUDIO_FIELDS = _fields(
    _field("language", "body.asr_options.language", value_kind="string", order=10),
    _field("format", "body.format", value_kind="string", order=20),
    _field("audio_format", "body.audio_format", value_kind="string", order=30),
)
_MIMO_AUDIO_DIRECT_KEYS = _accepted_keys(_MIMO_AUDIO_FIELDS)
_MIMO_AUDIO_RESERVED_KEYS = frozenset({"messages", "model", "stream"})


_CATALOGS: dict[tuple[ProviderPolicyKey, CapabilityKey], CapabilityParameterCatalog] = {
    # ── OpenAI Responses ──
    ("openai_responses", "response"): CapabilityParameterCatalog(
        provider="openai_responses",
        capability="response",
        title="OpenAI 文本生成参数",
        fields=_RESPONSES_FIELDS,
        direct_body_keys=_RESPONSES_DIRECT_KEYS,
        reserved_body_keys=_RESPONSES_RESERVED_KEYS,
    ),
    ("openai_responses", "embeddings"): CapabilityParameterCatalog(
        provider="openai_responses",
        capability="embeddings",
        title="OpenAI Embeddings 参数",
        fields=_OPENAI_EMBEDDING_FIELDS,
        direct_body_keys=_OPENAI_EMBEDDING_DIRECT_KEYS,
        reserved_body_keys=_OPENAI_EMBEDDING_RESERVED_KEYS,
    ),
    ("openai_responses", "audio_transcription"): CapabilityParameterCatalog(
        provider="openai_responses",
        capability="audio_transcription",
        title="OpenAI 语音转录参数",
        fields=_OPENAI_AUDIO_FIELDS,
        direct_body_keys=_OPENAI_AUDIO_DIRECT_KEYS,
        reserved_body_keys=_OPENAI_AUDIO_RESERVED_KEYS,
    ),
    # ── Anthropic Messages ──
    ("anthropic_messages", "chat_completion"): CapabilityParameterCatalog(
        provider="anthropic_messages",
        capability="chat_completion",
        title="Anthropic 文本生成参数",
        fields=_ANTHROPIC_FIELDS,
        direct_body_keys=_ANTHROPIC_DIRECT_KEYS,
        reserved_body_keys=_ANTHROPIC_RESERVED_KEYS,
    ),
    # ── 阿里云百炼 DashScope ──
    ("dashscope", "chat_completion"): CapabilityParameterCatalog(
        provider="dashscope",
        capability="chat_completion",
        title="阿里云百炼 DashScope 文本生成参数",
        fields=_DASHSCOPE_CHAT_FIELDS,
        direct_body_keys=_DASHSCOPE_CHAT_DIRECT_KEYS,
        reserved_body_keys=_DASHSCOPE_CHAT_RESERVED_KEYS,
    ),
    ("dashscope", "embeddings"): CapabilityParameterCatalog(
        provider="dashscope",
        capability="embeddings",
        title="阿里云百炼 DashScope Embeddings 参数",
        fields=_DASHSCOPE_EMBEDDING_FIELDS,
        direct_body_keys=_DASHSCOPE_EMBEDDING_DIRECT_KEYS,
        reserved_body_keys=_DASHSCOPE_EMBEDDING_RESERVED_KEYS,
    ),
    ("dashscope", "audio_transcription"): CapabilityParameterCatalog(
        provider="dashscope",
        capability="audio_transcription",
        title="阿里云百炼 DashScope 语音转录参数",
        fields=_DASHSCOPE_AUDIO_FIELDS,
        direct_body_keys=_DASHSCOPE_AUDIO_DIRECT_KEYS,
        reserved_body_keys=_DASHSCOPE_AUDIO_RESERVED_KEYS,
    ),
    # ── SiliconFlow ──
    ("siliconflow", "chat_completion"): CapabilityParameterCatalog(
        provider="siliconflow",
        capability="chat_completion",
        title="SiliconFlow 文本生成参数",
        fields=_SILICONFLOW_CHAT_FIELDS,
        direct_body_keys=_SILICONFLOW_CHAT_DIRECT_KEYS,
        reserved_body_keys=_SILICONFLOW_CHAT_RESERVED_KEYS,
    ),
    ("siliconflow", "embeddings"): CapabilityParameterCatalog(
        provider="siliconflow",
        capability="embeddings",
        title="SiliconFlow Embeddings 参数",
        fields=_SILICONFLOW_EMBEDDING_FIELDS,
        direct_body_keys=_SILICONFLOW_EMBEDDING_DIRECT_KEYS,
        reserved_body_keys=_SILICONFLOW_EMBEDDING_RESERVED_KEYS,
    ),
    ("siliconflow", "audio_transcription"): CapabilityParameterCatalog(
        provider="siliconflow",
        capability="audio_transcription",
        title="SiliconFlow 语音转录参数",
        fields=_SILICONFLOW_AUDIO_FIELDS,
        direct_body_keys=_SILICONFLOW_AUDIO_DIRECT_KEYS,
        reserved_body_keys=_SILICONFLOW_AUDIO_RESERVED_KEYS,
    ),
    # ── Volcengine Ark ──
    ("volcengine_ark", "response"): CapabilityParameterCatalog(
        provider="volcengine_ark",
        capability="response",
        title="Volcengine Ark 文本生成参数",
        fields=_RESPONSES_FIELDS,
        direct_body_keys=_RESPONSES_DIRECT_KEYS,
        reserved_body_keys=_RESPONSES_RESERVED_KEYS,
    ),
    ("volcengine_ark", "embeddings"): CapabilityParameterCatalog(
        provider="volcengine_ark",
        capability="embeddings",
        title="Volcengine Ark Embeddings 参数",
        fields=_ARK_EMBEDDING_FIELDS,
        direct_body_keys=_ARK_EMBEDDING_DIRECT_KEYS,
        reserved_body_keys=_ARK_EMBEDDING_RESERVED_KEYS,
    ),
    ("volcengine_ark", "audio_transcription"): CapabilityParameterCatalog(
        provider="volcengine_ark",
        capability="audio_transcription",
        title="Volcengine Ark 语音转录参数",
        fields=_ARK_AUDIO_FIELDS,
        direct_body_keys=_ARK_AUDIO_DIRECT_KEYS,
        reserved_body_keys=_ARK_AUDIO_RESERVED_KEYS,
    ),
    # ── Xiaomi Mimo ──
    ("xiaomi_mimo", "chat_completion"): CapabilityParameterCatalog(
        provider="xiaomi_mimo",
        capability="chat_completion",
        title="Xiaomi Mimo 文本生成参数",
        fields=_MIMO_CHAT_FIELDS,
        direct_body_keys=_MIMO_CHAT_DIRECT_KEYS,
        reserved_body_keys=_MIMO_CHAT_RESERVED_KEYS,
    ),
    ("xiaomi_mimo", "audio_transcription"): CapabilityParameterCatalog(
        provider="xiaomi_mimo",
        capability="audio_transcription",
        title="Xiaomi Mimo 语音转录参数",
        fields=_MIMO_AUDIO_FIELDS,
        direct_body_keys=_MIMO_AUDIO_DIRECT_KEYS,
        reserved_body_keys=_MIMO_AUDIO_RESERVED_KEYS,
    ),
}


def get_parameter_catalog(provider: ProviderPolicyKey, capability: CapabilityKey) -> CapabilityParameterCatalog:
    catalog = _CATALOGS.get((provider, capability))
    if catalog is None:
        return CapabilityParameterCatalog(
            provider=provider,
            capability=capability,
            title=f"{PROVIDER_TITLES[provider]} {CAPABILITY_TITLES[capability]} 参数",
            fields=(),
        )
    return catalog


def iter_parameter_catalogs() -> Iterable[CapabilityParameterCatalog]:
    for provider in _PROVIDER_ORDER:
        for capability in _CAPABILITY_ORDER:
            catalog = _CATALOGS.get((provider, capability))
            if catalog is not None:
                yield catalog


def provider_catalogs(
    provider: ProviderPolicyKey,
) -> tuple[CapabilityParameterCatalog, ...]:
    return tuple(catalog for catalog in iter_parameter_catalogs() if catalog.provider == provider)
