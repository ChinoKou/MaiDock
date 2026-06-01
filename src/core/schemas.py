from collections.abc import Mapping
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, object] | list[object]
JsonObject: TypeAlias = dict[str, object]
JsonArray: TypeAlias = list[object]
OpenAITextVerbosity: TypeAlias = Literal["low", "medium", "high"]
AnthropicImageMediaType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

_DICT_ADAPTER = TypeAdapter(JsonObject)
_LIST_ADAPTER = TypeAdapter(JsonArray)


@runtime_checkable
class ModelDumpable(Protocol):
    """OpenAI/Anthropic SDK Pydantic-like 响应对象协议。"""

    def model_dump(self, mode: str = "python") -> object:
        """导出 SDK 对象。"""


class HostDumpModel(BaseModel):
    """可以安全导出给 Host 的模型基类。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def to_host_dict(self) -> JsonObject:
        return _DICT_ADAPTER.validate_python(self.model_dump(mode="json", exclude_none=True, by_alias=True))


class IgnoreExtraModel(BaseModel):
    """Host 快照和上游响应都可能新增字段，读取时忽略未知项。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, arbitrary_types_allowed=True)


class ObjectFields(IgnoreExtraModel):
    """显式 object 包装，避免宽泛类型在 Pydantic 边界散落。"""

    fields: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_unknown(cls, value: object) -> "ObjectFields":
        if isinstance(value, ObjectFields):
            return value
        if isinstance(value, BaseModel):
            return cls(fields=_DICT_ADAPTER.validate_python(value.model_dump(mode="json", exclude_none=True)))
        if isinstance(value, ModelDumpable):
            return cls(fields=_DICT_ADAPTER.validate_python(value.model_dump(mode="python")))
        if isinstance(value, Mapping):
            return cls(fields=_DICT_ADAPTER.validate_python(dict(value)))
        if value is None:
            return cls()
        raise TypeError(f"期望 object/mapping，实际为 {type(value).__name__}")

    def to_plain_dict(self) -> JsonObject:
        return dict(self.fields)

    @field_validator("fields", mode="before")
    @classmethod
    def validate_fields(cls, value: object) -> JsonObject:
        if isinstance(value, Mapping):
            return _DICT_ADAPTER.validate_python(dict(value))
        if value is None:
            return {}
        raise TypeError(f"ObjectFields.fields 必须是 mapping，实际为 {type(value).__name__}")


class ProviderFunctionCall(HostDumpModel):
    """返回给 Host 的函数调用。"""

    name: str
    arguments: JsonObject = Field(default_factory=dict)


class ProviderToolCall(HostDumpModel):
    """返回给 Host 的工具调用。"""

    id: str
    function: ProviderFunctionCall
    extra_content: JsonObject = Field(default_factory=dict)


class ProviderUsage(HostDumpModel):
    """返回给 Host 的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0


class ProviderResponse(HostDumpModel):
    """LLM Provider 返回给 Host 的统一响应。"""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ProviderToolCall] = Field(default_factory=list)
    embedding: list[float] | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    raw_data: JsonObject | None = None


class ApiProviderSnapshot(IgnoreExtraModel):
    """Host 注入的 API Provider 配置快照。"""

    name: str = ""
    api_key: str = ""
    base_url: str | None = None
    client_type: str = ""
    auth_type: str = "bearer"
    auth_header_name: str = "Authorization"
    auth_header_prefix: str = "Bearer"
    auth_query_name: str = "api_key"
    default_headers: ObjectFields = Field(default_factory=ObjectFields)
    default_query: ObjectFields = Field(default_factory=ObjectFields)
    organization: str | None = None
    project: str | None = None
    model_list_endpoint: str = "/models"
    reasoning_parse_mode: str = "auto"
    tool_argument_parse_mode: str = "auto"
    timeout: int | float | None = None
    max_retry: int | None = None
    retry_interval: int | None = None

    @field_validator("default_headers", "default_query", mode="before")
    @classmethod
    def validate_object_fields(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)


class ModelInfoSnapshot(IgnoreExtraModel):
    """Host 序列化后的模型信息。"""

    model_identifier: str | None = None
    name: str | None = None
    api_provider: str | None = None
    temperature: int | float | None = None
    max_tokens: int | None = None
    force_stream_mode: bool = False
    visual: bool = False
    extra_params: ObjectFields = Field(default_factory=ObjectFields)

    @field_validator("extra_params", mode="before")
    @classmethod
    def validate_extra_params(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)


class MessagePartText(IgnoreExtraModel):
    """文本消息片段。"""

    type: Literal["text"] = "text"
    text: str = ""


class MessagePartImage(IgnoreExtraModel):
    """图片消息片段。"""

    type: Literal["image"] = "image"
    image_base64: str = ""
    image_format: str | None = None


class MessagePartUnknown(IgnoreExtraModel):
    """未知消息片段。"""

    type: str = "unknown"


MessagePart: TypeAlias = MessagePartText | MessagePartImage | MessagePartUnknown


class FunctionCallSnapshot(IgnoreExtraModel):
    """Host 快照中的 function call。"""

    name: str | None = None
    arguments: ObjectFields | str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_arguments(cls, value: object) -> ObjectFields | str | None:
        if isinstance(value, str) or value is None:
            return value
        return ObjectFields.from_unknown(value)


class ToolCallSnapshot(IgnoreExtraModel):
    """Host 快照中的工具调用。"""

    id: str | None = None
    call_id: str | None = None
    function: FunctionCallSnapshot = Field(default_factory=FunctionCallSnapshot)
    extra_content: ObjectFields = Field(default_factory=ObjectFields)

    @field_validator("extra_content", mode="before")
    @classmethod
    def validate_extra_content(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)

    def resolved_call_id(self) -> str:
        return (self.id or self.call_id or "").strip()


def _default_tool_parameters() -> ObjectFields:
    return ObjectFields(fields={"type": "object", "properties": {}})


def _validate_tool_parameters(value: object) -> ObjectFields:
    parsed = ObjectFields.from_unknown(value)
    if not parsed.fields:
        return _default_tool_parameters()
    return parsed


class ToolFunctionSnapshot(IgnoreExtraModel):
    """工具函数定义。"""

    name: str | None = None
    description: str = ""
    parameters: ObjectFields = Field(default_factory=_default_tool_parameters)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> ObjectFields:
        return _validate_tool_parameters(value)


class ToolOptionSnapshot(IgnoreExtraModel):
    """Host 快照中的工具定义。"""

    type: str | None = None
    function: ToolFunctionSnapshot | None = None
    name: str | None = None
    description: str = ""
    parameters: ObjectFields = Field(default_factory=_default_tool_parameters)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> ObjectFields:
        return _validate_tool_parameters(value)

    def function_definition(self) -> ToolFunctionSnapshot:
        if self.type == "function" and self.function is not None:
            return self.function
        return ToolFunctionSnapshot(name=self.name, description=self.description, parameters=self.parameters)


class ResponseFormatSchemaSnapshot(IgnoreExtraModel):
    """JSON schema 响应格式。"""

    name: str | None = None
    description: str | None = None
    schema_: ObjectFields | None = Field(default=None, alias="schema")
    strict: bool = False

    @field_validator("schema_", mode="before")
    @classmethod
    def validate_schema(cls, value: object) -> ObjectFields | None:
        return None if value is None else ObjectFields.from_unknown(value)


class ResponseFormatSnapshot(IgnoreExtraModel):
    """Host 序列化后的 response_format。"""

    format_type: str | None = None
    schema_: ResponseFormatSchemaSnapshot | ObjectFields | None = Field(default=None, alias="schema")

    @field_validator("schema_", mode="before")
    @classmethod
    def validate_schema(cls, value: object) -> ResponseFormatSchemaSnapshot | ObjectFields | None:
        if value is None:
            return None
        if isinstance(value, (ResponseFormatSchemaSnapshot, ObjectFields)):
            return value
        if isinstance(value, Mapping) and any(key in value for key in ("name", "description", "schema", "strict")):
            return ResponseFormatSchemaSnapshot.model_validate(dict(value))
        return ObjectFields.from_unknown(value)


class MessageSnapshot(IgnoreExtraModel):
    """Host 序列化后的消息。"""

    role: str = ""
    parts: list[MessagePart] = Field(default_factory=list)
    tool_calls: list[ToolCallSnapshot] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None

    @field_validator("parts", mode="before")
    @classmethod
    def validate_parts(cls, value: object) -> list[MessagePart]:
        if not isinstance(value, list):
            return []
        normalized: list[MessagePart] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            item_dict = dict(item)
            part_type = item_dict.get("type")
            if part_type == "text":
                normalized.append(MessagePartText.model_validate(item_dict))
            elif part_type == "image":
                normalized.append(MessagePartImage.model_validate(item_dict))
            else:
                normalized.append(MessagePartUnknown(type=str(part_type or "unknown")))
        return normalized

    @field_validator("tool_calls", mode="before")
    @classmethod
    def validate_tool_calls(cls, value: object) -> list[object]:
        return value if isinstance(value, list) else []


class BaseProviderRequestSnapshot(IgnoreExtraModel):
    """三类 Host 请求快照的公共字段。"""

    model_info: ModelInfoSnapshot = Field(default_factory=ModelInfoSnapshot)
    api_provider: ApiProviderSnapshot = Field(default_factory=ApiProviderSnapshot)
    extra_params: ObjectFields = Field(default_factory=ObjectFields)

    @field_validator("extra_params", mode="before")
    @classmethod
    def validate_extra_params(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)


class ResponseRequestSnapshot(BaseProviderRequestSnapshot):
    """文本/多模态响应请求快照。"""

    request_kind: str = "response"
    message_list: list[MessageSnapshot] = Field(default_factory=list)
    tool_options: list[ToolOptionSnapshot] = Field(default_factory=list)
    temperature: int | float | None = None
    max_tokens: int | None = None
    response_format: ResponseFormatSnapshot | None = None

    @field_validator("message_list", "tool_options", mode="before")
    @classmethod
    def validate_list_fields(cls, value: object) -> list[object]:
        return value if isinstance(value, list) else []


class EmbeddingRequestSnapshot(BaseProviderRequestSnapshot):
    """Embedding 请求快照。"""

    request_kind: str = "embedding"
    embedding_input: str = ""


class AudioTranscriptionRequestSnapshot(BaseProviderRequestSnapshot):
    """音频转写请求快照。"""

    request_kind: str = "audio_transcription"
    audio_base64: str = ""
    max_tokens: int | None = None


class OpenAIInputTextBlock(HostDumpModel):
    """OpenAI Responses input_text。"""

    type: Literal["input_text"] = "input_text"
    text: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "text": self.text}


class OpenAIOutputTextBlock(HostDumpModel):
    """OpenAI Responses output_text 回放块，仅用于真实服务端 output item。"""

    type: Literal["output_text"] = "output_text"
    text: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "text": self.text, "annotations": []}


class OpenAIInputImageBlock(HostDumpModel):
    """OpenAI Responses input_image。"""

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "image_url": self.image_url, "detail": self.detail}


OpenAIUserContentBlock: TypeAlias = OpenAIInputTextBlock | OpenAIInputImageBlock
OpenAIOutputMessageContentBlock: TypeAlias = OpenAIOutputTextBlock


class OpenAIInputMessage(HostDumpModel):
    """OpenAI Responses user/system input message。"""

    role: Literal["system", "user"]
    content: list[OpenAIUserContentBlock]

    def to_sdk_param(self) -> JsonObject:
        return {"role": self.role, "content": [part.to_sdk_param() for part in self.content]}


class OpenAIEasyInputMessage(HostDumpModel):
    """OpenAI Responses EasyInputMessage，用于普通 assistant 文本历史。"""

    role: Literal["assistant"] = "assistant"
    content: str

    def to_sdk_param(self) -> JsonObject:
        return {"role": self.role, "content": self.content}


class OpenAIResponseOutputMessageItem(HostDumpModel):
    """OpenAI Responses 服务端 output message item 回放。"""

    id: str
    content: list[OpenAIOutputMessageContentBlock]

    def to_sdk_param(self) -> JsonObject:
        return {
            "type": "message",
            "id": self.id,
            "role": "assistant",
            "status": "completed",
            "content": [part.to_sdk_param() for part in self.content],
        }


class OpenAIFunctionCallInputItem(HostDumpModel):
    """OpenAI Responses function_call 历史。"""

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None

    def to_sdk_param(self) -> JsonObject:
        payload: JsonObject = {
            "type": self.type,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id is not None:
            payload["id"] = self.id
        if self.status is not None:
            payload["status"] = self.status
        return payload


class OpenAIFunctionCallOutputItem(HostDumpModel):
    """OpenAI Responses function_call_output 历史。"""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "call_id": self.call_id, "output": self.output}


OpenAIResponseInputItem: TypeAlias = (
    OpenAIInputMessage
    | OpenAIEasyInputMessage
    | OpenAIResponseOutputMessageItem
    | OpenAIFunctionCallInputItem
    | OpenAIFunctionCallOutputItem
)


class OpenAIResponsesTool(HostDumpModel):
    """OpenAI Responses function tool。"""

    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: ObjectFields = Field(default_factory=_default_tool_parameters)
    strict: bool = False

    def to_sdk_param(self) -> JsonObject:
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_plain_dict(),
            "strict": self.strict,
        }


class OpenAITextFormatConfig(HostDumpModel):
    """OpenAI Responses text.format。"""

    type: Literal["text", "json_object", "json_schema"]
    name: str | None = None
    description: str | None = None
    schema_payload: ObjectFields | None = Field(default=None, alias="schema")
    strict: bool | None = None

    def to_sdk_param(self) -> JsonObject:
        if self.type == "text":
            return {"type": "text"}
        if self.type == "json_object":
            return {"type": "json_object"}
        if self.name is None or not self.name.strip():
            raise ValueError("OpenAI Responses text.format.name 必须是非空字符串")
        if self.schema_payload is None:
            raise ValueError("OpenAI Responses text.format.schema 不能为空")
        payload: JsonObject = {
            "type": "json_schema",
            "name": self.name.strip(),
            "schema": self.schema_payload.to_plain_dict(),
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.strict is not None:
            payload["strict"] = self.strict
        return payload


class OpenAITextConfig(HostDumpModel):
    """OpenAI Responses text 配置。"""

    format: OpenAITextFormatConfig | None = None
    verbosity: OpenAITextVerbosity | None = None

    def to_sdk_param(self) -> JsonObject:
        result: JsonObject = {}
        if self.format is not None:
            result["format"] = self.format.to_sdk_param()
        if self.verbosity is not None:
            result["verbosity"] = self.verbosity
        return result


class OpenAIResponsesRequest(HostDumpModel):
    """OpenAI Responses create 请求。"""

    model: str
    input: list[OpenAIResponseInputItem]
    max_output_tokens: int | None = None
    temperature: float | None = None
    tools: list[OpenAIResponsesTool] = Field(default_factory=list)
    text: OpenAITextConfig | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: JsonObject = Field(default_factory=dict)
    extra_body: JsonObject = Field(default_factory=dict)
    direct_params: JsonObject = Field(default_factory=dict)

    def input_params(self) -> list[JsonObject]:
        return [item.to_sdk_param() for item in self.input]

    def tool_params(self) -> list[JsonObject]:
        return [tool.to_sdk_param() for tool in self.tools]

    def text_param(self) -> JsonObject | None:
        return self.text.to_sdk_param() if self.text is not None else None


class GenericUsageSnapshot(IgnoreExtraModel):
    """多 SDK usage 字段的宽松读取模型。"""

    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    input_tokens_details: ObjectFields = Field(default_factory=ObjectFields)
    prompt_tokens_details: ObjectFields = Field(default_factory=ObjectFields)

    @field_validator("input_tokens_details", "prompt_tokens_details", mode="before")
    @classmethod
    def validate_details(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)


class OpenAIRawData(HostDumpModel):
    """OpenAI raw_data 摘要。"""

    id: str | None = None
    model: str = ""
    status: str = ""
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)


class OpenAIResponseOutputContentBlock(IgnoreExtraModel):
    """OpenAI Responses output content/summary block。"""

    type: str = ""
    text: str | None = None


class OpenAIResponseOutputItem(IgnoreExtraModel):
    """OpenAI Responses output item 摘要。"""

    type: str = ""
    call_id: str | None = None
    id: str | None = None
    name: str = ""
    arguments: str = ""
    status: str | None = None
    content: list[OpenAIResponseOutputContentBlock] = Field(default_factory=list)
    summary: list[OpenAIResponseOutputContentBlock] = Field(default_factory=list)

    @field_validator("content", "summary", mode="before")
    @classmethod
    def validate_blocks(cls, value: object) -> list[object]:
        return value if isinstance(value, list) else []


class OpenAIResponseSnapshot(IgnoreExtraModel):
    """OpenAI Responses 响应摘要。"""

    id: str | None = None
    model: str = ""
    status: str = ""
    output_text: str = ""
    output: list[OpenAIResponseOutputItem] = Field(default_factory=list)
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)


class AnthropicTextBlock(HostDumpModel):
    """Anthropic text block。"""

    type: Literal["text"] = "text"
    text: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "text": self.text}


class AnthropicImageSource(HostDumpModel):
    """Anthropic base64 image source。"""

    type: Literal["base64"] = "base64"
    media_type: AnthropicImageMediaType
    data: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "media_type": self.media_type, "data": self.data}


class AnthropicImageBlock(HostDumpModel):
    """Anthropic image block。"""

    type: Literal["image"] = "image"
    source: AnthropicImageSource

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "source": self.source.to_sdk_param()}


class AnthropicToolUseBlock(HostDumpModel):
    """Anthropic assistant tool_use block。"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: JsonObject = Field(default_factory=dict)

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "id": self.id, "name": self.name, "input": self.input}


class AnthropicToolResultBlock(HostDumpModel):
    """Anthropic user tool_result block。"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str

    def to_sdk_param(self) -> JsonObject:
        return {"type": self.type, "tool_use_id": self.tool_use_id, "content": self.content}


AnthropicContentBlock: TypeAlias = (
    AnthropicTextBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock
)


class AnthropicMessage(HostDumpModel):
    """Anthropic Messages message。"""

    role: Literal["user", "assistant"]
    content: list[AnthropicContentBlock]

    def to_sdk_param(self) -> JsonObject:
        return {"role": self.role, "content": [block.to_sdk_param() for block in self.content]}


class AnthropicTool(HostDumpModel):
    """Anthropic tool definition。"""

    name: str
    description: str = ""
    input_schema: ObjectFields = Field(default_factory=_default_tool_parameters)

    def to_sdk_param(self) -> JsonObject:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema.to_plain_dict()}


class AnthropicMessagesRequest(HostDumpModel):
    """Anthropic messages.create 请求。"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | None = None
    temperature: float | None = None
    tools: list[AnthropicTool] = Field(default_factory=list)
    tool_choice: ObjectFields | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: JsonObject = Field(default_factory=dict)
    extra_body: JsonObject = Field(default_factory=dict)
    direct_params: JsonObject = Field(default_factory=dict)

    def message_params(self) -> list[JsonObject]:
        return [message.to_sdk_param() for message in self.messages]

    def tool_params(self) -> list[JsonObject]:
        return [tool.to_sdk_param() for tool in self.tools]


class AnthropicRawData(HostDumpModel):
    """Anthropic raw_data 摘要。"""

    id: str | None = None
    model: str = ""
    stop_reason: str = ""
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)


class AnthropicResponseContentBlock(IgnoreExtraModel):
    """Anthropic response content block。"""

    type: str = ""
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: JsonObject = Field(default_factory=dict)

    @field_validator("input", mode="before")
    @classmethod
    def validate_input(cls, value: object) -> JsonObject:
        return ObjectFields.from_unknown(value).to_plain_dict()


class AnthropicResponseSnapshot(IgnoreExtraModel):
    """Anthropic Messages 响应摘要。"""

    id: str | None = None
    model: str = ""
    stop_reason: str = ""
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)
    content: list[AnthropicResponseContentBlock] = Field(default_factory=list)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: object) -> list[object]:
        return value if isinstance(value, list) else []


class SdkDumpAdapter:
    """把 SDK/Pydantic 响应规整为普通 JSON-like 结构。"""

    @staticmethod
    def to_plain(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, ModelDumpable):
            return value.model_dump(mode="python")
        if isinstance(value, Mapping):
            return {str(key): SdkDumpAdapter.to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [SdkDumpAdapter.to_plain(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return SdkDumpAdapter.to_plain(to_dict())
        return str(value)

    @staticmethod
    def to_plain_dict(value: object) -> JsonObject:
        plain = SdkDumpAdapter.to_plain(value)
        if isinstance(plain, Mapping):
            return _DICT_ADAPTER.validate_python(dict(plain))
        raise TypeError(f"不支持的 SDK 响应对象类型: {type(value).__name__}")

    @staticmethod
    def to_plain_list(value: object) -> JsonArray:
        plain = SdkDumpAdapter.to_plain(value)
        if isinstance(plain, list):
            return _LIST_ADAPTER.validate_python(plain)
        raise TypeError(f"不支持的 SDK 列表对象类型: {type(value).__name__}")
