from typing import Literal

from pydantic import Field, field_validator

from ..core.json_types import is_json_list, is_json_mapping, mapping_to_json_object
from .base import (
    IgnoreExtraModel,
    ObjectFields,
    default_tool_parameters,
    normalize_tool_parameters,
)


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


type MessagePart = MessagePartText | MessagePartImage | MessagePartUnknown


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


class ToolFunctionSnapshot(IgnoreExtraModel):
    """工具函数定义。"""

    name: str | None = None
    description: str = ""
    parameters: ObjectFields = Field(default_factory=default_tool_parameters)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> ObjectFields:
        return normalize_tool_parameters(value)


class ToolOptionSnapshot(IgnoreExtraModel):
    """Host 快照中的工具定义。"""

    type: str | None = None
    function: ToolFunctionSnapshot | None = None
    name: str | None = None
    description: str = ""
    parameters: ObjectFields = Field(default_factory=default_tool_parameters)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: object) -> ObjectFields:
        return normalize_tool_parameters(value)

    def function_definition(self) -> ToolFunctionSnapshot:
        if self.type == "function" and self.function is not None:
            return self.function
        return ToolFunctionSnapshot(name=self.name, description=self.description, parameters=self.parameters)


class ResponseFormatSchemaSnapshot(IgnoreExtraModel):
    """JSON schema 响应格式。"""

    name: str | None = None
    description: str | None = None
    schema_: ObjectFields | None = Field(default=None, alias="schema")
    strict: bool | None = None

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
        if is_json_mapping(value) and any(key in value for key in ("name", "description", "schema", "strict")):
            return ResponseFormatSchemaSnapshot.model_validate(mapping_to_json_object(value))
        return ObjectFields.from_unknown(value)


def _empty_message_part_list() -> list[MessagePart]:
    return []


def _empty_tool_call_snapshot_list() -> list[ToolCallSnapshot]:
    return []


class MessageSnapshot(IgnoreExtraModel):
    """Host 序列化后的消息。"""

    role: str = ""
    parts: list[MessagePart] = Field(default_factory=_empty_message_part_list)
    tool_calls: list[ToolCallSnapshot] = Field(default_factory=_empty_tool_call_snapshot_list)
    tool_call_id: str | None = None
    tool_name: str | None = None

    @field_validator("parts", mode="before")
    @classmethod
    def validate_parts(cls, value: object) -> list[MessagePart]:
        if not is_json_list(value):
            return []
        normalized: list[MessagePart] = []
        for item in value:
            if not is_json_mapping(item):
                continue
            item_dict = mapping_to_json_object(item)
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
    def validate_tool_calls(cls, value: object) -> list:
        return value if is_json_list(value) else []


class BaseProviderRequestSnapshot(IgnoreExtraModel):
    """三类 Host 请求快照的公共字段。"""

    model_info: ModelInfoSnapshot = Field(default_factory=ModelInfoSnapshot)
    api_provider: ApiProviderSnapshot = Field(default_factory=ApiProviderSnapshot)


def _empty_message_snapshot_list() -> list[MessageSnapshot]:
    return []


def _empty_tool_option_snapshot_list() -> list[ToolOptionSnapshot]:
    return []


class ResponseRequestSnapshot(BaseProviderRequestSnapshot):
    """文本/多模态响应请求快照。"""

    request_kind: str = "response"
    message_list: list[MessageSnapshot] = Field(default_factory=_empty_message_snapshot_list)
    tool_options: list[ToolOptionSnapshot] = Field(default_factory=_empty_tool_option_snapshot_list)
    temperature: int | float | None = None
    max_tokens: int | None = None
    response_format: ResponseFormatSnapshot | None = None

    @field_validator("message_list", "tool_options", mode="before")
    @classmethod
    def validate_list_fields(cls, value: object) -> list:
        return value if is_json_list(value) else []


class EmbeddingRequestSnapshot(BaseProviderRequestSnapshot):
    """Embedding 请求快照。"""

    request_kind: str = "embedding"
    embedding_input: str = ""
    dimensions: int | None = None


class AudioTranscriptionRequestSnapshot(BaseProviderRequestSnapshot):
    """音频转写请求快照。"""

    request_kind: str = "audio_transcription"
    audio_base64: str = ""
    max_tokens: int | None = None
