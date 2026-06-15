from typing import Literal

from pydantic import Field, field_validator

from ..core.json_types import is_json_list
from .base import (
    AnthropicImageMediaType,
    HostDumpModel,
    IgnoreExtraModel,
    ObjectFields,
    default_tool_parameters,
)
from .usage import GenericUsageSnapshot


class AnthropicTextBlock(HostDumpModel):
    """Anthropic text block。"""

    type: Literal["text"] = "text"
    text: str

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "text": self.text}


class AnthropicImageSource(HostDumpModel):
    """Anthropic base64 image source。"""

    type: Literal["base64"] = "base64"
    media_type: AnthropicImageMediaType
    data: str

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "media_type": self.media_type, "data": self.data}


class AnthropicImageBlock(HostDumpModel):
    """Anthropic image block。"""

    type: Literal["image"] = "image"
    source: AnthropicImageSource

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "source": self.source.to_sdk_param()}


class AnthropicToolUseBlock(HostDumpModel):
    """Anthropic assistant tool_use block。"""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict = Field(default_factory=dict)

    def to_sdk_param(self) -> dict:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


class AnthropicToolResultBlock(HostDumpModel):
    """Anthropic user tool_result block。"""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str

    def to_sdk_param(self) -> dict:
        return {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }


type AnthropicContentBlock = AnthropicTextBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock


class AnthropicMessage(HostDumpModel):
    """Anthropic Messages message。"""

    role: Literal["user", "assistant"]
    content: list[AnthropicContentBlock]

    def to_sdk_param(self) -> dict:
        return {
            "role": self.role,
            "content": [block.to_sdk_param() for block in self.content],
        }


class AnthropicTool(HostDumpModel):
    """Anthropic tool definition。"""

    name: str
    description: str = ""
    input_schema: ObjectFields = Field(default_factory=default_tool_parameters)

    def to_sdk_param(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.to_plain_dict(),
        }


def _empty_anthropic_tool_list() -> list[AnthropicTool]:
    return []


class AnthropicMessagesRequest(HostDumpModel):
    """Anthropic messages.create 请求。"""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    system: str | None = None
    temperature: float | None = None
    tools: list[AnthropicTool] = Field(default_factory=_empty_anthropic_tool_list)
    tool_choice: ObjectFields | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: dict = Field(default_factory=dict)
    normalized: object | None = None

    def message_params(self) -> list[dict]:
        return [message.to_sdk_param() for message in self.messages]

    def tool_params(self) -> list[dict]:
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
    input: dict = Field(default_factory=dict)

    @field_validator("input", mode="before")
    @classmethod
    def validate_input(cls, value: object) -> dict:
        return ObjectFields.from_unknown(value).to_plain_dict()


def _empty_anthropic_response_content_block_list() -> list[AnthropicResponseContentBlock]:
    return []


class AnthropicResponseSnapshot(IgnoreExtraModel):
    """Anthropic Messages 响应摘要。"""

    id: str | None = None
    model: str = ""
    stop_reason: str = ""
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)
    content: list[AnthropicResponseContentBlock] = Field(default_factory=_empty_anthropic_response_content_block_list)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: object) -> list:
        return value if is_json_list(value) else []
