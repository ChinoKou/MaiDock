from typing import Literal

from pydantic import Field, field_validator

from ..core.json_types import is_json_list
from .base import (
    HostDumpModel,
    IgnoreExtraModel,
    ObjectFields,
    OpenAITextVerbosity,
    default_tool_parameters,
)
from .usage import GenericUsageSnapshot


def default_openai_tool_parameters() -> ObjectFields:
    return default_tool_parameters()


class OpenAIInputTextBlock(HostDumpModel):
    """OpenAI Responses input_text。"""

    type: Literal["input_text"] = "input_text"
    text: str

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "text": self.text}


class OpenAIOutputTextBlock(HostDumpModel):
    """OpenAI Responses output_text 回放块，仅用于真实服务端 output item。"""

    type: Literal["output_text"] = "output_text"
    text: str

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "text": self.text, "annotations": []}


class OpenAIInputImageBlock(HostDumpModel):
    """OpenAI Responses input_image。"""

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "image_url": self.image_url, "detail": self.detail}


type OpenAIUserContentBlock = OpenAIInputTextBlock | OpenAIInputImageBlock
type OpenAIOutputMessageContentBlock = OpenAIOutputTextBlock


class OpenAIInputMessage(HostDumpModel):
    """OpenAI Responses user/system input message。"""

    role: Literal["system", "user"]
    content: list[OpenAIUserContentBlock]

    def to_sdk_param(self) -> dict:
        return {
            "role": self.role,
            "content": [part.to_sdk_param() for part in self.content],
        }


class OpenAIEasyInputMessage(HostDumpModel):
    """OpenAI Responses EasyInputMessage，用于普通 assistant 文本历史。"""

    role: Literal["assistant"] = "assistant"
    content: str

    def to_sdk_param(self) -> dict:
        return {"role": self.role, "content": self.content}


class OpenAIResponseOutputMessageItem(HostDumpModel):
    """OpenAI Responses 服务端 output message item 回放。"""

    id: str
    content: list[OpenAIOutputMessageContentBlock]

    def to_sdk_param(self) -> dict:
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

    def to_sdk_param(self) -> dict:
        payload: dict = {
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

    def to_sdk_param(self) -> dict:
        return {"type": self.type, "call_id": self.call_id, "output": self.output}


type OpenAIResponseInputItem = (
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
    parameters: ObjectFields = Field(default_factory=default_openai_tool_parameters)
    strict: bool = False

    def to_sdk_param(self) -> dict:
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

    def to_sdk_param(self) -> dict:
        if self.type == "text":
            return {"type": "text"}
        if self.type == "json_object":
            return {"type": "json_object"}
        if self.name is None or not self.name.strip():
            raise ValueError("OpenAI Responses text.format.name 必须是非空字符串")
        if self.schema_payload is None:
            raise ValueError("OpenAI Responses text.format.schema 不能为空")
        payload: dict = {
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

    def to_sdk_param(self) -> dict:
        result: dict = {}
        if self.format is not None:
            result["format"] = self.format.to_sdk_param()
        if self.verbosity is not None:
            result["verbosity"] = self.verbosity
        return result


def _empty_openai_responses_tool_list() -> list[OpenAIResponsesTool]:
    return []


def _empty_openai_response_output_item_list() -> list["OpenAIResponseOutputItem"]:
    return []


class OpenAIResponsesRequest(HostDumpModel):
    """OpenAI Responses create 请求。"""

    model: str
    input: list[OpenAIResponseInputItem]
    tools: list[OpenAIResponsesTool] = Field(default_factory=_empty_openai_responses_tool_list)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: dict = Field(default_factory=dict)
    normalized: object | None = None

    def input_params(self) -> list[dict]:
        return [item.to_sdk_param() for item in self.input]

    def tool_params(self) -> list[dict]:
        return [tool.to_sdk_param() for tool in self.tools]


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


def _empty_openai_response_output_content_block_list() -> list[OpenAIResponseOutputContentBlock]:
    return []


class OpenAIResponseOutputItem(IgnoreExtraModel):
    """OpenAI Responses output item 摘要。"""

    type: str = ""
    call_id: str | None = None
    id: str | None = None
    name: str = ""
    arguments: str = ""
    status: str | None = None
    content: list[OpenAIResponseOutputContentBlock] = Field(
        default_factory=_empty_openai_response_output_content_block_list
    )
    summary: list[OpenAIResponseOutputContentBlock] = Field(
        default_factory=_empty_openai_response_output_content_block_list
    )

    @field_validator("content", "summary", mode="before")
    @classmethod
    def validate_blocks(cls, value: object) -> list:
        return value if is_json_list(value) else []


class OpenAIResponseSnapshot(IgnoreExtraModel):
    """OpenAI Responses 响应摘要。"""

    id: str | None = None
    model: str = ""
    status: str = ""
    output_text: str = ""
    output: list[OpenAIResponseOutputItem] = Field(default_factory=_empty_openai_response_output_item_list)
    usage: GenericUsageSnapshot = Field(default_factory=GenericUsageSnapshot)
