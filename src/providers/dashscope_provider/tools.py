import json
from dataclasses import dataclass

from ...core.common import ProviderRuntimeOptions, tool_arguments_to_json
from ...core.json_types import (
    mapping_to_json_object,
    json_list_or_none,
    json_mapping_or_none,
)
from ...core.parsing import ToolArgumentParseMode
from ...schemas import (
    ProviderFunctionCall,
    ProviderToolCall,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)
from ..common.tools import normalize_tool_arguments_value, resolve_tool_call_id


@dataclass(slots=True)
class DashScopeToolCallChunk:
    call_id: str | None = None
    name: str = ""
    arguments: str = ""

    def merge_arguments(self, arguments: str) -> None:
        if not arguments:
            return
        if arguments.startswith(self.arguments):
            self.arguments = arguments
        else:
            self.arguments += arguments

    def to_tool_call(self, index: int, parse_mode: ToolArgumentParseMode) -> ProviderToolCall:
        call_id = resolve_tool_call_id(self.call_id, fallback_prefix="dashscope_tool", index=index)
        return ProviderToolCall(
            id=call_id,
            function=ProviderFunctionCall(
                name=self.name,
                arguments=normalize_tool_arguments_value(self.arguments, parse_mode),
            ),
            extra_content={
                "provider": "dashscope",
                "dashscope": {
                    "generated_call_id": self.call_id is None,
                    "raw_arguments": self.arguments,
                },
            },
        )


def convert_history_tool_call(tool_call: ToolCallSnapshot, *, options: ProviderRuntimeOptions) -> dict | None:
    name = tool_call.function.name
    if not name:
        return None
    call_id = tool_call.resolved_call_id() or resolve_tool_call_id(
        None, fallback_prefix="dashscope_history_tool", index=1
    )
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": tool_arguments_to_json(tool_call.function.arguments, options.tool_argument_parse_mode),
        },
    }


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[dict]:
    tools: list[dict] = []
    for tool in tool_options:
        function = tool.function_definition()
        if function.name is None or not function.name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters.to_plain_dict(),
                },
            }
        )
    return tools


def extract_tool_calls(raw_tool_calls: object, *, options: ProviderRuntimeOptions) -> list[ProviderToolCall]:
    tool_call_items = json_list_or_none(raw_tool_calls)
    if tool_call_items is None:
        return []
    tool_calls: list[ProviderToolCall] = []
    for index, item in enumerate(tool_call_items, start=1):
        tool_call = json_mapping_or_none(item)
        if tool_call is None:
            continue
        function_mapping = json_mapping_or_none(tool_call.get("function"))
        if function_mapping is None:
            continue
        name = function_mapping.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = function_mapping.get("arguments")
        argument_mapping = json_mapping_or_none(arguments)
        raw_call_id = tool_call.get("id")
        call_id = raw_call_id if isinstance(raw_call_id, str) else None
        tool_calls.append(
            ProviderToolCall(
                id=resolve_tool_call_id(
                    call_id,
                    fallback_prefix="dashscope_tool",
                    index=index,
                ),
                function=ProviderFunctionCall(
                    name=name,
                    arguments=normalize_tool_arguments_value(arguments, options.tool_argument_parse_mode),
                ),
                extra_content={
                    "provider": "dashscope",
                    "dashscope": {
                        "raw_arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(mapping_to_json_object(argument_mapping), ensure_ascii=False)
                        if argument_mapping is not None
                        else None,
                    },
                },
            )
        )
    return tool_calls
