from ...core.common import RuntimeOptionsView, message_text
from ...core.parsing import normalize_arguments
from ...i18n import runtime_subject, translate
from ...schemas import (
    AnthropicMessage,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolUseBlock,
    MessageSnapshot,
    ObjectFields,
    ToolOptionSnapshot,
)
from ...core.json_types import JsonValue


def convert_assistant_tool_calls(
    message: MessageSnapshot,
    *,
    options: RuntimeOptionsView,
) -> list[AnthropicToolUseBlock]:
    blocks: list[AnthropicToolUseBlock] = []
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        if not name:
            continue
        tool_use_id = tool_call.resolved_call_id()
        if not tool_use_id:
            raise ValueError(
                translate(
                    "runtime.error.required",
                    subject=f"Anthropic Messages {runtime_subject('historical_tool_call')} {name}",
                    field="tool_use id",
                )
            )
        blocks.append(
            AnthropicToolUseBlock(
                id=tool_use_id,
                name=name,
                input=normalize_arguments(tool_call.function.arguments, options.tool_argument_parse_mode),
            )
        )
    return blocks


def orphan_tool_result_message(message: MessageSnapshot) -> AnthropicMessage:
    call_id = (message.tool_call_id or "").strip()
    tool_name = (message.tool_name or "tool").strip() or "tool"
    label = f"{tool_name} ({call_id or 'unknown'})"
    return AnthropicMessage(
        role="user",
        content=[
            AnthropicTextBlock(
                text=f"工具调用结果（缺少可回放的 assistant tool_use）：{label}: {message_text(message)}"
            )
        ],
    )


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[AnthropicTool]:
    tools: list[AnthropicTool] = []
    for tool in tool_options:
        function = tool.function_definition()
        if function.name is None or not function.name:
            continue
        tools.append(
            AnthropicTool(
                name=function.name,
                description=function.description,
                input_schema=function.parameters,
            )
        )
    return tools


def build_default_tool_choice(tools: list[AnthropicTool], direct_params: dict[str, JsonValue]) -> ObjectFields | None:
    if not tools or "tool_choice" in direct_params:
        return None
    return ObjectFields(fields={"type": "any", "disable_parallel_tool_use": True})
