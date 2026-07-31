from ...core.common import RuntimeOptionsView
from ...schemas.host_snapshots import ToolOptionSnapshot
from ...schemas.provider_contracts import ProviderFunctionCall, ProviderToolCall
from ...schemas.responses_compat import OpenAIResponseOutputItem, OpenAIResponsesTool
from ..common.tools import normalize_tool_arguments_value, resolve_tool_call_id_from_seed


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
    tools: list[OpenAIResponsesTool] = []
    for tool in tool_options:
        function = tool.function_definition()
        if function.name is None or not function.name:
            continue
        tools.append(
            OpenAIResponsesTool(
                name=function.name,
                description=function.description,
                parameters=function.parameters,
                strict=False,
            )
        )
    return tools


def extract_tool_calls(
    output: list[OpenAIResponseOutputItem],
    *,
    options: RuntimeOptionsView,
    raw_provider: str,
) -> list[ProviderToolCall]:
    tool_calls: list[ProviderToolCall] = []
    for item in output:
        if item.type != "function_call" or not item.name:
            continue
        call_id = resolve_tool_call_id_from_seed(item.call_id or item.id, fallback_seed=item.name)
        tool_calls.append(
            ProviderToolCall(
                id=call_id,
                function=ProviderFunctionCall(
                    name=item.name,
                    arguments=normalize_tool_arguments_value(item.arguments, options.tool_argument_parse_mode),
                ),
                extra_content={
                    "provider": raw_provider,
                    "openai_responses": {
                        "item_id": item.id,
                        "status": item.status,
                        "raw_arguments": item.arguments,
                        "generated_call_id": item.call_id is None and item.id is not None,
                    },
                },
            )
        )
    return tool_calls
