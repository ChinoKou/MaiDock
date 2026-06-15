from ...core.common import ProviderRuntimeOptions
from ...core.parsing import fallback_tool_call_id, normalize_arguments
from ...schemas.provider_contracts import ProviderFunctionCall, ProviderToolCall
from ...schemas.responses_compat import OpenAIResponseOutputItem, OpenAIResponsesTool
from ...schemas.host_snapshots import ToolOptionSnapshot


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
    options: ProviderRuntimeOptions,
    raw_provider: str,
) -> list[ProviderToolCall]:
    tool_calls: list[ProviderToolCall] = []
    for item in output:
        if item.type != "function_call" or not item.name:
            continue
        call_id = (item.call_id or item.id or fallback_tool_call_id(item.name)).strip()
        if not call_id:
            call_id = fallback_tool_call_id(item.name)
        tool_calls.append(
            ProviderToolCall(
                id=call_id,
                function=ProviderFunctionCall(
                    name=item.name,
                    arguments=normalize_arguments(item.arguments, options.tool_argument_parse_mode),
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
