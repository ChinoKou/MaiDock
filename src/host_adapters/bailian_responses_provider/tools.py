from ...core.common import RuntimeOptionsView
from ...schemas.host_snapshots import ToolOptionSnapshot
from ...schemas.provider_contracts import ProviderToolCall
from ...schemas.responses_compat import OpenAIResponseOutputItem, OpenAIResponsesTool
from ..responses_family.tools import extract_tool_calls as _family_extract_tool_calls


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
    """转换 Host 工具定义；百炼 Responses 未声明 strict 字段，按协议省略。"""

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
                strict=None,
            )
        )
    return tools


def extract_tool_calls(
    output: list[OpenAIResponseOutputItem],
    *,
    options: RuntimeOptionsView,
) -> list[ProviderToolCall]:
    return _family_extract_tool_calls(output, options=options, raw_provider="bailian_responses")
