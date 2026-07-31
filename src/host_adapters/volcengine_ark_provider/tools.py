from ...core.common import RuntimeOptionsView
from ...schemas.host_snapshots import ToolOptionSnapshot
from ...schemas.provider_contracts import ProviderToolCall
from ...schemas.responses_compat import OpenAIResponseOutputItem, OpenAIResponsesTool
from ..responses_family.tools import convert_tools as _family_convert_tools
from ..responses_family.tools import extract_tool_calls as _family_extract_tool_calls


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
    return _family_convert_tools(tool_options)


def extract_tool_calls(
    output: list[OpenAIResponseOutputItem],
    *,
    options: RuntimeOptionsView,
) -> list[ProviderToolCall]:
    return _family_extract_tool_calls(output, options=options, raw_provider="volcengine_ark_responses")
