from ...core.common import RuntimeOptionsView
from ...schemas import ProviderToolCall, ToolCallSnapshot, ToolOptionSnapshot
from ..chat_completions_family import tools as family_tools
from ...core.json_types import JsonValue


def convert_history_tool_call(
    tool_call: ToolCallSnapshot,
    *,
    options: RuntimeOptionsView,
    index: int = 1,
) -> dict[str, JsonValue] | None:
    """把历史工具调用转给 Chat Completions family 标准实现。"""
    return family_tools.convert_history_tool_call(
        tool_call,
        options=options,
        fallback_prefix="mimo_history_tool",
        index=index,
    )


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[dict[str, JsonValue]]:
    """把工具定义转给 Chat Completions family 标准实现。"""
    return family_tools.convert_tools(tool_options)


def extract_tool_calls(raw_tool_calls: object, *, options: RuntimeOptionsView) -> list[ProviderToolCall]:
    """提取 Mimo 工具调用并保留 Provider 命名空间。"""
    return family_tools.extract_tool_calls(
        raw_tool_calls,
        options=options,
        provider="xiaomi_mimo",
        namespace="xiaomi_mimo",
        fallback_prefix="mimo_tool",
    )
