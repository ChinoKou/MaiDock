from ...core.common import ProviderRuntimeOptions
from ...schemas import ProviderToolCall, ToolCallSnapshot, ToolOptionSnapshot
from ..chat_completions_family import tools as family_tools


def convert_history_tool_call(
    tool_call: ToolCallSnapshot,
    *,
    options: ProviderRuntimeOptions,
    index: int = 1,
) -> dict | None:
    """把历史工具调用转给 Chat Completions family 标准实现。"""
    return family_tools.convert_history_tool_call(
        tool_call,
        options=options,
        fallback_prefix="siliconflow_history_tool",
        index=index,
    )


def convert_tools(tool_options: list[ToolOptionSnapshot]) -> list[dict]:
    """把工具定义转给 Chat Completions family 标准实现。"""
    return family_tools.convert_tools(tool_options)


def extract_tool_calls(raw_tool_calls: object, *, options: ProviderRuntimeOptions) -> list[ProviderToolCall]:
    """提取 SiliconFlow 工具调用并保留 Provider 命名空间。"""
    return family_tools.extract_tool_calls(
        raw_tool_calls,
        options=options,
        provider="siliconflow",
        namespace="siliconflow",
        fallback_prefix="siliconflow_tool",
    )
