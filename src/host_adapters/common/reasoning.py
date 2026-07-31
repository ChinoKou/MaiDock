from ...core.common import RuntimeOptionsView
from ...core.parsing import extract_xml_tool_calls, merge_native_or_text_reasoning
from ...schemas import ProviderToolCall

_TOOL_CALL_SOURCE_EXTRA_KEY = "tool_call_source"
_TOOL_CALL_SOURCE_REASONING = "reasoning"
_TOOL_CALL_SOURCE_RESPONSE = "response"


def _mark_tool_call_source(
    tool_calls: list[ProviderToolCall],
    source: str,
) -> None:
    for tool_call in tool_calls:
        tool_call.extra_content[_TOOL_CALL_SOURCE_EXTRA_KEY] = source


def merge_reasoning_and_xml_tool_fallback(
    *,
    content: str | None,
    native_reasoning: str | None,
    tool_calls: list[ProviderToolCall],
    options: RuntimeOptionsView,
) -> tuple[str | None, str | None]:
    reasoning_content, final_content = merge_native_or_text_reasoning(
        content=content,
        native_reasoning=native_reasoning,
        parse_mode=options.reasoning_parse_mode,
    )
    if tool_calls:
        source = (
            _TOOL_CALL_SOURCE_REASONING
            if reasoning_content and reasoning_content.strip()
            else _TOOL_CALL_SOURCE_RESPONSE
        )
        _mark_tool_call_source(tool_calls, source)
        return reasoning_content, final_content

    reasoning_content, reasoning_tool_calls = extract_xml_tool_calls(
        reasoning_content,
        options.tool_argument_parse_mode,
    )
    if reasoning_tool_calls:
        _mark_tool_call_source(reasoning_tool_calls, _TOOL_CALL_SOURCE_REASONING)
        tool_calls.extend(reasoning_tool_calls)
    final_content, content_tool_calls = extract_xml_tool_calls(
        final_content,
        options.tool_argument_parse_mode,
    )
    if content_tool_calls:
        _mark_tool_call_source(content_tool_calls, _TOOL_CALL_SOURCE_RESPONSE)
        tool_calls.extend(content_tool_calls)
    return reasoning_content, final_content
