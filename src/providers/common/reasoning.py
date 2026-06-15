from ...core.common import ProviderRuntimeOptions
from ...core.parsing import extract_xml_tool_calls, merge_native_or_text_reasoning
from ...schemas import ProviderToolCall


def merge_reasoning_and_xml_tool_fallback(
    *,
    content: str | None,
    native_reasoning: str | None,
    tool_calls: list[ProviderToolCall],
    options: ProviderRuntimeOptions,
) -> tuple[str | None, str | None]:
    reasoning_content, final_content = merge_native_or_text_reasoning(
        content=content,
        native_reasoning=native_reasoning,
        parse_mode=options.reasoning_parse_mode,
    )
    if not tool_calls:
        reasoning_content, reasoning_tool_calls = extract_xml_tool_calls(
            reasoning_content,
            options.tool_argument_parse_mode,
        )
        if reasoning_tool_calls:
            tool_calls.extend(reasoning_tool_calls)
        final_content, content_tool_calls = extract_xml_tool_calls(
            final_content,
            options.tool_argument_parse_mode,
        )
        if content_tool_calls:
            tool_calls.extend(content_tool_calls)
    return reasoning_content, final_content
