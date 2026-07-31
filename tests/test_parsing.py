import json
from dataclasses import dataclass

import pytest

from src.core.common import ProviderRuntimeOptions
from src.core.parsing import (
    ReasoningParseMode,
    ToolArgumentParseMode,
    arguments_to_json,
    extract_reasoning_and_content,
    extract_xml_tool_calls,
    fallback_tool_call_id,
    merge_native_or_text_reasoning,
    normalize_arguments,
    normalize_reasoning_parse_mode,
    normalize_tool_argument_parse_mode,
    parse_tool_arguments,
)
from src.host_adapters.common.reasoning import merge_reasoning_and_xml_tool_fallback
from src.schemas import ObjectFields, ProviderFunctionCall, ProviderToolCall


@dataclass(frozen=True)
class _FixedUuid:
    hex: str = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize(
    ("native_reasoning", "expected_source"),
    [("先思考", "reasoning"), (None, "response"), ("   ", "response")],
)
def test_reasoning_merge_marks_native_tool_call_source(
    native_reasoning: str | None,
    expected_source: str,
) -> None:
    tool_calls = [
        ProviderToolCall(
            id="call-1",
            function=ProviderFunctionCall(name="lookup", arguments={}),
            extra_content={"provider": "test"},
        )
    ]

    merge_reasoning_and_xml_tool_fallback(
        content="回答",
        native_reasoning=native_reasoning,
        tool_calls=tool_calls,
        options=ProviderRuntimeOptions(),
    )

    assert tool_calls[0].extra_content == {
        "provider": "test",
        "tool_call_source": expected_source,
    }


def test_reasoning_merge_marks_xml_tools_by_their_actual_region() -> None:
    tool_calls: list[ProviderToolCall] = []

    reasoning, content = merge_reasoning_and_xml_tool_fallback(
        content=(
            "<think><tool_call><function=analyze>{}</function></tool_call></think>"
            "正文<tool_call><function=answer>{}</function></tool_call>"
        ),
        native_reasoning=None,
        tool_calls=tool_calls,
        options=ProviderRuntimeOptions(),
    )

    assert reasoning is None
    assert content == "正文"
    assert [tool_call.function.name for tool_call in tool_calls] == ["analyze", "answer"]
    assert [tool_call.extra_content["tool_call_source"] for tool_call in tool_calls] == [
        "reasoning",
        "response",
    ]


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        pytest.param("strict", "strict", id="strict"),
        pytest.param("repair", "repair", id="repair"),
        pytest.param("double_decode", "double_decode", id="double-decode"),
        pytest.param("auto", "auto", id="auto"),
        pytest.param("unknown", "auto", id="unknown-falls-back-to-auto"),
        pytest.param(None, "auto", id="none-falls-back-to-auto"),
    ],
)
def test_normalize_tool_argument_parse_mode(
    raw_mode: str | None,
    expected: ToolArgumentParseMode,
) -> None:
    assert normalize_tool_argument_parse_mode(raw_mode) == expected


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        pytest.param("native", "native", id="native"),
        pytest.param("think_tag", "think_tag", id="think-tag"),
        pytest.param("none", "none", id="none"),
        pytest.param("auto", "auto", id="auto"),
        pytest.param("unknown", "auto", id="unknown-falls-back-to-auto"),
        pytest.param(None, "auto", id="none-falls-back-to-auto"),
    ],
)
def test_normalize_reasoning_parse_mode(
    raw_mode: str | None,
    expected: ReasoningParseMode,
) -> None:
    assert normalize_reasoning_parse_mode(raw_mode) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(ObjectFields(fields={"city": "杭州"}), {"city": "杭州"}, id="object-fields"),
        pytest.param(None, {}, id="none"),
        pytest.param('{"city":"杭州"}', {"city": "杭州"}, id="json-string"),
    ],
)
def test_normalize_arguments(
    value: ObjectFields | str | None,
    expected: dict,
) -> None:
    assert normalize_arguments(value) == expected


@pytest.mark.parametrize(
    ("raw_arguments", "parse_mode", "expected"),
    [
        pytest.param("", "strict", {}, id="empty"),
        pytest.param("  \n", "repair", {}, id="whitespace"),
        pytest.param('{"count":2}', "strict", {"count": 2}, id="strict-object"),
        pytest.param("{count: 2,}", "repair", {"count": 2}, id="repair-object"),
        pytest.param('"{\\"count\\":2}"', "auto", {"count": 2}, id="auto-double-decode"),
        pytest.param('"{\\"count\\":2}"', "double_decode", {"count": 2}, id="explicit-double-decode"),
    ],
)
def test_parse_tool_arguments_modes(
    raw_arguments: str,
    parse_mode: ToolArgumentParseMode,
    expected: dict,
) -> None:
    assert parse_tool_arguments(raw_arguments, parse_mode) == expected


@pytest.mark.parametrize(
    ("raw_arguments", "parse_mode"),
    [
        pytest.param("[1, 2]", "strict", id="strict-list"),
        pytest.param('"plain text"', "repair", id="repair-string"),
        pytest.param("null", "auto", id="auto-null"),
    ],
)
def test_parse_tool_arguments_rejects_non_objects(
    raw_arguments: str,
    parse_mode: ToolArgumentParseMode,
) -> None:
    with pytest.raises(ValueError, match=r"preview=<text:\d+>"):
        parse_tool_arguments(raw_arguments, parse_mode)


def test_parse_tool_arguments_redacts_invalid_sensitive_text() -> None:
    sensitive_arguments = '{"secret":"sk-sensitive"'

    with pytest.raises(ValueError) as exc_info:
        parse_tool_arguments(sensitive_arguments, "strict")

    message = str(exc_info.value)
    assert f"<text:{len(sensitive_arguments)}>" in message
    assert "sk-sensitive" not in message


@pytest.mark.parametrize(
    ("value", "parse_mode", "expected"),
    [
        pytest.param('{"city":"杭州"}', "strict", {"city": "杭州"}, id="string"),
        pytest.param(ObjectFields(fields={"city": "杭州"}), "auto", {"city": "杭州"}, id="object-fields"),
        pytest.param(None, "auto", {}, id="none"),
    ],
)
def test_arguments_to_json_round_trip(
    value: ObjectFields | str | None,
    parse_mode: ToolArgumentParseMode,
    expected: dict,
) -> None:
    encoded = arguments_to_json(value, parse_mode)

    assert json.loads(encoded) == expected
    assert "\\u676d" not in encoded


@pytest.mark.parametrize(
    ("prefix", "max_length", "expected"),
    [
        pytest.param("", 96, "tool_call_0123456789abcdef0123456789abcdef", id="empty-prefix"),
        pytest.param("  call  ", 96, "call_0123456789abcdef0123456789abcdef", id="trimmed-prefix"),
        pytest.param("x" * 100, 40, "xxxxxxx_0123456789abcdef0123456789abcdef", id="truncated-prefix"),
    ],
)
def test_fallback_tool_call_id_has_stable_shape(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    max_length: int,
    expected: str,
) -> None:
    monkeypatch.setattr("src.core.parsing.uuid4", _FixedUuid)

    assert fallback_tool_call_id(prefix, max_length=max_length) == expected


@pytest.mark.parametrize(
    ("content", "parse_mode", "expected"),
    [
        pytest.param("<think> 先想 </think> 最终答案 ", "auto", ("先想", "最终答案"), id="closed"),
        pytest.param("<think>   </think>answer", "think_tag", (None, "answer"), id="empty-thinking"),
        pytest.param("<think>unfinished", "auto", ("unfinished", None), id="unclosed"),
        pytest.param("<think>   ", "auto", (None, None), id="unclosed-whitespace"),
        pytest.param("plain answer", "auto", (None, "plain answer"), id="plain"),
        pytest.param("   ", "auto", (None, None), id="whitespace"),
        pytest.param("", "auto", (None, ""), id="empty"),
        pytest.param("<think>keep</think>answer", "native", (None, "<think>keep</think>answer"), id="native"),
        pytest.param("<think>keep</think>answer", "none", (None, "<think>keep</think>answer"), id="none"),
    ],
)
def test_extract_reasoning_and_content(
    content: str,
    parse_mode: ReasoningParseMode,
    expected: tuple[str | None, str | None],
) -> None:
    assert extract_reasoning_and_content(content, parse_mode) == expected


@pytest.mark.parametrize(
    ("content", "native_reasoning", "parse_mode", "expected"),
    [
        pytest.param(
            "<think>text</think>answer", "native", "auto", ("native", "<think>text</think>answer"), id="native-priority"
        ),
        pytest.param(
            "<think>text</think>answer",
            "native",
            "none",
            (None, "<think>text</think>answer"),
            id="none-discards-native",
        ),
        pytest.param("<think>text</think>answer", None, "auto", ("text", "answer"), id="text-fallback"),
        pytest.param(None, None, "auto", (None, None), id="no-content"),
    ],
)
def test_merge_native_or_text_reasoning(
    content: str | None,
    native_reasoning: str | None,
    parse_mode: ReasoningParseMode,
    expected: tuple[str | None, str | None],
) -> None:
    assert (
        merge_native_or_text_reasoning(
            content=content,
            native_reasoning=native_reasoning,
            parse_mode=parse_mode,
        )
        == expected
    )


def test_extract_xml_tool_calls_parses_multiple_calls_and_preserves_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.core.parsing.uuid4", _FixedUuid)
    raw_text = """before
<tool_call><function=lookup>
<parameter=empty> </parameter>
<parameter=enabled>TRUE</parameter>
<parameter=disabled>false</parameter>
<parameter=missing>null</parameter>
<parameter=also_missing>None</parameter>
<parameter=config>{enabled: true}</parameter>
<parameter=items>[1, 2]</parameter>
<parameter=duplicate>first</parameter>
<parameter=duplicate>second</parameter>
</function></tool_call>
middle
<TOOL_CALL><FUNCTION=notify>{"text":"完成"}</FUNCTION></TOOL_CALL>
after"""

    content, tool_calls = extract_xml_tool_calls(raw_text)

    assert content == "before\n\nmiddle\n\nafter"
    assert [tool_call.id for tool_call in tool_calls] == [
        "xml_tool_call_0123456789abcdef0123456789abcdef",
        "xml_tool_call_0123456789abcdef0123456789abcdef",
    ]
    assert tool_calls[0].function.name == "lookup"
    assert tool_calls[0].function.arguments == {
        "empty": "",
        "enabled": True,
        "disabled": False,
        "missing": None,
        "also_missing": None,
        "config": {"enabled": True},
        "items": [1, 2],
        "duplicate": "second",
    }
    assert tool_calls[0].extra_content == {"provider": "xml_fallback"}
    assert tool_calls[1].function.name == "notify"
    assert tool_calls[1].function.arguments == {"text": "完成"}


@pytest.mark.parametrize(
    ("raw_text", "expected_content"),
    [
        pytest.param(None, None, id="none"),
        pytest.param("   ", "   ", id="whitespace"),
        pytest.param("plain text", "plain text", id="plain"),
        pytest.param(
            "<tool_call>missing function</tool_call>", "<tool_call>missing function</tool_call>", id="missing-function"
        ),
        pytest.param(
            "<tool_call><function=lookup>{}</function>",
            "<tool_call><function=lookup>{}</function>",
            id="unclosed-tool-call",
        ),
    ],
)
def test_extract_xml_tool_calls_ignores_non_calls(
    raw_text: str | None,
    expected_content: str | None,
) -> None:
    content, tool_calls = extract_xml_tool_calls(raw_text)

    assert content == expected_content
    assert tool_calls == []


def test_extract_xml_tool_calls_supports_empty_arguments() -> None:
    content, tool_calls = extract_xml_tool_calls("<tool_call><function=ping> </function></tool_call>")

    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "ping"
    assert tool_calls[0].function.arguments == {}


def test_extract_xml_tool_calls_uses_requested_strict_mode() -> None:
    sensitive_arguments = '{"secret":"sk-sensitive"'
    raw_text = f"<tool_call><function=lookup>{sensitive_arguments}</function></tool_call>"

    with pytest.raises(ValueError) as exc_info:
        extract_xml_tool_calls(raw_text, "strict")

    message = str(exc_info.value)
    assert f"<text:{len(sensitive_arguments)}>" in message
    assert "sk-sensitive" not in message
