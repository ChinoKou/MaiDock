import json
import re
from typing import Literal
from uuid import uuid4

from json_repair import repair_json
from pydantic import TypeAdapter

from .schemas import JsonObject, JsonValue, ObjectFields, ProviderFunctionCall, ProviderToolCall

ToolArgumentParseMode = Literal["auto", "strict", "repair", "double_decode"]
ReasoningParseMode = Literal["auto", "native", "think_tag", "none"]

_DICT_ADAPTER = TypeAdapter(JsonObject)

THINK_CONTENT_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>(?P<content>.*)|<think>(?P<think_unclosed>.*)|(?P<content_only>.+)",
    re.DOTALL,
)
XML_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
XML_FUNCTION_CALL_PATTERN = re.compile(
    r"<function=(?P<name>[A-Za-z0-9_.-]+)>\s*(?P<arguments>.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
XML_PARAMETER_PATTERN = re.compile(
    r"<parameter=(?P<name>[A-Za-z0-9_.-]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def normalize_tool_argument_parse_mode(parse_mode: str | None) -> ToolArgumentParseMode:
    """规范化工具参数解析模式。"""

    if parse_mode == "strict":
        return "strict"
    if parse_mode == "repair":
        return "repair"
    if parse_mode == "double_decode":
        return "double_decode"
    return "auto"


def normalize_reasoning_parse_mode(parse_mode: str | None) -> ReasoningParseMode:
    """规范化 reasoning 解析模式。"""

    if parse_mode == "native":
        return "native"
    if parse_mode == "think_tag":
        return "think_tag"
    if parse_mode == "none":
        return "none"
    return "auto"


def normalize_arguments(value: ObjectFields | str | None, parse_mode: ToolArgumentParseMode = "auto") -> JsonObject:
    """解析 Host 快照或上游响应中的工具参数。"""

    if isinstance(value, ObjectFields):
        return value.to_plain_dict()
    if value is None:
        return {}
    return parse_tool_arguments(value, parse_mode)


def _argument_preview(raw_arguments: str) -> str:
    return f"<text:{len(raw_arguments)}>"


def _tool_argument_error(raw_arguments: str, reason: str) -> ValueError:
    return ValueError(f"无法解析工具调用参数: {reason}。参数预览: {_argument_preview(raw_arguments)}")


def parse_tool_arguments(raw_arguments: str, parse_mode: ToolArgumentParseMode = "auto") -> JsonObject:
    """解析工具调用参数字符串，空字符串按无参函数处理。"""

    if not raw_arguments.strip():
        return {}

    try:
        if parse_mode == "strict":
            arguments: object = json.loads(raw_arguments)
        elif parse_mode == "repair":
            arguments = repair_json(raw_arguments, return_objects=True, logging=False)
        else:
            arguments = repair_json(raw_arguments, return_objects=True, logging=False)
            if isinstance(arguments, str) and parse_mode in {"auto", "double_decode"}:
                arguments = repair_json(arguments, return_objects=True, logging=False)
    except Exception as exc:
        raise _tool_argument_error(raw_arguments, type(exc).__name__) from exc

    if not isinstance(arguments, dict):
        raise _tool_argument_error(raw_arguments, f"解析结果必须是 object，实际为 {type(arguments).__name__}")
    return _DICT_ADAPTER.validate_python(arguments)


def arguments_to_json(value: ObjectFields | str | None, parse_mode: ToolArgumentParseMode = "auto") -> str:
    """将 Host 快照工具参数转为上游需要的 JSON 字符串。"""

    if isinstance(value, str):
        parsed = parse_tool_arguments(value, parse_mode)
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(value, ObjectFields):
        return json.dumps(value.to_plain_dict(), ensure_ascii=False)
    return "{}"


def fallback_tool_call_id(prefix: str, *, max_length: int = 96) -> str:
    """为缺失 id 的工具调用生成稳定格式的兜底 id。"""

    normalized_prefix = str(prefix or "tool_call").strip() or "tool_call"
    suffix = uuid4().hex
    result = f"{normalized_prefix}_{suffix}"
    if len(result) <= max_length:
        return result
    return f"{normalized_prefix[: max_length - len(suffix) - 1]}_{suffix}"


def extract_reasoning_and_content(
    content: str, parse_mode: ReasoningParseMode = "auto"
) -> tuple[str | None, str | None]:
    """从 `<think>` 文本中分离 reasoning 和正式内容。"""

    if parse_mode in {"native", "none"}:
        return None, content
    match = THINK_CONTENT_PATTERN.match(content)
    if not match:
        return None, content
    if match.group("think") is not None:
        reasoning_content = match.group("think").strip() or None
        final_content = match.group("content").strip() or None
        return reasoning_content, final_content
    if match.group("think_unclosed") is not None:
        return match.group("think_unclosed").strip() or None, None
    return None, match.group("content_only").strip() or None


def merge_native_or_text_reasoning(
    *,
    content: str | None,
    native_reasoning: str | None,
    parse_mode: ReasoningParseMode,
) -> tuple[str | None, str | None]:
    """根据解析模式合并 provider 原生 reasoning 与文本 reasoning。"""

    if parse_mode == "none":
        return None, content
    if native_reasoning:
        return native_reasoning, content
    if content:
        return extract_reasoning_and_content(content, parse_mode)
    return None, content


def _coerce_xml_parameter_value(raw_value: str) -> JsonValue:
    normalized_value = raw_value.strip()
    if not normalized_value:
        return ""
    lowered_value = normalized_value.lower()
    if lowered_value == "true":
        return True
    if lowered_value == "false":
        return False
    if lowered_value in {"null", "none"}:
        return None
    if normalized_value.startswith(("{", "[")):
        try:
            return repair_json(normalized_value, return_objects=True, logging=False)
        except Exception:
            return normalized_value
    return normalized_value


def _parse_xml_parameters(raw_arguments: str) -> JsonObject | None:
    parameters: JsonObject = {}
    for match in XML_PARAMETER_PATTERN.finditer(raw_arguments):
        parameters[match.group("name").strip()] = _coerce_xml_parameter_value(match.group("value"))
    return parameters or None


def extract_xml_tool_calls(
    raw_text: str | None,
    parse_mode: ToolArgumentParseMode = "auto",
) -> tuple[str | None, list[ProviderToolCall]]:
    """从 XML 风格文本中兜底解析工具调用。"""

    if not isinstance(raw_text, str) or not raw_text.strip():
        return raw_text, []

    tool_calls: list[ProviderToolCall] = []

    def replace_tool_call(match: re.Match[str]) -> str:
        body = match.group("body")
        function_match = XML_FUNCTION_CALL_PATTERN.search(body)
        if function_match is None:
            return match.group(0)

        function_name = function_match.group("name").strip()
        raw_arguments = function_match.group("arguments").strip()
        arguments = _parse_xml_parameters(raw_arguments)
        if arguments is None:
            arguments = parse_tool_arguments(raw_arguments, parse_mode) if raw_arguments else {}
        tool_calls.append(
            ProviderToolCall(
                id=fallback_tool_call_id("xml_tool_call"),
                function=ProviderFunctionCall(name=function_name, arguments=arguments),
                extra_content={"provider": "xml_fallback"},
            )
        )
        return ""

    cleaned_text = XML_TOOL_CALL_PATTERN.sub(replace_tool_call, raw_text).strip() or None
    return cleaned_text, tool_calls
