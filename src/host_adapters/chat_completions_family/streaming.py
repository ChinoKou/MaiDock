from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field

from ...clients.common import SseJsonEvent
from ...core.common import RuntimeOptionsView, build_usage_from_snapshot
from ...core.diagnostics import build_parse_error_message, sanitize_json_object
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none, mapping_field, mapping_to_json_object
from ...core.parsing import ToolArgumentParseMode
from ...i18n import runtime_item, translate
from ...schemas import GenericUsageSnapshot, ProviderFunctionCall, ProviderResponse, ProviderToolCall
from ..common.httpx import HttpxProviderError, HttpxProviderParseError
from ..common.payloads import raw_data_or_none
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from ..common.tools import normalize_tool_arguments_value, resolve_tool_call_id


def _empty_str_list() -> list[str]:
    return []


def _empty_tool_chunks() -> dict[str, "ChatCompletionsToolCallChunk"]:
    return {}


@dataclass(slots=True)
class ChatCompletionsToolCallChunk:
    """流式工具调用片段合并器。"""

    call_id: str | None = None
    name: str = ""
    arguments_chunks: list[str] = field(default_factory=_empty_str_list)

    def merge_arguments(self, arguments: str) -> None:
        if not arguments:
            return
        current = "".join(self.arguments_chunks)
        if arguments.startswith(current):
            self.arguments_chunks = [arguments]
        else:
            self.arguments_chunks.append(arguments)

    def to_tool_call(
        self,
        index: int,
        parse_mode: ToolArgumentParseMode,
        *,
        extra_content: dict[str, JsonValue] | None = None,
    ) -> ProviderToolCall:
        raw_arguments = "".join(self.arguments_chunks)
        return ProviderToolCall(
            id=resolve_tool_call_id(self.call_id, fallback_prefix="chat_completions_tool", index=index),
            function=ProviderFunctionCall(
                name=self.name,
                arguments=normalize_tool_arguments_value(raw_arguments, parse_mode),
            ),
            extra_content=extra_content or {},
        )


@dataclass(slots=True)
class ChatCompletionsStreamAccumulator:
    """Chat Completions SSE 流增量收集器。"""

    options: RuntimeOptionsView
    content: str = ""
    reasoning_content: str = ""
    tools: dict[str, ChatCompletionsToolCallChunk] = field(default_factory=_empty_tool_chunks)
    usage: dict[str, JsonValue] = field(default_factory=dict)
    final_payload: dict[str, JsonValue] | None = None

    def merge_payload(self, payload: dict[str, JsonValue]) -> None:
        self.final_payload = payload
        usage = json_mapping_or_none(payload.get("usage"))
        if usage is not None:
            self.usage = mapping_to_json_object(usage)
        choice = self._first_choice(payload)
        if choice is None:
            return
        delta = mapping_field(choice, "delta")
        if delta is not None:
            self._merge_delta(delta)
        message = mapping_field(choice, "message")
        if message is not None:
            self._merge_message(message)

    def _merge_delta(self, delta: Mapping[str, JsonValue]) -> None:
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.content += content
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self.reasoning_content += reasoning
        self._merge_tool_calls(delta.get("tool_calls"))

    def _merge_message(self, message: Mapping[str, JsonValue]) -> None:
        content = message.get("content")
        if content is not None:
            from .chat import _message_content_text

            message_content_text_val = _message_content_text(content)
            if message_content_text_val:
                self.content = (
                    message_content_text_val
                    if message_content_text_val.startswith(self.content)
                    else self.content + message_content_text_val
                )
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            if reasoning.startswith(self.reasoning_content):
                self.reasoning_content = reasoning
            else:
                self.reasoning_content += reasoning
        self._merge_tool_calls(message.get("tool_calls"))

    def _merge_tool_calls(self, raw_tool_calls: object) -> None:
        tool_call_items = json_list_or_none(raw_tool_calls)
        if tool_call_items is None:
            return
        for index, item in enumerate(tool_call_items, start=1):
            tool_call = json_mapping_or_none(item)
            if tool_call is None:
                continue
            key = _tool_call_key(tool_call, index)
            chunk = self.tools.setdefault(key, ChatCompletionsToolCallChunk())
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id.strip():
                chunk.call_id = call_id
            function = mapping_field(tool_call, "function")
            if function is None:
                continue
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                chunk.name = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                chunk.merge_arguments(arguments)

    @staticmethod
    def _build_tool_extra_content(index: int, chunk: ChatCompletionsToolCallChunk) -> dict[str, JsonValue]:
        del index, chunk
        return {}

    def to_provider_response(self) -> ProviderResponse:
        tool_calls = [
            chunk.to_tool_call(
                index,
                self.options.tool_argument_parse_mode,
                extra_content=self._build_tool_extra_content(index, chunk),
            )
            for index, chunk in enumerate(self.tools.values(), start=1)
        ]
        reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
            content=self.content or None,
            native_reasoning=self.reasoning_content or None,
            tool_calls=tool_calls,
            options=self.options,
        )
        if not final_content and not tool_calls:
            message = translate(
                "runtime.error.output_missing",
                item=runtime_item("output_stream_text_or_tools"),
            )
            raise HttpxProviderParseError(build_parse_error_message("Chat Completions", message))
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(self.usage)),
            raw_data=self.final_payload,
        )

    @staticmethod
    def _first_choice(
        payload: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue] | None:
        from ...core.json_types import list_field

        choices = list_field(payload, "choices")
        if not choices:
            return None
        return json_mapping_or_none(choices[0])


def _tool_call_key(tool_call: Mapping[str, JsonValue], index: int) -> str:
    for key in ("id", "index"):
        value = tool_call.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return str(index)


def _stream_error_message(provider_label: str, payload: dict[str, JsonValue]) -> str | None:
    error = json_mapping_or_none(payload.get("error"))
    if error is not None:
        return translate(
            "runtime.error.stream",
            provider=provider_label,
            details=sanitize_json_object(error),
        )
    return None


async def collect_chat_completions_stream(
    events: AsyncIterator[SseJsonEvent],
    *,
    options: RuntimeOptionsView,
    provider_label: str,
) -> ProviderResponse:
    """收集 Chat Completions SSE 流并返回 ProviderResponse。"""
    accumulator = ChatCompletionsStreamAccumulator(options=options)
    async for event in events:
        error_message = _stream_error_message(provider_label, event.data)
        if error_message is not None:
            raise HttpxProviderError(error_message)
        accumulator.merge_payload(event.data)
    result = accumulator.to_provider_response()
    result.raw_data = raw_data_or_none(result.raw_data or {}, options=options)
    return result
