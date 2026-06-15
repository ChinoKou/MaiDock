from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from ...core.diagnostics import sanitize_for_log, sanitize_json_object
from ...core.json_types import mapping_field, mapping_to_json_object, json_mapping_or_none, string_field
from ...schemas import OpenAIResponseOutputContentBlock, OpenAIResponseOutputItem
from ..common.httpx import HttpxProviderError, stream_sse_json


def _empty_str_list() -> list[str]:
    return []


def _empty_tool_chunks() -> dict[str, "ResponsesToolCallChunk"]:
    return {}


@dataclass(slots=True)
class ResponsesToolCallChunk:
    item_id: str | None = None
    call_id: str | None = None
    name: str = ""
    arguments_chunks: list[str] = field(default_factory=_empty_str_list)
    status: str | None = None

    def append_arguments(self, delta: str) -> None:
        if delta:
            self.arguments_chunks.append(delta)

    def set_arguments(self, arguments: str) -> None:
        self.arguments_chunks = [arguments] if arguments else []

    def to_output_item(self, index: int, *, fallback_prefix: str) -> OpenAIResponseOutputItem:
        call_id = self.call_id or f"{fallback_prefix}_{index}"
        return OpenAIResponseOutputItem(
            type="function_call",
            id=self.item_id,
            call_id=call_id,
            name=self.name,
            arguments="".join(self.arguments_chunks),
            status=self.status or "completed",
        )


@dataclass(slots=True)
class ResponsesStreamAccumulator:
    model: str
    tool_fallback_prefix: str
    text_chunks: list[str] = field(default_factory=_empty_str_list)
    reasoning_chunks: list[str] = field(default_factory=_empty_str_list)
    tools: dict[str, ResponsesToolCallChunk] = field(default_factory=_empty_tool_chunks)
    usage: Mapping = field(default_factory=dict)

    def append_text(self, delta: str) -> None:
        if delta:
            self.text_chunks.append(delta)

    def set_text(self, text: str) -> None:
        if text:
            self.text_chunks = [text]

    def append_reasoning(self, delta: str) -> None:
        if delta:
            self.reasoning_chunks.append(delta)

    def set_reasoning(self, text: str) -> None:
        if text:
            self.reasoning_chunks = [text]

    def merge_tool_item(self, item: dict) -> None:
        if item.get("type") != "function_call":
            return
        key = _event_key(item)
        tool = self.tools.setdefault(key, ResponsesToolCallChunk())
        item_id = item.get("id")
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        status = item.get("status")
        if isinstance(item_id, str) and item_id.strip():
            tool.item_id = item_id
        if isinstance(call_id, str) and call_id.strip():
            tool.call_id = call_id
        if isinstance(name, str) and name.strip():
            tool.name = name
        if isinstance(arguments, str):
            tool.set_arguments(arguments)
        if isinstance(status, str) and status.strip():
            tool.status = status

    def append_tool_arguments(self, event: dict, delta: str) -> None:
        if not delta:
            return
        tool = self._tool_for_event(event)
        tool.append_arguments(delta)

    def set_tool_arguments(self, event: dict, arguments: str) -> None:
        tool = self._tool_for_event(event)
        tool.set_arguments(arguments)

    def merge_usage(self, value: object) -> None:
        usage = json_mapping_or_none(value)
        if usage is not None:
            merged = dict(self.usage)
            merged.update(usage)
            self.usage = merged

    def _tool_for_event(self, event: dict) -> ResponsesToolCallChunk:
        key = _event_key(event)
        tool = self.tools.setdefault(key, ResponsesToolCallChunk())
        item_id = event.get("item_id") or event.get("id")
        call_id = event.get("call_id")
        name = event.get("name")
        if isinstance(item_id, str) and item_id.strip():
            tool.item_id = item_id
        if isinstance(call_id, str) and call_id.strip():
            tool.call_id = call_id
        if isinstance(name, str) and name.strip():
            tool.name = name
        return tool

    def to_response_payload(self) -> dict:
        output: list[object] = []
        text_content = "".join(self.text_chunks)
        if text_content:
            output.append(
                OpenAIResponseOutputItem(
                    type="message",
                    status="completed",
                    content=[OpenAIResponseOutputContentBlock(type="output_text", text=text_content)],
                ).model_dump(mode="json", exclude_none=True)
            )
        reasoning_content = "".join(self.reasoning_chunks)
        if reasoning_content:
            output.append(
                OpenAIResponseOutputItem(
                    type="reasoning_summary",
                    status="completed",
                    summary=[OpenAIResponseOutputContentBlock(type="summary_text", text=reasoning_content)],
                ).model_dump(mode="json", exclude_none=True)
            )
        for index, tool in enumerate(self.tools.values(), start=1):
            output.append(
                tool.to_output_item(index, fallback_prefix=self.tool_fallback_prefix).model_dump(
                    mode="json", exclude_none=True
                )
            )
        return {
            "model": self.model,
            "status": "completed",
            "output_text": text_content,
            "output": output,
            "usage": self.usage,
        }


def _event_type(event: dict, sse_event: str | None) -> str:
    value = event.get("type") or event.get("event") or sse_event
    return value if isinstance(value, str) else ""


def _event_key(event: dict) -> str:
    for key in ("item_id", "id", "output_index", "index", "call_id"):
        value = event.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return "default"


def _json_response_field(event: dict) -> Mapping | None:
    response = json_mapping_or_none(event.get("response"))
    if response is not None:
        return response
    if "output" in event and ("object" in event or "status" in event or "output_text" in event):
        return sanitize_json_object(event)
    return None


def _terminal_error_message(provider_label: str, event_type: str, event: dict) -> str | None:
    bare_error = event.get("error")
    if event_type == "error" or bare_error is not None:
        return f"{provider_label} 流式响应返回错误: {sanitize_for_log(bare_error or sanitize_json_object(event))}"
    if event_type not in {"response.failed", "response.incomplete"}:
        return None
    response = mapping_field(event, "response")
    error_payload: object = None
    if response is not None:
        error_payload = response.get("error") or response.get("incomplete_details") or sanitize_json_object(response)
    if error_payload is None:
        error_payload = event.get("error") or event.get("message") or sanitize_json_object(event)
    return f"{provider_label} 流式响应状态为 {event_type}: {sanitize_for_log(error_payload)}"


def _merge_stream_item(
    event_type: str,
    event_mapping: dict,
    accumulator: ResponsesStreamAccumulator,
) -> None:
    item = mapping_field(event_mapping, "item") or mapping_field(event_mapping, "output_item")
    if item is not None:
        accumulator.merge_tool_item(mapping_to_json_object(item))
    if event_type == "response.output_text.delta":
        delta = string_field(event_mapping, "delta")
        if delta is not None:
            accumulator.append_text(delta)
    elif event_type == "response.output_text.done":
        text = string_field(event_mapping, "text")
        if text is not None:
            accumulator.set_text(text)
    elif event_type == "response.function_call_arguments.delta":
        delta = string_field(event_mapping, "delta")
        if delta is not None:
            accumulator.append_tool_arguments(event_mapping, delta)
    elif event_type == "response.function_call_arguments.done":
        arguments = string_field(event_mapping, "arguments")
        if arguments is not None:
            accumulator.set_tool_arguments(event_mapping, arguments)
    elif event_type == "response.reasoning_summary_text.delta":
        delta = string_field(event_mapping, "delta")
        if delta is not None:
            accumulator.append_reasoning(delta)
    elif event_type == "response.reasoning_summary_text.done":
        text = string_field(event_mapping, "text")
        if text is not None:
            accumulator.set_reasoning(text)


async def collect_responses_stream(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
    model: str,
    provider_label: str,
    tool_fallback_prefix: str,
    max_retries: int = 0,
) -> Mapping:
    accumulator = ResponsesStreamAccumulator(model=model, tool_fallback_prefix=tool_fallback_prefix)
    final_response: Mapping | None = None
    async for event in stream_sse_json(
        client,
        path,
        json_body=body,
        headers=headers,
        query=query,
        provider_label=provider_label,
        max_retries=max_retries,
    ):
        event_mapping = event.data
        event_type = _event_type(event_mapping, event.event)
        error_message = _terminal_error_message(provider_label, event_type, event_mapping)
        if error_message is not None:
            raise HttpxProviderError(error_message)
        response = _json_response_field(event_mapping)
        if response is not None:
            final_response = response
        usage = event_mapping.get("usage")
        if usage is not None:
            accumulator.merge_usage(usage)
        response_mapping = mapping_field(event_mapping, "response")
        if response_mapping is not None:
            accumulator.merge_usage(response_mapping.get("usage"))
        _merge_stream_item(event_type, event_mapping, accumulator)

    if final_response is not None:
        return final_response
    return accumulator.to_response_payload()
