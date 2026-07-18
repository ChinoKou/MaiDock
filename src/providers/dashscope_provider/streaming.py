import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from ...core.common import ProviderRuntimeOptions, build_usage_from_snapshot
from ...core.diagnostics import build_parse_error_message, sanitize_for_log
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none, mapping_field, mapping_to_json_object
from ...schemas import GenericUsageSnapshot, ProviderResponse
from ..common.httpx import HttpxProviderError, HttpxProviderParseError, stream_sse_json
from ..common.payloads import raw_data_or_none
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .chat import (
    DASHSCOPE_PROVIDER_LABEL,
    extract_content_text,
    extract_reasoning_from_mapping,
    first_choice_message,
    is_multimodal_endpoint,
)
from .tools import DashScopeToolCallChunk


def _empty_dashscope_tool_dict() -> dict[str, DashScopeToolCallChunk]:
    return {}


@dataclass(slots=True)
class DashScopeStreamAccumulator:
    options: ProviderRuntimeOptions
    is_multimodal: bool = False
    content: str = ""
    reasoning_content: str = ""
    tools: dict[str, DashScopeToolCallChunk] = field(default_factory=_empty_dashscope_tool_dict)
    usage: dict = field(default_factory=dict)
    final_payload: dict | None = None

    def merge_payload(self, payload: dict) -> None:
        self.final_payload = payload
        usage = json_mapping_or_none(payload.get("usage"))
        if usage is not None:
            self.usage = mapping_to_json_object(usage)
        message = first_choice_message(payload)
        if message is None:
            output = mapping_field(payload, "output")
            if output is not None:
                text = output.get("text")
                if isinstance(text, str):
                    self._merge_content(text)
                self._merge_reasoning(extract_reasoning_from_mapping(output))
            return
        content = extract_content_text(message.get("content"), is_multimodal=self.is_multimodal)
        if content:
            self._merge_content(content)
        self._merge_reasoning(extract_reasoning_from_mapping(message))
        self._merge_tool_calls(message.get("tool_calls"))

    def _merge_content(self, content: str) -> None:
        if not content:
            return
        if content.startswith(self.content):
            self.content = content
        else:
            self.content += content

    def _merge_reasoning(self, reasoning: str | None) -> None:
        if not reasoning:
            return
        if reasoning.startswith(self.reasoning_content):
            self.reasoning_content = reasoning
        else:
            separator = "\n" if self.reasoning_content else ""
            self.reasoning_content = f"{self.reasoning_content}{separator}{reasoning}"

    def _merge_tool_calls(self, raw_tool_calls: object) -> None:
        tool_call_items = json_list_or_none(raw_tool_calls)
        if tool_call_items is None:
            return
        for index, item in enumerate(tool_call_items, start=1):
            tool_call = json_mapping_or_none(item)
            if tool_call is None:
                continue
            key = _tool_call_key(tool_call, index)
            chunk = self.tools.setdefault(key, DashScopeToolCallChunk())
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id.strip():
                chunk.call_id = call_id
            function_mapping = json_mapping_or_none(tool_call.get("function"))
            if function_mapping is None:
                continue
            name = function_mapping.get("name")
            arguments = function_mapping.get("arguments")
            if isinstance(name, str) and name.strip():
                chunk.name = name
            if isinstance(arguments, str):
                chunk.merge_arguments(arguments)
                continue
            argument_mapping = json_mapping_or_none(arguments)
            if argument_mapping is not None:
                chunk.merge_arguments(
                    json.dumps(
                        mapping_to_json_object(argument_mapping),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )

    def to_provider_response(self) -> ProviderResponse:
        tool_calls = [
            chunk.to_tool_call(index, self.options.tool_argument_parse_mode)
            for index, chunk in enumerate(self.tools.values(), start=1)
        ]
        reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
            content=self.content or None,
            native_reasoning=self.reasoning_content or None,
            tool_calls=tool_calls,
            options=self.options,
        )
        if not final_content and not tool_calls:
            raise HttpxProviderParseError(
                build_parse_error_message(DASHSCOPE_PROVIDER_LABEL, "流式响应中既没有文本内容，也没有工具调用")
            )
        raw_data: dict = self.final_payload if self.final_payload is not None else {"usage": self.usage}
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(self.usage)),
            raw_data=raw_data,
        )


def _tool_call_key(tool_call: Mapping[str, JsonValue], index: int) -> str:
    for key in ("id", "index"):
        value = tool_call.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return str(index)


def _stream_error_message(payload: dict, *, event_name: str | None, status: int | None) -> str | None:
    if event_name == "error":
        return f"{DASHSCOPE_PROVIDER_LABEL} 流式响应返回错误: {sanitize_for_log(payload)}"
    if status is not None and not 200 <= status < 300:
        return f"{DASHSCOPE_PROVIDER_LABEL} 流式响应状态码 {status}: {sanitize_for_log(payload)}"
    code = payload.get("code")
    if isinstance(code, str) and code.strip() and code.strip().lower() not in {"success", "ok"}:
        return f"{DASHSCOPE_PROVIDER_LABEL} 流式响应返回错误: {sanitize_for_log(payload)}"
    return None


async def collect_stream_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
    options: ProviderRuntimeOptions,
    max_retries: int,
    retry_interval: float,
) -> ProviderResponse:
    accumulator = DashScopeStreamAccumulator(options=options, is_multimodal=is_multimodal_endpoint(path))
    async for event in stream_sse_json(
        client,
        path,
        json_body=body,
        headers=headers,
        query=query,
        provider_label=DASHSCOPE_PROVIDER_LABEL,
        max_retries=max_retries,
        retry_interval=retry_interval,
    ):
        error_message = _stream_error_message(event.data, event_name=event.event, status=event.status)
        if error_message is not None:
            raise HttpxProviderError(error_message)
        accumulator.merge_payload(event.data)
    result = accumulator.to_provider_response()
    result.raw_data = raw_data_or_none(result.raw_data or {}, options=options)
    return result
