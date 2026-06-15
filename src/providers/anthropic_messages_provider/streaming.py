from collections.abc import Mapping

import httpx

from ...core.json_types import JsonValue, json_mapping_or_none, mapping_to_json_object
from ...core.parsing import ToolArgumentParseMode, normalize_arguments
from ..common.httpx import stream_sse_json
from .messages import ANTHROPIC_PROVIDER_LABEL


async def collect_stream_response(
    client: httpx.AsyncClient,
    path: str,
    body: dict,
    *,
    headers: Mapping[str, str],
    query: Mapping[str, object],
    parse_mode: ToolArgumentParseMode,
) -> dict:
    content_blocks: list[dict] = []
    input_json_parts: dict[int, list[str]] = {}
    message_payload: dict = {"content": content_blocks, "usage": {}}
    async for event in stream_sse_json(
        client,
        path,
        json_body=body,
        headers=headers,
        query=query,
        provider_label=ANTHROPIC_PROVIDER_LABEL,
    ):
        event_payload = event.data
        event_type = str(event_payload.get("type") or event.event or "")
        message = json_mapping_or_none(event_payload.get("message"))
        content_block = json_mapping_or_none(event_payload.get("content_block"))
        delta = json_mapping_or_none(event_payload.get("delta"))
        if event_type == "message_start" and message is not None:
            merge_message_start(message_payload, message)
        elif event_type == "content_block_start" and content_block is not None:
            index = stream_block_index(event_payload, content_blocks)
            put_stream_block(
                content_blocks,
                index,
                content_block,
            )
        elif event_type == "content_block_delta" and delta is not None:
            index = stream_block_index(event_payload, content_blocks)
            block = ensure_stream_block(content_blocks, index)
            apply_stream_delta(
                block,
                delta,
                input_json_parts,
                index,
            )
        elif event_type == "content_block_stop":
            index = stream_block_index(event_payload, content_blocks)
            if 0 <= index < len(content_blocks):
                finalize_stream_block(content_blocks[index], input_json_parts.get(index, []), parse_mode=parse_mode)
        elif event_type == "message_delta":
            merge_message_delta(message_payload, event_payload)
    message_payload["content"] = content_blocks
    return message_payload


def merge_message_start(message_payload: dict, message: Mapping) -> None:
    for key, value in message.items():
        if key == "content":
            continue
        if key == "usage":
            usage = json_mapping_or_none(value)
            if usage is not None:
                merge_stream_usage(message_payload, usage)
            continue
        message_payload[key] = value


def merge_message_delta(message_payload: dict, event: dict) -> None:
    delta = json_mapping_or_none(event.get("delta"))
    if delta is not None:
        for key in ("stop_reason", "stop_sequence"):
            value = delta.get(key)
            if value is not None:
                message_payload[key] = value
        usage = json_mapping_or_none(delta.get("usage"))
        if usage is not None:
            merge_stream_usage(message_payload, usage)
    usage = json_mapping_or_none(event.get("usage"))
    if usage is not None:
        merge_stream_usage(message_payload, usage)


def merge_stream_usage(message_payload: dict, usage: Mapping[str, JsonValue]) -> None:
    current_usage = json_mapping_or_none(message_payload.get("usage"))
    merged_usage = mapping_to_json_object(current_usage) if current_usage is not None else {}
    merged_usage.update(mapping_to_json_object(usage))
    message_payload["usage"] = merged_usage


def stream_block_index(event: dict, content_blocks: list[dict]) -> int:
    index = event.get("index")
    if isinstance(index, int) and index >= 0:
        return index
    return max(len(content_blocks) - 1, 0)


def ensure_stream_block(content_blocks: list[dict], index: int) -> dict:
    while len(content_blocks) <= index:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks[index]


def put_stream_block(content_blocks: list[dict], index: int, block: Mapping) -> None:
    ensure_stream_block(content_blocks, index)
    content_blocks[index] = dict(block)


def apply_stream_delta(
    block: dict,
    delta: Mapping,
    input_json_parts: dict[int, list[str]],
    index: int,
) -> None:
    delta_type = str(delta.get("type") or "")
    text = delta.get("text")
    thinking = delta.get("thinking")
    partial_json = delta.get("partial_json")
    if isinstance(text, str) or delta_type == "text_delta":
        block["type"] = block.get("type") or "text"
        block["text"] = str(block.get("text") or "") + (text if isinstance(text, str) else "")
    if isinstance(thinking, str) or delta_type == "thinking_delta":
        block["type"] = block.get("type") or "thinking"
        block["thinking"] = str(block.get("thinking") or "") + (thinking if isinstance(thinking, str) else "")
    if isinstance(partial_json, str):
        input_json_parts.setdefault(index, []).append(partial_json)


def finalize_stream_block(
    block: dict,
    input_parts: list[str],
    *,
    parse_mode: ToolArgumentParseMode,
) -> None:
    if block.get("type") not in ("tool_use", "tool_calls") or not input_parts:
        return
    raw_input = "".join(input_parts)
    block["input"] = normalize_arguments(raw_input, parse_mode)
