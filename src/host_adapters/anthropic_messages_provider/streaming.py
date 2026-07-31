from collections.abc import AsyncIterator, Mapping

from ...clients.common import SseJsonEvent

from ...core.json_types import JsonValue, json_array, json_mapping_or_none, mapping_to_json_object
from ...core.parsing import ToolArgumentParseMode, normalize_arguments


async def collect_stream_response(
    events: AsyncIterator[SseJsonEvent],
    *,
    parse_mode: ToolArgumentParseMode,
) -> dict[str, JsonValue]:
    content_blocks: list[dict[str, JsonValue]] = []
    input_json_parts: dict[int, list[str]] = {}
    message_payload: dict[str, JsonValue] = {"content": json_array(content_blocks), "usage": {}}
    async for event in events:
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
                finalize_stream_block(
                    content_blocks[index],
                    input_json_parts.get(index, []),
                    parse_mode=parse_mode,
                )
        elif event_type == "message_delta":
            merge_message_delta(message_payload, event_payload)
    message_payload["content"] = json_array(content_blocks)
    return message_payload


def merge_message_start(message_payload: dict[str, JsonValue], message: Mapping[str, JsonValue]) -> None:
    for key, value in message.items():
        if key == "content":
            continue
        if key == "usage":
            usage = json_mapping_or_none(value)
            if usage is not None:
                merge_stream_usage(message_payload, usage)
            continue
        message_payload[key] = value


def merge_message_delta(message_payload: dict[str, JsonValue], event: dict[str, JsonValue]) -> None:
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


def merge_stream_usage(message_payload: dict[str, JsonValue], usage: Mapping[str, JsonValue]) -> None:
    current_usage = json_mapping_or_none(message_payload.get("usage"))
    merged_usage = mapping_to_json_object(current_usage) if current_usage is not None else {}
    merged_usage.update(mapping_to_json_object(usage))
    message_payload["usage"] = merged_usage


def stream_block_index(event: dict[str, JsonValue], content_blocks: list[dict[str, JsonValue]]) -> int:
    index = event.get("index")
    if isinstance(index, int) and index >= 0:
        return index
    return max(len(content_blocks) - 1, 0)


def ensure_stream_block(content_blocks: list[dict[str, JsonValue]], index: int) -> dict[str, JsonValue]:
    while len(content_blocks) <= index:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks[index]


def put_stream_block(content_blocks: list[dict[str, JsonValue]], index: int, block: Mapping[str, JsonValue]) -> None:
    ensure_stream_block(content_blocks, index)
    content_blocks[index] = dict(block)


def apply_stream_delta(
    block: dict[str, JsonValue],
    delta: Mapping[str, JsonValue],
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
    block: dict[str, JsonValue],
    input_parts: list[str],
    *,
    parse_mode: ToolArgumentParseMode,
) -> None:
    if block.get("type") != "tool_use" or not input_parts:
        return
    raw_input = "".join(input_parts)
    block["input"] = normalize_arguments(raw_input, parse_mode)
