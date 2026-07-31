import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ...clients.common import SseJsonEvent
from ...core.common import RuntimeOptionsView, build_usage_from_snapshot
from ...core.diagnostics import build_parse_error_message
from ...core.json_types import JsonValue, json_list_or_none, json_mapping_or_none, mapping_field, mapping_to_json_object
from ...i18n import runtime_item, translate
from ...schemas import GenericUsageSnapshot, ProviderResponse
from ..common.httpx import HttpxProviderParseError
from ..common.payloads import raw_data_or_none
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .chat import (
    DASHSCOPE_PROVIDER_LABEL,
    extract_content_text,
    extract_reasoning_from_mapping,
    first_choice_message,
)
from .tools import DashScopeToolCallChunk, merge_dashscope_stream_text


def _empty_dashscope_tool_list() -> list[DashScopeToolCallChunk]:
    return []


@dataclass(slots=True)
class DashScopeStreamAccumulator:
    options: RuntimeOptionsView
    is_multimodal: bool = False
    content: str = ""
    reasoning_content: str = ""
    tools: list[DashScopeToolCallChunk] = field(default_factory=_empty_dashscope_tool_list)
    usage: dict[str, JsonValue] = field(default_factory=dict)
    final_payload: dict[str, JsonValue] | None = None
    _index_slots: dict[int, DashScopeToolCallChunk] = field(default_factory=dict)
    _id_slots: dict[str, list[DashScopeToolCallChunk]] = field(default_factory=dict)
    _previous_event_slots: list[DashScopeToolCallChunk] = field(default_factory=list)
    _next_slot_id: int = 1
    _next_fragment_sequence: int = 1

    def merge_payload(self, payload: dict[str, JsonValue]) -> None:
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
        self.content = merge_dashscope_stream_text(self.content, content)

    def _merge_reasoning(self, reasoning: str | None) -> None:
        if not reasoning:
            return
        self.reasoning_content = merge_dashscope_stream_text(self.reasoning_content, reasoning)

    def _merge_tool_calls(self, raw_tool_calls: object) -> None:
        tool_call_items = json_list_or_none(raw_tool_calls)
        if tool_call_items is None:
            return
        current_event_slots: list[DashScopeToolCallChunk] = []
        used_slot_ids: set[int] = set()
        had_slots_before_event = bool(self.tools)
        for ordinal, item in enumerate(tool_call_items):
            tool_call = json_mapping_or_none(item)
            if tool_call is None:
                continue
            explicit_index = _tool_call_index(tool_call.get("index"))
            raw_call_id = tool_call.get("id")
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id else ""
            chunk = self._resolve_tool_slot(
                ordinal=ordinal,
                explicit_index=explicit_index,
                call_id=call_id,
                used_slot_ids=used_slot_ids,
                had_slots_before_event=had_slots_before_event,
            )
            self._bind_tool_identifiers(chunk, explicit_index=explicit_index)
            function_mapping = json_mapping_or_none(tool_call.get("function"))
            name = ""
            arguments = ""
            if function_mapping is not None:
                raw_name = function_mapping.get("name")
                if isinstance(raw_name, str):
                    name = raw_name
                raw_arguments = function_mapping.get("arguments")
                if isinstance(raw_arguments, str):
                    arguments = raw_arguments
                else:
                    argument_mapping = json_mapping_or_none(raw_arguments)
                    if argument_mapping is not None:
                        arguments = json.dumps(
                            mapping_to_json_object(argument_mapping),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
            chunk.add_fragment(
                sequence=self._next_fragment_sequence,
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
            self._next_fragment_sequence += 1
            if chunk.call_id:
                self._bind_id_alias(chunk.call_id, chunk)
            current_event_slots.append(chunk)
            used_slot_ids.add(chunk.slot_id)
        if current_event_slots:
            self._previous_event_slots = current_event_slots

    def _resolve_tool_slot(
        self,
        *,
        ordinal: int,
        explicit_index: int | None,
        call_id: str,
        used_slot_ids: set[int],
        had_slots_before_event: bool,
    ) -> DashScopeToolCallChunk:
        index_slot = self._index_slots.get(explicit_index) if explicit_index is not None else None
        id_slot = self._lookup_id_slot(call_id)
        if id_slot is not None and id_slot.slot_id in used_slot_ids:
            id_slot = None
        if index_slot is not None and id_slot is not None and index_slot is not id_slot:
            raise self._tool_stream_conflict("index/id")
        explicit_slot = index_slot or id_slot
        ordinal_slot = self._previous_event_slots[ordinal] if ordinal < len(self._previous_event_slots) else None
        if ordinal_slot is not None and ordinal_slot.slot_id in used_slot_ids:
            ordinal_slot = None

        if explicit_slot is not None:
            if (
                ordinal_slot is not None
                and ordinal_slot is not explicit_slot
                and not ordinal_slot.has_explicit_identity
            ):
                explicit_slot = self._merge_tool_slots(explicit_slot, ordinal_slot)
            return explicit_slot
        if ordinal_slot is not None:
            if explicit_index is not None and ordinal_slot.explicit_index not in {None, explicit_index}:
                return self._new_tool_slot()
            return ordinal_slot

        available_slots = [slot for slot in self.tools if slot.slot_id not in used_slot_ids]
        if len(available_slots) == 1:
            return available_slots[0]
        if len(available_slots) > 1 and explicit_index is None and not call_id:
            raise self._tool_stream_ambiguous(ordinal)
        if explicit_index is not None or call_id:
            return self._new_tool_slot()
        if not had_slots_before_event or not available_slots:
            return self._new_tool_slot()
        raise self._tool_stream_ambiguous(ordinal)

    def _new_tool_slot(self) -> DashScopeToolCallChunk:
        slot = DashScopeToolCallChunk(
            slot_id=self._next_slot_id,
            created_order=self._next_slot_id,
        )
        self._next_slot_id += 1
        self.tools.append(slot)
        return slot

    def _bind_tool_identifiers(
        self,
        slot: DashScopeToolCallChunk,
        *,
        explicit_index: int | None,
    ) -> None:
        if explicit_index is not None:
            existing = self._index_slots.get(explicit_index)
            if existing is not None and existing is not slot:
                raise self._tool_stream_conflict(f"index={explicit_index}")
            if slot.explicit_index is not None and slot.explicit_index != explicit_index:
                raise self._tool_stream_conflict(f"index={explicit_index}")
            slot.explicit_index = explicit_index
            self._index_slots[explicit_index] = slot

    def _lookup_id_slot(self, call_id: str) -> DashScopeToolCallChunk | None:
        if not call_id:
            return None
        candidates = self._id_slots.get(call_id)
        if candidates is None or len(candidates) != 1:
            return None
        return candidates[0]

    def _bind_id_alias(self, call_id: str, slot: DashScopeToolCallChunk) -> None:
        if not call_id:
            return
        candidates = self._id_slots.setdefault(call_id, [])
        if all(candidate is not slot for candidate in candidates):
            candidates.append(slot)

    def _merge_tool_slots(
        self,
        target: DashScopeToolCallChunk,
        source: DashScopeToolCallChunk,
    ) -> DashScopeToolCallChunk:
        if target is source:
            return target
        if target.has_explicit_identity and source.has_explicit_identity:
            raise self._tool_stream_conflict("identified slots")
        target.absorb(source)
        self.tools = [slot for slot in self.tools if slot is not source]
        self._index_slots = {index: target if slot is source else slot for index, slot in self._index_slots.items()}
        updated_id_slots: dict[str, list[DashScopeToolCallChunk]] = {}
        for alias, candidates in self._id_slots.items():
            updated_candidates: list[DashScopeToolCallChunk] = []
            for candidate in candidates:
                replacement = target if candidate is source else candidate
                if all(existing is not replacement for existing in updated_candidates):
                    updated_candidates.append(replacement)
            updated_id_slots[alias] = updated_candidates
        self._id_slots = updated_id_slots
        self._previous_event_slots = [target if slot is source else slot for slot in self._previous_event_slots]
        return target

    @staticmethod
    def _tool_stream_ambiguous(ordinal: int) -> HttpxProviderParseError:
        message = translate("runtime.error.dashscope_tool_stream_ambiguous", ordinal=ordinal + 1)
        return HttpxProviderParseError(build_parse_error_message(DASHSCOPE_PROVIDER_LABEL, message))

    @staticmethod
    def _tool_stream_conflict(details: str) -> HttpxProviderParseError:
        message = translate("runtime.error.dashscope_tool_stream_conflict", details=details)
        return HttpxProviderParseError(build_parse_error_message(DASHSCOPE_PROVIDER_LABEL, message))

    def to_provider_response(self) -> ProviderResponse:
        ordered_tools = sorted(
            self.tools,
            key=lambda chunk: (
                chunk.explicit_index is None,
                chunk.explicit_index if chunk.explicit_index is not None else chunk.created_order,
            ),
        )
        tool_calls = [
            chunk.to_tool_call(index, self.options.tool_argument_parse_mode)
            for index, chunk in enumerate(ordered_tools, start=1)
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
            raise HttpxProviderParseError(build_parse_error_message(DASHSCOPE_PROVIDER_LABEL, message))
        raw_data: dict[str, JsonValue] = self.final_payload if self.final_payload is not None else {"usage": self.usage}
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=build_usage_from_snapshot(GenericUsageSnapshot.model_validate(self.usage)),
            raw_data=raw_data,
        )


def _tool_call_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


async def collect_stream_response(
    events: AsyncIterator[SseJsonEvent],
    *,
    options: RuntimeOptionsView,
    is_multimodal: bool,
) -> ProviderResponse:
    accumulator = DashScopeStreamAccumulator(options=options, is_multimodal=is_multimodal)
    async for event in events:
        accumulator.merge_payload(event.data)
    result = accumulator.to_provider_response()
    result.raw_data = raw_data_or_none(result.raw_data or {}, options=options)
    return result
