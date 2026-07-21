"""Responses SSE 流式累积与终止条件单元测试。"""

from src.providers.responses_family.streaming import (
    ResponsesStreamAccumulator,
    ResponsesToolCallChunk,
    _event_key,
    _event_type,
    _json_response_field,
    _merge_stream_item,
    _terminal_error_message,
)

# ------------------------------------------------------------------
# ResponsesToolCallChunk
# ------------------------------------------------------------------


class TestResponsesToolCallChunk:
    def test_append_arguments_accumulates(self):
        chunk = ResponsesToolCallChunk()
        chunk.append_arguments("hello")
        chunk.append_arguments(" world")
        assert chunk.arguments_chunks == ["hello", " world"]

    def test_append_arguments_ignores_empty(self):
        chunk = ResponsesToolCallChunk()
        chunk.append_arguments("")
        chunk.append_arguments("ok")
        assert chunk.arguments_chunks == ["ok"]

    def test_set_arguments_replaces(self):
        chunk = ResponsesToolCallChunk()
        chunk.append_arguments("delta")
        chunk.set_arguments("final")
        assert chunk.arguments_chunks == ["final"]

    def test_set_arguments_empty_clears(self):
        chunk = ResponsesToolCallChunk()
        chunk.append_arguments("delta")
        chunk.set_arguments("")
        assert chunk.arguments_chunks == []

    def test_to_output_item_uses_call_id_when_present(self):
        chunk = ResponsesToolCallChunk(call_id="abc", name="search", arguments_chunks=["{}"])
        item = chunk.to_output_item(1, fallback_prefix="tool")
        assert item.call_id == "abc"
        assert item.name == "search"
        assert item.arguments == "{}"

    def test_to_output_item_falls_back_to_prefix(self):
        chunk = ResponsesToolCallChunk(name="search", arguments_chunks=["{}"])
        item = chunk.to_output_item(3, fallback_prefix="tool")
        assert item.call_id == "tool_3"


# ------------------------------------------------------------------
# ResponsesStreamAccumulator
# ------------------------------------------------------------------


class TestResponsesStreamAccumulator:
    def _accumulator(self, *, model: str = "gpt-test", prefix: str = "tool") -> ResponsesStreamAccumulator:
        return ResponsesStreamAccumulator(model=model, tool_fallback_prefix=prefix)

    # --- text ---

    def test_append_text(self):
        acc = self._accumulator()
        acc.append_text("hello")
        acc.append_text(" world")
        assert acc.text_chunks == ["hello", " world"]

    def test_set_text_replaces(self):
        acc = self._accumulator()
        acc.append_text("delta")
        acc.set_text("final")
        assert acc.text_chunks == ["final"]

    # --- merge_usage ---

    def test_merge_usage_combines_two_sources(self):
        acc = self._accumulator()
        acc.merge_usage({"input_tokens": 100, "output_tokens": 50})
        acc.merge_usage({"output_tokens": 60, "total_tokens": 160})
        assert dict(acc.usage) == {
            "input_tokens": 100,
            "output_tokens": 60,
            "total_tokens": 160,
        }

    def test_merge_usage_ignores_none(self):
        acc = self._accumulator()
        acc.merge_usage({"input_tokens": 100})
        acc.merge_usage(None)
        assert dict(acc.usage) == {"input_tokens": 100}

    def test_merge_usage_handles_non_mapping(self):
        acc = self._accumulator()
        acc.merge_usage("not_a_dict")
        acc.merge_usage(42)
        assert dict(acc.usage) == {}

    # --- merge_tool_item ---

    def test_merge_tool_item_basic(self):
        acc = self._accumulator()
        acc.merge_tool_item({"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "search"})
        assert "fc_1" in acc.tools
        assert acc.tools["fc_1"].name == "search"

    def test_merge_tool_item_skips_non_function_call(self):
        acc = self._accumulator()
        acc.merge_tool_item({"type": "message", "id": "msg_1"})
        assert not acc.tools

    # --- append/set tool arguments ---

    def test_append_tool_arguments_by_item_id(self):
        acc = self._accumulator()
        acc.append_tool_arguments({"item_id": "fc_1"}, "{")
        acc.append_tool_arguments({"item_id": "fc_1"}, "}")
        assert acc.tools["fc_1"].arguments_chunks == ["{", "}"]

    def test_set_tool_arguments_by_item_id(self):
        acc = self._accumulator()
        acc.append_tool_arguments({"item_id": "fc_1"}, "partial")
        acc.set_tool_arguments({"item_id": "fc_1"}, '{"q":"beijing"}')
        assert acc.tools["fc_1"].arguments_chunks == ['{"q":"beijing"}']

    # --- to_response_payload (delta-only fallback) ---

    def test_to_response_payload_delta_only(self):
        """仅 delta 事件无最终 response 时回退到合成 payload。"""
        acc = self._accumulator(model="gpt-4")
        acc.append_text("Hello world")
        acc.append_reasoning("Let me think...")
        acc.merge_tool_item({"type": "function_call", "id": "fc_1", "name": "search", "arguments": "{}"})
        acc.merge_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        payload = acc.to_response_payload()
        assert payload["model"] == "gpt-4"
        assert payload["status"] == "completed"
        assert payload["output_text"] == "Hello world"
        assert len(payload["output"]) == 3  # text + reasoning + tool
        assert payload["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_to_response_payload_empty(self):
        acc = self._accumulator()
        payload = acc.to_response_payload()
        assert payload["output"] == []
        assert payload["output_text"] == ""


# ------------------------------------------------------------------
# _event_key
# ------------------------------------------------------------------


class TestEventKey:
    def test_prefers_item_id(self):
        assert _event_key({"item_id": "abc", "id": "xyz"}) == "abc"

    def test_falls_back_to_id(self):
        assert _event_key({"id": "xyz"}) == "xyz"

    def test_falls_back_to_output_index(self):
        assert _event_key({"output_index": 0}) == "0"

    def test_falls_back_to_index(self):
        assert _event_key({"index": 2}) == "2"

    def test_falls_back_to_call_id(self):
        assert _event_key({"call_id": "call_123"}) == "call_123"

    def test_default_when_nothing_matches(self):
        assert _event_key({"other": 42}) == "default"


# ------------------------------------------------------------------
# _event_type
# ------------------------------------------------------------------


class TestEventType:
    def test_uses_type_field(self):
        assert _event_type({"type": "response.completed"}, None) == "response.completed"

    def test_falls_back_to_event_field(self):
        assert _event_type({"event": "response.output_text.delta"}, None) == "response.output_text.delta"

    def test_falls_back_to_sse_event(self):
        assert _event_type({}, "response.created") == "response.created"

    def test_empty_when_none_match(self):
        assert _event_type({}, None) == ""


# ------------------------------------------------------------------
# _terminal_error_message
# ------------------------------------------------------------------


class TestTerminalErrorMessage:
    LABEL = "TestProvider"

    def test_error_event_type(self):
        msg = _terminal_error_message(self.LABEL, "error", {"error": {"message": "bad"}})
        assert msg is not None
        assert "TestProvider" in msg
        assert "错误" in msg

    def test_bare_error_field(self):
        msg = _terminal_error_message(self.LABEL, "unknown", {"error": {"message": "bad"}})
        assert msg is not None

    def test_response_failed(self):
        msg = _terminal_error_message(self.LABEL, "response.failed", {"response": {"error": "quota"}})
        assert msg is not None
        assert "response.failed" in msg

    def test_response_incomplete(self):
        msg = _terminal_error_message(
            self.LABEL,
            "response.incomplete",
            {"response": {"incomplete_details": "timeout"}},
        )
        assert msg is not None

    def test_no_error_for_normal_event(self):
        msg = _terminal_error_message(self.LABEL, "response.output_text.delta", {"delta": "hi"})
        assert msg is None


# ------------------------------------------------------------------
# _json_response_field
# ------------------------------------------------------------------


class TestJsonResponseField:
    def test_detects_nested_response_object(self):
        result = _json_response_field({"response": {"object": "response", "status": "completed"}})
        assert result is not None
        assert result["status"] == "completed"

    def test_detects_inline_response(self):
        result = _json_response_field({"object": "response", "output": [], "status": "completed"})
        assert result is not None
        assert result["status"] == "completed"

    def test_no_response_for_delta_event(self):
        result = _json_response_field({"type": "response.output_text.delta", "delta": "hello"})
        assert result is None


# ------------------------------------------------------------------
# _merge_stream_item
# ------------------------------------------------------------------


class TestMergeStreamItem:
    def _acc(self) -> ResponsesStreamAccumulator:
        return ResponsesStreamAccumulator(model="gpt-test", tool_fallback_prefix="tool")

    def test_merge_text_delta(self):
        acc = self._acc()
        _merge_stream_item("response.output_text.delta", {"delta": "hello"}, acc)
        assert acc.text_chunks == ["hello"]

    def test_merge_text_done(self):
        acc = self._acc()
        acc.append_text("old")
        _merge_stream_item("response.output_text.done", {"text": "final"}, acc)
        assert acc.text_chunks == ["final"]

    def test_merge_tool_arguments_delta_and_done(self):
        """多 tool 并行：验证每个 tool 独立累积 arguments 且 key 不冲突。"""
        acc = self._acc()
        _merge_stream_item(
            "response.output_item.added",
            {"item": {"type": "function_call", "id": "fc_1", "name": "search"}},
            acc,
        )
        _merge_stream_item(
            "response.output_item.added",
            {"item": {"type": "function_call", "id": "fc_2", "name": "calc"}},
            acc,
        )
        _merge_stream_item(
            "response.function_call_arguments.delta",
            {"item_id": "fc_1", "delta": '{"q":'},
            acc,
        )
        _merge_stream_item(
            "response.function_call_arguments.delta",
            {"item_id": "fc_2", "delta": '{"expr":'},
            acc,
        )
        _merge_stream_item(
            "response.function_call_arguments.done",
            {"item_id": "fc_1", "arguments": '{"q":"beijing"}'},
            acc,
        )
        _merge_stream_item(
            "response.function_call_arguments.delta",
            {"item_id": "fc_2", "delta": '"2+2"}'},
            acc,
        )
        assert acc.tools["fc_1"].arguments_chunks == ['{"q":"beijing"}']
        assert acc.tools["fc_2"].arguments_chunks == ['{"expr":', '"2+2"}']

    def test_merge_reasoning_delta(self):
        acc = self._acc()
        _merge_stream_item("response.reasoning_summary_text.delta", {"delta": "thinking..."}, acc)
        assert acc.reasoning_chunks == ["thinking..."]

    def test_merge_reasoning_done(self):
        acc = self._acc()
        acc.append_reasoning("old thinking")
        _merge_stream_item("response.reasoning_summary_text.done", {"text": "final thought"}, acc)
        assert acc.reasoning_chunks == ["final thought"]
