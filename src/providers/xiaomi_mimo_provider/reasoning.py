from hashlib import sha256
from time import monotonic, time
from typing import Literal
from pydantic import BaseModel, Field

import asyncio
import json

from ...core.json_types import json_mapping_or_none, mapping_to_json_object
from ...core.state_store import PluginStateStore
from ...schemas import ProviderResponse, ResponseRequestSnapshot, ToolCallSnapshot

MIMO_REASONING_NAMESPACE = "xiaomi_mimo.reasoning.v1"
_CLEANUP_INTERVAL_SECONDS = 3600.0


class MimoReasoningState(BaseModel):
    """SQLite 中保存的 Mimo 工具调用思考内容。"""

    schema_version: Literal[1] = 1
    reasoning_content: str = Field(min_length=1)
    created_at: float


class MimoReasoningManager:
    """保存并恢复 Mimo 多轮工具调用所需的 reasoning_content。"""

    def __init__(self, state_store: PluginStateStore, *, retention_days: int) -> None:
        self._state_store = state_store
        self._ttl_seconds = retention_days * 86400
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup_at = -_CLEANUP_INTERVAL_SECONDS

    async def restore_history(
        self,
        request: ResponseRequestSnapshot,
        body: dict,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        """从工具元数据或 SQLite 恢复历史 assistant 的 reasoning_content。"""

        await self._cleanup_if_due()
        assistant_messages = self._assistant_messages_by_call_id(body)
        for message in request.message_list:
            if message.role != "assistant" or not message.tool_calls:
                continue
            call_ids = self._history_call_ids(message.tool_calls)
            target = assistant_messages.get(call_ids[0])
            if target is None or any(assistant_messages.get(call_id) is not target for call_id in call_ids):
                raise ValueError("Mimo 历史工具调用无法与出站 assistant 消息对应")
            reasoning = await self._resolve_message_reasoning(
                message.tool_calls,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            target["reasoning_content"] = reasoning

    async def preserve_response(
        self,
        result: ProviderResponse,
        *,
        base_url: str,
        api_key: str,
        model: str,
        thinking_enabled: bool,
    ) -> None:
        """把新响应的 reasoning 写入工具元数据和 SQLite。"""

        if not result.tool_calls:
            return
        reasoning = (result.reasoning_content or "").strip()
        if not reasoning:
            if thinking_enabled:
                raise ValueError("Mimo 已启用思考，但工具调用响应缺少 reasoning_content")
            return

        await self._cleanup_if_due()
        created_at = time()
        for tool_call in result.tool_calls:
            call_id = tool_call.id.strip()
            if not call_id:
                raise ValueError("Mimo 工具调用缺少 call_id，无法保存 reasoning_content")
            extra_content = dict(tool_call.extra_content)
            provider_payload = self._provider_payload(extra_content)
            existing = provider_payload.get("reasoning_content")
            if existing is not None and existing != reasoning:
                raise ValueError(f"Mimo 工具调用 {call_id} 的 reasoning_content 元数据发生冲突")
            provider_payload["reasoning_content"] = reasoning
            provider_payload.setdefault(
                "raw_arguments",
                json.dumps(
                    tool_call.function.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            extra_content["provider"] = "xiaomi_mimo"
            extra_content["xiaomi_mimo"] = provider_payload
            tool_call.extra_content = extra_content
            await self._save_state(
                self._state_key(base_url, api_key, model, call_id),
                MimoReasoningState(
                    reasoning_content=reasoning,
                    created_at=created_at,
                ),
            )

    async def _resolve_message_reasoning(
        self,
        tool_calls: list[ToolCallSnapshot],
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> str:
        resolved: str | None = None
        states: dict[str, MimoReasoningState | None] = {}
        for tool_call in tool_calls:
            call_id = tool_call.resolved_call_id()
            state_key = self._state_key(base_url, api_key, model, call_id)
            state = await self._load_state(state_key)
            states[call_id] = state
            metadata_reasoning = self._metadata_reasoning(tool_call)
            for candidate in (metadata_reasoning, None if state is None else state.reasoning_content):
                if candidate is None:
                    continue
                if resolved is not None and candidate != resolved:
                    raise ValueError(f"Mimo 历史工具调用 {call_id} 的 reasoning_content 来源发生冲突")
                resolved = candidate

        if resolved is None:
            joined_call_ids = ", ".join(states)
            raise ValueError(f"Mimo 历史工具调用缺少可回传的 reasoning_content: {joined_call_ids}")

        current_time = time()
        for call_id, state in states.items():
            await self._save_state(
                self._state_key(base_url, api_key, model, call_id),
                MimoReasoningState(
                    reasoning_content=resolved,
                    created_at=current_time if state is None else state.created_at,
                ),
            )
        return resolved

    async def _load_state(self, key: str) -> MimoReasoningState | None:
        raw_state = await self._state_store.get(MIMO_REASONING_NAMESPACE, key)
        if raw_state is None:
            return None
        return MimoReasoningState.model_validate(raw_state)

    async def _save_state(self, key: str, state: MimoReasoningState) -> None:
        await self._state_store.set(
            MIMO_REASONING_NAMESPACE,
            key,
            state.model_dump(mode="json"),
            expires_at=time() + self._ttl_seconds,
        )

    async def _cleanup_if_due(self) -> None:
        current = monotonic()
        if current - self._last_cleanup_at < _CLEANUP_INTERVAL_SECONDS:
            return
        async with self._cleanup_lock:
            current = monotonic()
            if current - self._last_cleanup_at < _CLEANUP_INTERVAL_SECONDS:
                return
            await self._state_store.delete_expired(MIMO_REASONING_NAMESPACE)
            self._last_cleanup_at = current

    @staticmethod
    def _state_key(base_url: str, api_key: str, model: str, call_id: str) -> str:
        credential_digest = sha256(api_key.encode("utf-8")).hexdigest()
        scope = "\0".join(("v1", base_url.rstrip("/"), credential_digest, model, call_id))
        return sha256(scope.encode("utf-8")).hexdigest()

    @staticmethod
    def _history_call_ids(tool_calls: list[ToolCallSnapshot]) -> list[str]:
        call_ids: list[str] = []
        for tool_call in tool_calls:
            call_id = tool_call.resolved_call_id()
            if not call_id:
                raise ValueError("Mimo 历史工具调用缺少 call_id，无法恢复 reasoning_content")
            if call_id in call_ids:
                raise ValueError(f"Mimo 历史 assistant 中存在重复的 call_id: {call_id}")
            call_ids.append(call_id)
        return call_ids

    @staticmethod
    def _assistant_messages_by_call_id(body: dict) -> dict[str, dict]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise TypeError("Mimo 请求体 messages 必须是数组")
        result: dict[str, dict] = {}
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise TypeError("Mimo assistant.tool_calls 必须由 object 组成")
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError("Mimo 出站历史工具调用缺少 call_id")
                if call_id in result:
                    raise ValueError(f"Mimo 出站历史中存在重复的 call_id: {call_id}")
                result[call_id] = message
        return result

    @staticmethod
    def _metadata_reasoning(tool_call: ToolCallSnapshot) -> str | None:
        extra_content = tool_call.extra_content.to_plain_dict()
        provider_payload = json_mapping_or_none(extra_content.get("xiaomi_mimo"))
        if provider_payload is None:
            if "xiaomi_mimo" in extra_content:
                raise TypeError("Mimo 工具调用 extra_content.xiaomi_mimo 必须是 object")
            return None
        value = provider_payload.get("reasoning_content")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Mimo 工具调用 reasoning_content 元数据必须是非空字符串")
        return value.strip()

    @staticmethod
    def _provider_payload(extra_content: dict) -> dict:
        raw_payload = extra_content.get("xiaomi_mimo")
        if raw_payload is None:
            return {}
        payload = json_mapping_or_none(raw_payload)
        if payload is None:
            raise TypeError("Mimo 工具调用 extra_content.xiaomi_mimo 必须是 object")
        return mapping_to_json_object(payload)
