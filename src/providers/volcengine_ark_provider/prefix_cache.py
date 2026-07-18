from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import time
from typing import Final, cast
from pydantic import BaseModel, ConfigDict, Field
from weakref import WeakValueDictionary
import asyncio
import httpx
import json
import logging
import uuid

from ...core.json_types import (
    JsonValue,
    is_json_mapping,
    json_list_or_none,
    json_mapping_or_none,
    mapping_to_json_object,
    normalize_json_value,
)
from ...core.state_store import PluginStateStore
from ..common.httpx import HttpxClientConfig, post_json

ARK_TOKENIZATION_ENDPOINT: Final[str] = "tokenization"
PREFIX_CACHE_NAMESPACE: Final[str] = "volcengine_ark.prefix_cache.v1"
PREFIX_CACHE_MIN_TOKENS: Final[int] = 256
PREFIX_CACHE_EXPIRY_SAFETY_SECONDS: Final[int] = 60
PREFIX_CACHE_CREATION_DELAY_SECONDS: Final[float] = 0.1
PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS: Final[int] = 3600
PREFIX_CACHE_INELIGIBLE_TTL_SECONDS: Final[int] = 3600
PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES: Final[int] = 1024
_ARK_REQUEST_ID_HEADER: Final[str] = "X-Client-Request-Id"

logger = logging.getLogger("maibot_plugin.maidock.volcengine_ark.prefix_cache")


class PrefixCacheEntry(BaseModel):
    """持久化的 ARK 前缀缓存条目。"""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(min_length=1)
    expires_at: float
    input_tokens: int = Field(ge=PREFIX_CACHE_MIN_TOKENS)
    created_at: float


@dataclass(frozen=True, slots=True)
class PrefixCachePlan:
    cache_key: str
    model: str
    prefix_input: list[dict[str, JsonValue]]
    suffix_input: list[JsonValue]
    tools: list[dict[str, JsonValue]]
    thinking_present: bool
    thinking: JsonValue
    tokenization_texts: list[str]


@dataclass(frozen=True, slots=True)
class PrefixCacheResolution:
    body: dict
    cache_key: str | None = None


class PrefixCacheManager:
    """按照 ARK Responses API 协议管理固定前缀缓存。"""

    def __init__(self, state_store: PluginStateStore, *, ttl_seconds: int) -> None:
        if not 3600 <= ttl_seconds <= 604800:
            raise ValueError("ARK 前缀缓存 TTL 必须位于 3600..604800 秒")
        self._state_store = state_store
        self._ttl_seconds = ttl_seconds
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._locks_guard = asyncio.Lock()
        self._ineligible_until: dict[str, float] = {}
        self._cleanup_lock = asyncio.Lock()
        self._next_cleanup_at = 0.0

    async def resolve(
        self,
        client: httpx.AsyncClient,
        *,
        responses_path: str,
        tokenization_path: str,
        body: dict,
        headers: Mapping[str, str],
        query: Mapping[str, object],
        client_config: HttpxClientConfig,
        max_retries: int,
        retry_interval: float,
        now: float | None = None,
    ) -> PrefixCacheResolution:
        """创建或复用缓存，并返回只包含未缓存后缀的真实请求体。"""

        current_time = time() if now is None else now
        plan = self.build_plan(
            body,
            client_config=client_config,
            headers=headers,
            query=query,
        )
        if plan is None:
            return PrefixCacheResolution(body=dict(body))
        await self._delete_expired_if_due(now=current_time)

        lock = await self._lock_for(plan.cache_key)
        async with lock:
            entry = await self._load_entry(plan.cache_key, now=current_time)
            if entry is None:
                if self._is_ineligible(plan.cache_key, now=current_time):
                    return PrefixCacheResolution(body=dict(body))
                input_tokens = await self._count_prefix_tokens(
                    client,
                    tokenization_path=tokenization_path,
                    plan=plan,
                    headers=headers,
                    query=query,
                    max_retries=max_retries,
                    retry_interval=retry_interval,
                )
                if input_tokens < PREFIX_CACHE_MIN_TOKENS:
                    self._remember_ineligible(plan.cache_key, now=current_time)
                    logger.info(
                        "ARK system 前缀仅 %d tokens，小于 %d，跳过显式缓存",
                        input_tokens,
                        PREFIX_CACHE_MIN_TOKENS,
                    )
                    return PrefixCacheResolution(body=dict(body))
                entry = await self._create_entry(
                    client,
                    responses_path=responses_path,
                    plan=plan,
                    input_tokens=input_tokens,
                    headers=headers,
                    query=query,
                    max_retries=max_retries,
                    retry_interval=retry_interval,
                    now=current_time,
                )
                await asyncio.sleep(PREFIX_CACHE_CREATION_DELAY_SECONDS)

        reuse_body = dict(body)
        reuse_body["input"] = plan.suffix_input
        reuse_body["previous_response_id"] = entry.response_id
        reuse_body.pop("caching", None)
        reuse_body.pop("tools", None)
        logger.debug("复用 ARK 前缀缓存: model=%s key=%s", plan.model, plan.cache_key[:12])
        return PrefixCacheResolution(body=reuse_body, cache_key=plan.cache_key)

    def build_plan(
        self,
        body: Mapping[str, JsonValue],
        *,
        client_config: HttpxClientConfig,
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, object] | None = None,
    ) -> PrefixCachePlan | None:
        """根据最终请求体判断自动缓存是否适用，并生成稳定指纹。"""

        if "caching" in body or "previous_response_id" in body:
            logger.info("请求已显式设置 caching/previous_response_id，跳过自动前缀缓存")
            return None
        if body.get("store") is False:
            logger.info("请求显式设置 store=false，跳过自动前缀缓存")
            return None
        if self._has_instructions(body.get("instructions")):
            logger.info("ARK instructions 与显式缓存不兼容，跳过自动前缀缓存")
            return None
        if self._uses_json_schema(body):
            logger.info("ARK json_schema 与显式缓存不兼容，跳过自动前缀缓存")
            return None

        raw_input = json_list_or_none(body.get("input"))
        if raw_input is None:
            raise TypeError("ARK Responses input 必须是数组")
        prefix_input, suffix_input, tokenization_texts = self._split_system_prefix(raw_input)
        if not prefix_input or not suffix_input:
            return None

        tools = self._function_tools(body.get("tools"))
        if tools is None:
            logger.info("ARK 自动前缀缓存只支持 function tools，当前请求跳过缓存")
            return None

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ARK Responses 请求缺少 model")
        thinking_present = "thinking" in body
        thinking = normalize_json_value(body.get("thinking"))
        cache_key = self._cache_key(
            client_config=client_config,
            headers={} if headers is None else headers,
            query={} if query is None else query,
            model=model,
            prefix_input=prefix_input,
            tools=tools,
            thinking_present=thinking_present,
            thinking=thinking,
        )
        return PrefixCachePlan(
            cache_key=cache_key,
            model=model,
            prefix_input=prefix_input,
            suffix_input=suffix_input,
            tools=tools,
            thinking_present=thinking_present,
            thinking=thinking,
            tokenization_texts=tokenization_texts,
        )

    async def invalidate(self, cache_key: str) -> None:
        """删除失效的本地缓存引用。"""

        await self._state_store.delete(PREFIX_CACHE_NAMESPACE, cache_key)
        logger.info("已删除失效的 ARK 前缀缓存引用: key=%s", cache_key[:12])

    async def _load_entry(self, cache_key: str, *, now: float) -> PrefixCacheEntry | None:
        raw_entry = await self._state_store.get(PREFIX_CACHE_NAMESPACE, cache_key, now=now)
        if raw_entry is None:
            return None
        if not is_json_mapping(raw_entry):
            raise ValueError("ARK 前缀缓存持久化条目不是 JSON object")
        entry = PrefixCacheEntry.model_validate(raw_entry)
        if entry.expires_at <= now + PREFIX_CACHE_EXPIRY_SAFETY_SECONDS:
            await self.invalidate(cache_key)
            return None
        return entry

    async def _create_entry(
        self,
        client: httpx.AsyncClient,
        *,
        responses_path: str,
        plan: PrefixCachePlan,
        input_tokens: int,
        headers: Mapping[str, str],
        query: Mapping[str, object],
        max_retries: int,
        retry_interval: float,
        now: float,
    ) -> PrefixCacheEntry:
        expire_at = int(now + self._ttl_seconds)
        create_body: dict = {
            "model": plan.model,
            "input": plan.prefix_input,
            "store": True,
            "caching": {"type": "enabled", "prefix": True},
            "expire_at": expire_at,
        }
        if plan.tools:
            create_body["tools"] = plan.tools
        if plan.thinking_present:
            create_body["thinking"] = plan.thinking
        payload = await post_json(
            client,
            responses_path,
            json_body=create_body,
            headers=self._fresh_request_headers(headers),
            query=query,
            provider_label="Volcengine Ark Prefix Cache",
            max_retries=max_retries,
            retry_interval=retry_interval,
        )
        status = payload.get("status")
        if status in {"failed", "incomplete"}:
            details = payload.get("error") or payload.get("incomplete_details") or status
            raise ValueError(f"ARK 前缀缓存创建失败: {details}")
        response_id = payload.get("id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("ARK 前缀缓存创建响应缺少有效 id")
        entry = PrefixCacheEntry(
            response_id=response_id.strip(),
            expires_at=float(expire_at),
            input_tokens=input_tokens,
            created_at=now,
        )
        await self._state_store.set(
            PREFIX_CACHE_NAMESPACE,
            plan.cache_key,
            cast(JsonValue, entry.model_dump(mode="json")),
            expires_at=entry.expires_at,
            now=now,
        )
        logger.info(
            "已创建 ARK 前缀缓存: model=%s tokens=%d expires_at=%d key=%s",
            plan.model,
            input_tokens,
            expire_at,
            plan.cache_key[:12],
        )
        return entry

    async def _count_prefix_tokens(
        self,
        client: httpx.AsyncClient,
        *,
        tokenization_path: str,
        plan: PrefixCachePlan,
        headers: Mapping[str, str],
        query: Mapping[str, object],
        max_retries: int,
        retry_interval: float,
    ) -> int:
        text: str | list[str] = (
            plan.tokenization_texts[0] if len(plan.tokenization_texts) == 1 else plan.tokenization_texts
        )
        payload = await post_json(
            client,
            tokenization_path,
            json_body={"model": plan.model, "text": text},
            headers=self._fresh_request_headers(headers),
            query=query,
            provider_label="Volcengine Ark Tokenization",
            max_retries=max_retries,
            retry_interval=retry_interval,
        )
        data = json_list_or_none(payload.get("data"))
        if data is None or len(data) != len(plan.tokenization_texts):
            raise ValueError("ARK 分词响应 data 数量与 system 文本数量不一致")
        total_tokens = 0
        for item in data:
            item_mapping = json_mapping_or_none(item)
            if item_mapping is None:
                raise ValueError("ARK 分词响应 data 项不是 object")
            item_tokens = item_mapping.get("total_tokens")
            if not isinstance(item_tokens, int) or isinstance(item_tokens, bool) or item_tokens < 0:
                raise ValueError("ARK 分词响应 total_tokens 不是非负整数")
            total_tokens += item_tokens
        return total_tokens

    async def _delete_expired_if_due(self, *, now: float) -> None:
        if now < self._next_cleanup_at:
            return
        async with self._cleanup_lock:
            if now < self._next_cleanup_at:
                return
            deleted = await self._state_store.delete_expired(
                PREFIX_CACHE_NAMESPACE,
                now=now,
            )
            self._next_cleanup_at = now + PREFIX_CACHE_CLEANUP_INTERVAL_SECONDS
            if deleted:
                logger.info("已清理 %d 条过期 ARK 前缀缓存引用", deleted)

    def _is_ineligible(self, cache_key: str, *, now: float) -> bool:
        ineligible_until = self._ineligible_until.get(cache_key)
        if ineligible_until is None:
            return False
        if ineligible_until <= now:
            self._ineligible_until.pop(cache_key, None)
            return False
        return True

    def _remember_ineligible(self, cache_key: str, *, now: float) -> None:
        self._prune_ineligible(now=now)
        if (
            cache_key not in self._ineligible_until
            and len(self._ineligible_until) >= PREFIX_CACHE_INELIGIBLE_MAX_ENTRIES
        ):
            oldest_key = next(iter(self._ineligible_until))
            self._ineligible_until.pop(oldest_key)
        self._ineligible_until[cache_key] = now + PREFIX_CACHE_INELIGIBLE_TTL_SECONDS

    def _prune_ineligible(self, *, now: float) -> None:
        expired_keys = [
            cache_key
            for cache_key, ineligible_until in self._ineligible_until.items()
            if ineligible_until <= now
        ]
        for cache_key in expired_keys:
            self._ineligible_until.pop(cache_key)


    async def _lock_for(self, cache_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(cache_key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[cache_key] = lock
            return lock

    @staticmethod
    def _split_system_prefix(
        raw_input: list[JsonValue],
    ) -> tuple[list[dict[str, JsonValue]], list[JsonValue], list[str]]:
        prefix_input: list[dict[str, JsonValue]] = []
        tokenization_texts: list[str] = []
        for item in raw_input:
            item_mapping = json_mapping_or_none(item)
            if item_mapping is None or item_mapping.get("role") != "system":
                break
            system_item = mapping_to_json_object(item_mapping)
            texts = PrefixCacheManager._system_texts(system_item)
            if not texts:
                return [], raw_input, []
            prefix_input.append(system_item)
            tokenization_texts.extend(texts)
        return prefix_input, raw_input[len(prefix_input) :], tokenization_texts

    @staticmethod
    def _system_texts(message: Mapping[str, JsonValue]) -> list[str]:
        content = message.get("content")
        if isinstance(content, str):
            return [content]
        blocks = json_list_or_none(content)
        if blocks is None:
            return []
        texts: list[str] = []
        for block in blocks:
            block_mapping = json_mapping_or_none(block)
            if block_mapping is None or block_mapping.get("type") != "input_text":
                return []
            text = block_mapping.get("text")
            if not isinstance(text, str):
                return []
            texts.append(text)
        return texts

    @staticmethod
    def _function_tools(raw_tools: JsonValue) -> list[dict[str, JsonValue]] | None:
        if raw_tools is None:
            return []
        tools = json_list_or_none(raw_tools)
        if tools is None:
            return None
        result: list[dict[str, JsonValue]] = []
        for tool in tools:
            tool_mapping = json_mapping_or_none(tool)
            if tool_mapping is None or tool_mapping.get("type") != "function":
                return None
            result.append(mapping_to_json_object(tool_mapping))
        return result

    @staticmethod
    def _has_instructions(value: JsonValue) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _uses_json_schema(body: Mapping[str, JsonValue]) -> bool:
        text_config = json_mapping_or_none(body.get("text"))
        if text_config is None:
            return False
        format_config = json_mapping_or_none(text_config.get("format"))
        return format_config is not None and format_config.get("type") == "json_schema"

    @staticmethod
    def _cache_key(
        *,
        client_config: HttpxClientConfig,
        model: str,
        prefix_input: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]],
        headers: Mapping[str, str],
        query: Mapping[str, object],
        thinking_present: bool,
        thinking: JsonValue,
    ) -> str:
        effective_headers = {
            key.lower(): value
            for key, value in client_config.default_headers.items()
            if key.lower() != _ARK_REQUEST_ID_HEADER.lower()
        }
        effective_headers.update(
            (key.lower(), value)
            for key, value in headers.items()
            if key.lower() != _ARK_REQUEST_ID_HEADER.lower()
        )
        effective_query = {
            str(key): normalize_json_value(value)
            for key, value in client_config.default_query.items()
        }
        effective_query.update((str(key), normalize_json_value(value)) for key, value in query.items())
        scope_payload = {
            "base_url": client_config.base_url,
            "headers": sorted(effective_headers.items()),
            "query": sorted(effective_query.items()),
        }
        scope_digest = sha256(PrefixCacheManager._canonical_json(scope_payload).encode("utf-8")).hexdigest()
        fingerprint = {
            "version": 1,
            "scope": scope_digest,
            "model": model,
            "prefix_input": prefix_input,
            "tools": tools,
            "thinking": {"present": thinking_present, "value": thinking},
        }
        return sha256(PrefixCacheManager._canonical_json(fingerprint).encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _fresh_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
        result = {
            key: value
            for key, value in headers.items()
            if key.lower() != _ARK_REQUEST_ID_HEADER.lower()
        }
        result[_ARK_REQUEST_ID_HEADER] = str(uuid.uuid4())
        return result
