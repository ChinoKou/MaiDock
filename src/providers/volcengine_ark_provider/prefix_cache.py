"""Volcengine ARK Responses API 前缀缓存管理器。

自动管理"首次创建 → 后续复用"的有状态前缀缓存。
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("maibot_plugin.maidock.volcengine_ark.prefix_cache")


class PrefixCacheManager:
    """管理 Volcengine ARK Responses API 的前缀缓存生命周期。

    核心逻辑：
    - 首次调用：发送 caching:{type:"enabled", prefix:true}，存储返回的 response id
    - 后续调用：发送 previous_response_id:<cached_id>，不发送 caching
    - 过期前自动重建缓存
    """

    _instances: dict[str, "PrefixCacheManager"] = {}

    def __init__(
        self,
        cache_id_path: str = "data/volcengine_prefix_cache.json",
        ttl_seconds: int = 7 * 24 * 3600,  # 7 days
        renew_before_seconds: int = 24 * 3600,  # 1 day before expiry
    ):
        self.cache_id_path = Path(cache_id_path)
        self.ttl_seconds = ttl_seconds
        self.renew_before_seconds = renew_before_seconds
        self._lock = asyncio.Lock()
        self._state: dict[str, dict] = {}  # model_name → {cache_id, expire_at, prefix_hash}
        self._loaded = False

    @classmethod
    def get_instance(cls, cache_id_path: str = "data/volcengine_prefix_cache.json") -> "PrefixCacheManager":
        """获取单例（按文件路径区分）。"""
        if cache_id_path not in cls._instances:
            cls._instances[cache_id_path] = cls(cache_id_path=cache_id_path)
        return cls._instances[cache_id_path]

    def _load(self):
        """从磁盘加载缓存状态。"""
        if self._loaded:
            return
        self._loaded = True
        try:
            if self.cache_id_path.exists():
                data = json.loads(self.cache_id_path.read_text())
                self._state = data.get("caches", {})
                logger.info(f"前缀缓存状态已加载: {len(self._state)} 个条目")
        except Exception as e:
            logger.warning(f"加载前缀缓存状态失败: {e}，使用空状态")
            self._state = {}

    def _save(self):
        """原子写入缓存状态到磁盘。"""
        try:
            self.cache_id_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_id_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({"caches": self._state, "updated_at": time.time()}))
            os.rename(tmp_path, self.cache_id_path)
        except Exception as e:
            logger.warning(f"保存前缀缓存状态失败: {e}")

    def _make_prefix_hash(self, messages: list) -> str:
        """计算消息列表的 system prompt 前缀哈希。"""
        system_text = ""
        for msg in messages:
            if hasattr(msg, "role") and getattr(msg, "role", "") == "system":
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    system_text += json.dumps(content, sort_keys=True)
                else:
                    system_text += str(content)
                break  # 只取第一个 system 消息
        return hashlib.sha256(system_text.encode()).hexdigest()[:16]

    async def resolve(self, model: str, messages: list) -> dict:
        """为当前请求解析缓存参数。

        Returns:
            dict: 应合并到请求体的缓存相关参数。
            可能是 {"caching": {"type": "enabled", "prefix": true}}
            或 {"previous_response_id": "<cached_id>"}
            或 {} （未启用缓存管理）
        """
        self._load()

        prefix_hash = self._make_prefix_hash(messages)
        cached = self._state.get(model)

        async with self._lock:
            # 检查是否需要重建：无缓存 / 过期 / prefix 变了
            need_create = (
                cached is None
                or cached.get("prefix_hash") != prefix_hash
                or cached.get("expire_at", 0) < time.time() + self.renew_before_seconds
            )

            if need_create:
                # 需要创建缓存 — 但不能在实际 API 调用前创建
                # 返回 caching 参数，实际缓存 ID 在 API 响应后存储
                logger.info(
                    f"前缀缓存: 需要创建 model={model} prefix_hash={prefix_hash}"
                    + (f" (新)" if cached is None else f" (过期/变化, 旧hash={cached.get('prefix_hash', 'N/A')[:8]}...新={prefix_hash[:8]}...)")
                )
                self._state[model] = {
                    "prefix_hash": prefix_hash,
                    "pending": True,  # 标记为待确认
                }
                self._save()
                return {"caching": {"type": "enabled", "prefix": True}}

            # 使用已有缓存
            logger.debug(f"前缀缓存: 复用 model={model} cache_id={cached['cache_id'][:24]}...")
            return {"previous_response_id": cached["cache_id"]}

    async def confirm(self, model: str, response_id: str, expire_at: Optional[int] = None):
        """API 调用成功后，确认缓存已创建，存储 cache_id。

        Args:
            model: 模型名称
            response_id: API 返回的 response id
            expire_at: 缓存过期时间戳，默认为当前时间 + ttl
        """
        async with self._lock:
            entry = self._state.get(model, {})
            if not entry.get("pending"):
                return  # 不是我们创建的

            entry["cache_id"] = response_id
            entry["expire_at"] = expire_at or (time.time() + self.ttl_seconds)
            entry.pop("pending", None)
            self._state[model] = entry
            self._save()
            logger.info(f"前缀缓存已确认: model={model} cache_id={response_id[:24]}... expire_at={entry['expire_at']}")

    async def invalidate(self, model: str):
        """使指定模型的缓存失效。"""
        async with self._lock:
            if model in self._state:
                del self._state[model]
                self._save()
                logger.info(f"前缀缓存已失效: model={model}")
