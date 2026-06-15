from pydantic import Field

from .base import HostDumpModel


class ProviderFunctionCall(HostDumpModel):
    """返回给 Host 的函数调用。"""

    name: str
    arguments: dict = Field(default_factory=dict)


class ProviderToolCall(HostDumpModel):
    """返回给 Host 的工具调用。"""

    id: str
    function: ProviderFunctionCall
    extra_content: dict = Field(default_factory=dict)


class ProviderUsage(HostDumpModel):
    """返回给 Host 的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0


def _empty_provider_tool_call_list() -> list[ProviderToolCall]:
    return []


class ProviderResponse(HostDumpModel):
    """LLM Provider 返回给 Host 的统一响应。"""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ProviderToolCall] = Field(default_factory=_empty_provider_tool_call_list)
    embedding: list[float] | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    raw_data: dict | None = None
