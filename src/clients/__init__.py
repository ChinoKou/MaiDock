"""供应商 Client 与协议资源。"""

from .common import HttpConnection, RetryPolicy, SharedHttpClient

__all__ = ["HttpConnection", "RetryPolicy", "SharedHttpClient"]
