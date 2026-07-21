from ...core.diagnostics import sanitize_for_log
from ...i18n import translate
from ..common.httpx import HttpxProviderError


class DashScopeApiError(HttpxProviderError):
    """保留 DashScope 原生错误字段的 Provider 异常。"""

    def __init__(
        self,
        message: str,
        *,
        code: str | None,
        upstream_message: str | None,
        request_id: str | None,
        status_code: int | None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.code = code
        self.upstream_message = upstream_message
        self.request_id = request_id

    @property
    def is_endpoint_mismatch(self) -> bool:
        code = (self.code or "").lower()
        message = (self.upstream_message or "").lower()
        return "invalidparameter" in code and "url error" in message


def dashscope_error_factory(
    payload: dict,
    status_code: int | None,
    event_name: str | None,
) -> DashScopeApiError | None:
    """把 HTTP JSON 与 SSE 错误统一为 DashScopeApiError。"""

    raw_code = payload.get("code")
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else None
    raw_message = payload.get("message")
    upstream_message = raw_message if isinstance(raw_message, str) and raw_message else None
    payload_status = payload.get("status_code")
    effective_status = status_code
    if effective_status is None and isinstance(payload_status, int):
        effective_status = payload_status

    has_error_code = code is not None and code.lower() not in {"success", "ok"}
    has_error_status = effective_status is not None and not 200 <= effective_status < 300
    if event_name != "error" and not has_error_code and not has_error_status:
        return None

    raw_request_id = payload.get("request_id") or payload.get("requestId")
    request_id = raw_request_id if isinstance(raw_request_id, str) and raw_request_id else None
    details = {
        "status_code": effective_status,
        "request_id": request_id,
        "code": code,
        "message": upstream_message,
    }
    message = translate(
        "runtime.error.upstream_status",
        provider="阿里云百炼 DashScope",
        details=sanitize_for_log(details),
    )
    return DashScopeApiError(
        message,
        code=code,
        upstream_message=upstream_message,
        request_id=request_id,
        status_code=effective_status,
    )


def raise_for_dashscope_error(payload: dict) -> None:
    """检查已经取得的 DashScope JSON 响应。"""

    error = dashscope_error_factory(payload, None, None)
    if error is not None:
        raise error
