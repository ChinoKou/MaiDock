from ..i18n import translate


_PUBLIC_ERROR_KEYS = {
    "MEDIA_API_DISABLED": "runtime.media_error.disabled",
    "INVALID_REQUEST": "runtime.media_error.invalid_request",
    "INVALID_ARGUMENT": "runtime.media_error.invalid_request",
    "INVALID_BYTES": "runtime.media_error.invalid_request",
    "INVALID_OFFSET": "runtime.media_error.invalid_request",
    "INVALID_LENGTH": "runtime.media_error.invalid_request",
    "UPLOAD_TOO_LARGE": "runtime.media_error.upload_too_large",
    "CHUNK_TOO_LARGE": "runtime.media_error.upload_too_large",
    "UPLOAD_SIZE_MISMATCH": "runtime.media_error.upload_size_mismatch",
    "EMPTY_CHUNK": "runtime.media_error.upload_state",
    "UPLOAD_NOT_FOUND": "runtime.media_error.upload_not_found",
    "UPLOAD_ALREADY_COMPLETE": "runtime.media_error.upload_state",
    "UPLOAD_EXPIRED": "runtime.media_error.upload_state",
    "UPLOAD_INCOMPLETE": "runtime.media_error.upload_state",
    "UPLOAD_IN_USE": "runtime.media_error.upload_state",
    "UPLOAD_NOT_READY": "runtime.media_error.upload_state",
    "UPLOAD_UNSUPPORTED": "runtime.media_error.upload_unsupported",
    "UPLOAD_OFFSET_MISMATCH": "runtime.media_error.upload_offset",
    "UPLOAD_SHA256_MISMATCH": "runtime.media_error.upload_hash",
    "IDEMPOTENCY_CONFLICT": "runtime.media_error.idempotency_conflict",
    "QUEUE_FULL": "runtime.media_error.queue_full",
    "JOB_NOT_FOUND": "runtime.media_error.job_not_found",
    "JOB_NOT_TERMINAL": "runtime.media_error.job_not_terminal",
    "EXECUTION_PAYLOAD_MISSING": "runtime.media_error.store_corrupt",
    "STORE_CORRUPT": "runtime.media_error.store_corrupt",
    "ARTIFACT_TOO_LARGE": "runtime.media_error.artifact_too_large",
    "ARTIFACT_NOT_FOUND": "runtime.media_error.artifact_not_found",
    "ARTIFACT_EXPIRED": "runtime.media_error.artifact_expired",
    "STORAGE_QUOTA_EXCEEDED": "runtime.media_error.storage_quota",
    "PROFILE_REQUIRED": "runtime.media_error.profile_required",
    "PROFILE_API_KEY_MISSING": "runtime.media_error.profile_key_missing",
    "PROFILE_NOT_FOUND": "runtime.media_error.profile_not_found",
    "PROFILE_CHANGED": "runtime.media_error.profile_changed",
    "EXECUTION_UNCERTAIN": "runtime.media_error.execution_uncertain",
    "TASK_TRACKING_EXPIRED": "runtime.media_error.task_tracking_expired",
    "UPSTREAM_TASK_FAILED": "runtime.media_error.upstream_task_failed",
    "UPSTREAM_TASK_EXPIRED": "runtime.media_error.upstream_task_expired",
    "UPSTREAM_PROTOCOL_ERROR": "runtime.media_error.upstream_protocol",
    "UPSTREAM_UNAVAILABLE": "runtime.media_error.upstream_unavailable",
    "ARTIFACT_DOWNLOAD_FAILED": "runtime.media_error.artifact_download",
    "OSS_UPLOAD_FAILED": "runtime.media_error.oss_upload",
    "NO_MEDIA_OUTPUT": "runtime.media_error.no_media_output",
    "INTERNAL_ERROR": "runtime.media_error.internal",
}


def public_media_error_message(code: str) -> str:
    key = _PUBLIC_ERROR_KEYS.get(code)
    if key is None:
        return translate("runtime.media_error.upstream", code=code)
    return translate(key)


class MediaApiError(RuntimeError):
    """跨插件 Public API 的稳定业务错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        uncertain: bool = False,
        provider_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.uncertain = uncertain
        self.provider_request_id = provider_request_id
