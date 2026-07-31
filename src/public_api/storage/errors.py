class PublicApiStorageError(RuntimeError):
    """Public API 持久化边界的稳定错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
