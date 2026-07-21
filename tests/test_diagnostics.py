from src.core.diagnostics import sanitize_for_log


def test_sanitize_for_log_reports_memoryview_nbytes_without_copy() -> None:
    value = memoryview(bytearray(8)).cast("H")

    assert sanitize_for_log(value) == "<bytes:8>"
