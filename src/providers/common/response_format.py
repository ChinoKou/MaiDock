from ...schemas import ResponseFormatSnapshot


def normalize_response_format_snapshot(value: object) -> ResponseFormatSnapshot:
    if isinstance(value, ResponseFormatSnapshot):
        return value
    return ResponseFormatSnapshot.model_validate(value)
