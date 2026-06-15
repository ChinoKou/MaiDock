from ..common.response_format import build_chat_response_format_payload


def build_chat_response_format_body(response_format: object) -> dict | None:
    """将 Host response_format 转换为 Chat Completions response_format body。"""
    return build_chat_response_format_payload(response_format, provider_label="Chat Completions")
