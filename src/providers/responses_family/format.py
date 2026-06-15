from ...schemas import OpenAITextConfig, OpenAITextFormatConfig, ResponseRequestSnapshot
from ..common.response_format import build_responses_text_format_payload


def build_responses_text_config(request: ResponseRequestSnapshot) -> OpenAITextConfig | None:
    """将 Host response_format 转换为 OpenAI Responses text 配置。"""

    return build_responses_text_config_from_value(request.response_format)


def build_responses_text_config_from_value(response_format: object) -> OpenAITextConfig | None:
    """将 response_format 快照或值转换为 OpenAI Responses text 配置。"""

    format_payload = build_responses_text_format_payload(response_format, provider_label="OpenAI Responses")
    if format_payload is None:
        return None
    return OpenAITextConfig(format=OpenAITextFormatConfig.model_validate(format_payload))
