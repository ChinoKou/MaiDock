import logging

from src.core.common import ProviderRuntimeOptions
from src.providers.chat_completions_family.chat import ChatCompletionsMapper
from src.providers.common.parameter_translation import (
    TranslationContext,
    TranslationEnvelope,
)
from src.providers.siliconflow_provider import tools as siliconflow_tools
from src.providers.xiaomi_mimo_provider import tools as mimo_tools
from src.schemas import ResponseRequestSnapshot


def _response_request() -> ResponseRequestSnapshot:
    return ResponseRequestSnapshot.model_validate(
        {
            "model_info": {"model_identifier": "chat-model"},
            "api_provider": {"api_key": "test-key"},
            "message_list": [{"role": "user", "parts": [{"type": "text", "text": "你好"}]}],
        }
    )


class HookedMapper(ChatCompletionsMapper):
    """测试用 Mapper，记录 Provider 参数钩子是否被调用。"""

    def _apply_chat_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        envelope.body["hooked_model"] = context.model
        envelope.headers["X-Hook"] = "1"
        envelope.query["hooked"] = True


def test_chat_mapper_calls_provider_parameter_hook() -> None:
    mapper = HookedMapper(
        options=ProviderRuntimeOptions(),
        logger=logging.getLogger(__name__),
        provider_label="Hooked Chat",
        raw_provider="hooked",
        policy_provider="siliconflow",
    )

    body, headers, query = mapper.build_request_body(_response_request(), stream=False, apply_policy=False)

    assert body["model"] == "chat-model"
    assert body["hooked_model"] == "chat-model"
    assert headers == {"X-Hook": "1"}
    assert query == {"hooked": True}


def test_provider_tool_forwarders_preserve_provider_namespaces() -> None:
    raw_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]

    siliconflow_call = siliconflow_tools.extract_tool_calls(raw_tool_calls, options=ProviderRuntimeOptions())[0]
    mimo_call = mimo_tools.extract_tool_calls(raw_tool_calls, options=ProviderRuntimeOptions())[0]

    assert siliconflow_call.id == "call_1"
    assert siliconflow_call.extra_content == {
        "provider": "siliconflow",
        "siliconflow": {"raw_arguments": '{"q":"x"}'},
    }
    assert mimo_call.id == "call_1"
    assert mimo_call.extra_content == {
        "provider": "xiaomi_mimo",
        "xiaomi_mimo": {"raw_arguments": '{"q":"x"}'},
    }
