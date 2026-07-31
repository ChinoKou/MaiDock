from dataclasses import dataclass, field
from typing import Literal

from ..i18n import translate
from .json_types import JsonValue

type ProviderPolicyKey = Literal[
    "openai_responses",
    "anthropic_messages",
    "volcengine_ark",
    "dashscope",
    "bailian_responses",
    "siliconflow",
    "xiaomi_mimo",
]
type CapabilityKey = Literal[
    "response",
    "chat_completion",
    "embeddings",
    "audio_transcription",
]

_CAPABILITY_KEYS: tuple[CapabilityKey, ...] = (
    "response",
    "chat_completion",
    "embeddings",
    "audio_transcription",
)
_PROVIDER_POLICY_KEYS: tuple[ProviderPolicyKey, ...] = (
    "openai_responses",
    "anthropic_messages",
    "volcengine_ark",
    "dashscope",
    "bailian_responses",
    "siliconflow",
    "xiaomi_mimo",
)


@dataclass(frozen=True, slots=True)
class ParameterOverrideSet:
    """已解析的目录覆写值。

    只保存经过类型解析的非空覆写条目，key 为参数目录中的规范参数名。
    空白覆写（表示不覆写）不会进入此集合。
    """

    values: dict[str, JsonValue] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.values)

    def __contains__(self, key: str) -> bool:
        return key in self.values


@dataclass(slots=True)
class ProviderCapabilityOverrides:
    """Provider 所暴露所有能力的覆写集合。"""

    response: ParameterOverrideSet = field(default_factory=ParameterOverrideSet)
    chat_completion: ParameterOverrideSet = field(default_factory=ParameterOverrideSet)
    embeddings: ParameterOverrideSet = field(default_factory=ParameterOverrideSet)
    audio_transcription: ParameterOverrideSet = field(default_factory=ParameterOverrideSet)

    def get(self, capability: CapabilityKey) -> ParameterOverrideSet:
        match capability:
            case "response":
                return self.response
            case "chat_completion":
                return self.chat_completion
            case "embeddings":
                return self.embeddings
            case "audio_transcription":
                return self.audio_transcription
        raise ValueError(
            translate(
                "runtime.error.capability_policy_unsupported",
                value=repr(capability),
                allowed=", ".join(_CAPABILITY_KEYS),
            )
        )


@dataclass(slots=True)
class ParameterOverrideRegistry:
    """Provider/能力维度的参数覆写注册表。"""

    openai_responses: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    anthropic_messages: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    dashscope: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    bailian_responses: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    siliconflow: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    volcengine_ark: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)
    xiaomi_mimo: ProviderCapabilityOverrides = field(default_factory=ProviderCapabilityOverrides)

    def get(self, provider: ProviderPolicyKey, capability: CapabilityKey) -> ParameterOverrideSet:
        match provider:
            case "openai_responses":
                return self.openai_responses.get(capability)
            case "anthropic_messages":
                return self.anthropic_messages.get(capability)
            case "dashscope":
                return self.dashscope.get(capability)
            case "bailian_responses":
                return self.bailian_responses.get(capability)
            case "siliconflow":
                return self.siliconflow.get(capability)
            case "volcengine_ark":
                return self.volcengine_ark.get(capability)
            case "xiaomi_mimo":
                return self.xiaomi_mimo.get(capability)
        raise ValueError(
            translate(
                "runtime.error.provider_policy_unsupported",
                value=repr(provider),
                allowed=", ".join(_PROVIDER_POLICY_KEYS),
            )
        )
