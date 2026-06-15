from pydantic import Field, field_validator

from .base import IgnoreExtraModel, ObjectFields


class GenericUsageSnapshot(IgnoreExtraModel):
    """多上游 usage 字段的宽松读取模型。"""

    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    input_tokens_details: ObjectFields = Field(default_factory=ObjectFields)
    prompt_tokens_details: ObjectFields = Field(default_factory=ObjectFields)

    @field_validator("input_tokens_details", "prompt_tokens_details", mode="before")
    @classmethod
    def validate_details(cls, value: object) -> ObjectFields:
        return ObjectFields.from_unknown(value)
