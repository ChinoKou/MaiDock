from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from ..core.json_types import JsonValue, is_json_mapping, mapping_to_json_object, value_to_json_object
from ..i18n import runtime_expected, runtime_subject, translate

type OpenAITextVerbosity = Literal["low", "medium", "high"]
type AnthropicImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

_DICT_ADAPTER: TypeAdapter[dict] = TypeAdapter(dict)


@runtime_checkable
class ModelDumpable(Protocol):
    """Pydantic-like 响应对象协议。"""

    def model_dump(self, mode: str = "python") -> object:
        """导出模型对象。"""


class HostDumpModel(BaseModel):
    """可以安全导出给 Host 的模型基类。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def to_host_dict(self) -> dict[str, JsonValue]:
        # mode="json" 保证 dump 出来只有 JSON 值，value_to_json_object 因此是恒等变换，
        # 但它给出的是 dict[str, JsonValue] 而不是 TypeAdapter(dict) 的 dict[Unknown, Unknown]，
        # 这样 RPC 边界的返回类型才能真正落地。
        return value_to_json_object(self.model_dump(mode="json", exclude_none=True, by_alias=True))


class IgnoreExtraModel(BaseModel):
    """Host 快照和上游响应都可能新增字段，读取时忽略未知项。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, arbitrary_types_allowed=True)


class ObjectFields(IgnoreExtraModel):
    """显式 object 包装，避免宽泛类型在 Pydantic 边界散落。"""

    fields: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])

    @classmethod
    def from_unknown(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if isinstance(value, ObjectFields):
            return cls(fields=value.to_plain_dict())
        if isinstance(value, BaseModel):
            return cls(fields=_DICT_ADAPTER.validate_python(value.model_dump(mode="json", exclude_none=True)))
        if isinstance(value, ModelDumpable):
            return cls(fields=_DICT_ADAPTER.validate_python(value.model_dump(mode="python")))
        if is_json_mapping(value):
            return cls(fields=_DICT_ADAPTER.validate_python(mapping_to_json_object(value)))
        if value is None:
            return cls()
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject=runtime_subject("value"),
                expected=runtime_expected("mapping"),
                actual=type(value).__name__,
            )
        )

    def to_plain_dict(self) -> dict[str, JsonValue]:
        return dict(self.fields)

    @field_validator("fields", mode="before")
    @classmethod
    def validate_fields(cls, value: object) -> dict[str, JsonValue]:
        if is_json_mapping(value):
            # mapping_to_json_object 已经返回全新的纯 JSON dict，再过一遍 TypeAdapter(dict)
            # 只是又浅拷贝一次，却把静态类型退化成 dict[Unknown, Unknown]。
            return mapping_to_json_object(value)
        if value is None:
            return {}
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="ObjectFields.fields",
                expected=runtime_expected("mapping"),
                actual=type(value).__name__,
            )
        )


def default_tool_parameters() -> ObjectFields:
    return ObjectFields(fields={"type": "object", "properties": {}})


def normalize_tool_parameters(value: object) -> ObjectFields:
    parsed = ObjectFields.from_unknown(value)
    if not parsed.fields:
        return default_tool_parameters()
    return parsed
