from pydantic import BaseModel, TypeAdapter

from ..core.json_types import is_json_iterable, is_json_mapping, mapping_to_json_object
from .base import ModelDumpable

_DICT_ADAPTER: TypeAdapter[dict] = TypeAdapter(dict)
_LIST_ADAPTER: TypeAdapter[list] = TypeAdapter(list)


class SdkDumpAdapter:
    """把 Pydantic/JSON-like 响应规整为普通 JSON-like 结构。"""

    @staticmethod
    def to_plain(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, ModelDumpable):
            return value.model_dump(mode="python")
        if is_json_mapping(value):
            return {str(key): SdkDumpAdapter.to_plain(item) for key, item in value.items()}
        if is_json_iterable(value):
            return [SdkDumpAdapter.to_plain(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return SdkDumpAdapter.to_plain(to_dict())
        return str(value)

    @staticmethod
    def to_plain_dict(value: object) -> dict:
        plain = SdkDumpAdapter.to_plain(value)
        if is_json_mapping(plain):
            return _DICT_ADAPTER.validate_python(mapping_to_json_object(plain))
        raise TypeError(f"不支持的响应对象类型: {type(value).__name__}")

    @staticmethod
    def to_plain_list(value: object) -> list:
        plain = SdkDumpAdapter.to_plain(value)
        if isinstance(plain, list):
            return _LIST_ADAPTER.validate_python(plain)
        raise TypeError(f"不支持的列表对象类型: {type(value).__name__}")
