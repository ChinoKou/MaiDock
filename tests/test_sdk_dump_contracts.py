import pytest
from pydantic import BaseModel

from src.schemas import SdkDumpAdapter


class ExamplePydanticModel(BaseModel):
    visible: str
    omitted: str | None = None


class ExampleModelDumpable:
    def model_dump(self, mode: str = "python") -> object:
        assert mode == "python"
        return {"dumped": (1, 2)}


class ExampleToDict:
    def to_dict(self) -> object:
        return {"converted": ExamplePydanticModel(visible="yes")}


class ExampleStringFallback:
    def __str__(self) -> str:
        return "string-fallback"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param("text", id="string"),
        pytest.param(7, id="integer"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="boolean"),
    ],
)
def test_sdk_dump_preserves_json_scalars(value: object) -> None:
    assert SdkDumpAdapter.to_plain(value) == value


def test_sdk_dump_serializes_pydantic_and_model_dumpable_objects() -> None:
    assert SdkDumpAdapter.to_plain(ExamplePydanticModel(visible="yes")) == {"visible": "yes"}
    assert SdkDumpAdapter.to_plain(ExampleModelDumpable()) == {"dumped": (1, 2)}


def test_sdk_dump_recursively_serializes_mapping_and_tuple() -> None:
    value = {
        7: ExamplePydanticModel(visible="mapped"),
        "items": (ExamplePydanticModel(visible="tuple"), {8: False}),
    }

    assert SdkDumpAdapter.to_plain(value) == {
        "7": {"visible": "mapped"},
        "items": [{"visible": "tuple"}, {"8": False}],
    }


def test_sdk_dump_uses_to_dict_before_string_fallback() -> None:
    assert SdkDumpAdapter.to_plain(ExampleToDict()) == {"converted": {"visible": "yes"}}
    assert SdkDumpAdapter.to_plain(ExampleStringFallback()) == "string-fallback"


def test_sdk_dump_plain_dict_accepts_mapping_sources() -> None:
    assert SdkDumpAdapter.to_plain_dict(ExamplePydanticModel(visible="yes")) == {"visible": "yes"}
    assert SdkDumpAdapter.to_plain_dict(ExampleToDict()) == {"converted": {"visible": "yes"}}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, 2], id="list"),
        pytest.param("scalar", id="scalar"),
    ],
)
def test_sdk_dump_plain_dict_rejects_non_objects(value: object) -> None:
    with pytest.raises(TypeError, match="(mapping|映射)"):
        SdkDumpAdapter.to_plain_dict(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param((1, ExamplePydanticModel(visible="yes")), [1, {"visible": "yes"}], id="tuple"),
        pytest.param([{"nested": True}], [{"nested": True}], id="list"),
    ],
)
def test_sdk_dump_plain_list_accepts_json_iterables(value: object, expected: list[object]) -> None:
    assert SdkDumpAdapter.to_plain_list(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"not": "a-list"}, id="mapping"),
        pytest.param(42, id="scalar"),
    ],
)
def test_sdk_dump_plain_list_rejects_non_lists(value: object) -> None:
    with pytest.raises(TypeError, match="(list|列表)"):
        SdkDumpAdapter.to_plain_list(value)
