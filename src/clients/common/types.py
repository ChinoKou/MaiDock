"""Client 层自有的 JSON 类型。"""

type JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
type JsonObject = dict[str, JsonValue]
