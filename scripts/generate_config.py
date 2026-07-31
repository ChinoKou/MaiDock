import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "config.toml"

type JsonValue = str | int | float | bool | None | dict[str, JsonValue] | list[JsonValue]


def _ensure_project_root() -> None:
    """确保直接执行脚本时可以导入 MaiDock 源码。"""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _toml_value(value: JsonValue | list[JsonValue]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = ", ".join(_toml_value(item) for item in value)
        return f"[{items}]"
    return '""'


def _write_section(lines: list[str], path: str, data: Mapping[str, JsonValue]) -> None:
    lines.append(f"[{path}]")
    for key, value in data.items():
        lines.append(f"{key} = {_toml_value(value)}")


def _write_pairs(lines: list[str], path: str, pairs: list[tuple[str, JsonValue]]) -> None:
    lines.append(f"[{path}]")
    for key, value in pairs:
        lines.append(f"{key} = {_toml_value(value)}")


def _provider_base_pairs(
    raw: Mapping[str, JsonValue], capability_order: tuple[str, ...]
) -> list[tuple[str, JsonValue]]:
    sub_sections = set(capability_order)
    return [(str(k), v) for k, v in raw.items() if k not in sub_sections]


def generate_config_toml() -> str:
    """生成完整 config.toml 并返回字符串。"""
    _ensure_project_root()

    from src.config import normalize_maidock_config_data
    from src.core.json_types import json_mapping_or_none, mapping_to_json_object
    from src.core.parameter_catalog import _CAPABILITY_ORDER, _PROVIDER_ORDER
    from src.public_api.providers import PUBLIC_API_CONFIG_CATALOG
    from src.version import __version__

    normalized, _ = normalize_maidock_config_data({})
    lines: list[str] = []

    plugin = mapping_to_json_object(json_mapping_or_none(normalized.get("plugin")) or {})
    plugin["config_version"] = __version__
    _write_section(lines, "plugin", plugin)
    lines.append("")

    _write_section(
        lines,
        "diagnostics",
        mapping_to_json_object(json_mapping_or_none(normalized.get("diagnostics")) or {}),
    )
    lines.append("")

    public_api = mapping_to_json_object(json_mapping_or_none(normalized.get("public_api")) or {})
    # 供应商小节按贡献目录动态展开，而不是写死名字：漏掉一家的后果不是少一个小节，
    # 而是它会以 `name = ""` 的形式泄漏进 [public_api] 顶层，生成出无法解析的配置。
    vendor_keys = tuple(
        contribution.config_path.removeprefix("public_api.") for contribution in PUBLIC_API_CONFIG_CATALOG
    )
    nested_keys = {"resources", *vendor_keys}
    _write_pairs(
        lines,
        "public_api",
        [(str(key), value) for key, value in public_api.items() if key not in nested_keys],
    )
    lines.append("")
    _write_section(
        lines,
        "public_api.resources",
        mapping_to_json_object(json_mapping_or_none(public_api.get("resources")) or {}),
    )
    lines.append("")
    for vendor_key in vendor_keys:
        vendor_section = mapping_to_json_object(json_mapping_or_none(public_api.get(vendor_key)) or {})
        _write_section(lines, f"public_api.{vendor_key}", vendor_section)
        lines.append("")

    for provider in _PROVIDER_ORDER:
        provider_raw = mapping_to_json_object(json_mapping_or_none(normalized.get(provider)) or {})
        _write_pairs(lines, provider, _provider_base_pairs(provider_raw, _CAPABILITY_ORDER))
        lines.append("")

        for capability in _CAPABILITY_ORDER:
            cap_raw = mapping_to_json_object(json_mapping_or_none(provider_raw.get(capability)) or {})
            if not cap_raw:
                continue
            cap_path = f"{provider}.{capability}"
            overrides_raw = mapping_to_json_object(json_mapping_or_none(cap_raw.get("overrides")) or {})
            _write_pairs(
                lines,
                cap_path,
                [(str(k), v) for k, v in cap_raw.items() if k != "overrides"],
            )
            lines.append("")
            _write_section(lines, f"{cap_path}.overrides", overrides_raw)
            lines.append("")

    compatibility = mapping_to_json_object(json_mapping_or_none(normalized.get("compatibility")) or {})
    _write_section(lines, "compatibility", compatibility)

    return "\n".join(lines) + "\n"


def write_generated_config(output_path: Path) -> Path:
    """将生成的配置写入指定路径并返回解析后的路径。"""
    resolved = output_path.resolve()
    resolved.write_text(generate_config_toml(), encoding="utf-8")
    return resolved


def main() -> None:
    """使用模块级代码常量重新生成配置模板。"""
    output_path = write_generated_config(OUTPUT_PATH)
    print(f"已写入：{output_path}")


if __name__ == "__main__":
    main()
