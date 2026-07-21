"""测试配置模板生成脚本。"""

from pathlib import Path

import generate_config as generate_config_module
from generate_config import OUTPUT_PATH, generate_config_toml, write_generated_config


def test_generated_config_matches_checked_in_template() -> None:
    assert generate_config_toml() == OUTPUT_PATH.read_text(encoding="utf-8")


def test_write_generated_config_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "config.toml"

    first_path = write_generated_config(output)
    first_content = output.read_bytes()
    second_path = write_generated_config(output)

    assert first_path == output.resolve()
    assert second_path == output.resolve()
    assert output.read_bytes() == first_content
    assert output.read_text(encoding="utf-8") == generate_config_toml()


def test_main_uses_module_output_path(monkeypatch, tmp_path: Path, capsys) -> None:
    output = tmp_path / "generated.toml"
    monkeypatch.setattr(generate_config_module, "OUTPUT_PATH", output)

    generate_config_module.main()

    assert output.read_text(encoding="utf-8") == generate_config_toml()
    assert str(output.resolve()) in capsys.readouterr().out
