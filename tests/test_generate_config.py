"""测试配置模板生成脚本。"""

from pathlib import Path

import generate_config as generate_config_module
import tomllib

from generate_config import generate_config_toml, write_generated_config


def test_generated_config_contains_public_api_structure() -> None:
    generated = generate_config_toml()
    config = tomllib.loads(generated)

    assert config["public_api"]["enabled"] is False
    assert config["public_api"]["resources"]["max_concurrent_jobs"] == 2
    assert config["public_api"]["dashscope"]["profiles"] == []
    assert config["public_api"]["volcengine_ark"]["profiles"] == []


def test_every_public_api_vendor_gets_its_own_section() -> None:
    """供应商小节必须按贡献目录动态展开。

    写死名字时，新增的一家不会只是"少一个小节"：它的整个配置对象会以
    `name = ""` 的形式泄漏进 [public_api] 顶层，生成出无法解析回原结构的配置。
    """

    from src.public_api.providers import PUBLIC_API_CONFIG_CATALOG

    config = tomllib.loads(generate_config_toml())
    public_api = config["public_api"]

    for contribution in PUBLIC_API_CONFIG_CATALOG:
        vendor_key = contribution.config_path.removeprefix("public_api.")
        assert isinstance(public_api.get(vendor_key), dict), vendor_key
        assert public_api[vendor_key]["profiles"] == []


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
