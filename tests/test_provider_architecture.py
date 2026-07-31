import ast
import logging
from pathlib import Path
from typing import Protocol, cast

import pytest

from src.core.common import ProviderRuntimeOptions
from src.config_schema import build_maidock_config_schema
from src.host_adapters.openai_responses_provider import multimodal as openai_multimodal
from src.host_adapters.openai_responses_provider import responses as openai_responses
from src.host_adapters.openai_responses_provider import tools as openai_tools
from src.host_adapters.siliconflow_provider import chat as siliconflow_chat
from src.host_adapters.siliconflow_provider import multimodal as siliconflow_multimodal
from src.host_adapters.siliconflow_provider import tools as siliconflow_tools
from src.host_adapters.volcengine_ark_provider import multimodal as ark_multimodal
from src.host_adapters.volcengine_ark_provider import responses as ark_responses
from src.host_adapters.volcengine_ark_provider import tools as ark_tools
from src.host_adapters.xiaomi_mimo_provider import chat as mimo_chat
from src.host_adapters.xiaomi_mimo_provider import multimodal as mimo_multimodal
from src.host_adapters.xiaomi_mimo_provider import tools as mimo_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CLIENTS_ROOT = SRC_ROOT / "clients"
CORE_ROOT = SRC_ROOT / "core"
HOST_ADAPTERS_ROOT = SRC_ROOT / "host_adapters"
PUBLIC_API_ROOT = SRC_ROOT / "public_api"


class _ToolHooks(Protocol):
    def _convert_tools(self, tool_options: list[object]) -> object: ...


class _ResponseParameterHooks(Protocol):
    def _apply_response_parameters(self, context: object, envelope: object) -> None: ...


class _ChatParameterHooks(Protocol):
    def _apply_chat_parameters(self, context: object, envelope: object) -> None: ...


class _ResponseMultimodalHooks(Protocol):
    def _extract_text_content(self, response: object) -> str: ...


class _ChatMultimodalHooks(Protocol):
    def _message_content_text(self, content: object) -> str | None: ...


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _public_api_vendor_packages() -> tuple[str, ...]:
    """public_api/providers 下的供应商包名（providers/common 是供应商无关的共享层）。"""
    root = PUBLIC_API_ROOT / "providers"
    return tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and path.name != "common" and (path / "__init__.py").exists()
        )
    )


_BARE_CONTAINER_NAMES = frozenset(
    {"dict", "list", "tuple", "set", "frozenset", "Mapping", "Sequence", "Iterable", "Iterator"}
)


def _annotations(tree: ast.AST) -> list[ast.expr]:
    """收集全部注解表达式：变量注解、返回注解、各类形参注解与 type 别名右值。"""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            found.append(node.annotation)
        elif isinstance(node, ast.TypeAlias):
            # `type X = dict` 的右值就是类型表达式，裸容器藏在这里同样是转义舱口。
            found.append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                found.append(node.returns)
            arguments = node.args
            for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                if arg.annotation is not None:
                    found.append(arg.annotation)
            for arg in (arguments.vararg, arguments.kwarg):
                if arg is not None and arg.annotation is not None:
                    found.append(arg.annotation)
    return found


def _bare_container_hits(path: Path) -> list[str]:
    """注解里出现的未参数化容器。

    `dict[str, JsonValue]` 里的 `dict` 是 Subscript 的基名，不算裸容器；
    只有单独出现的 `dict` / `Mapping` 这类才算——它们等价于把值类型退化成 Unknown。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for annotation in _annotations(tree):
        parameterized = {
            id(node.value)
            for node in ast.walk(annotation)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        }
        for node in ast.walk(annotation):
            if isinstance(node, ast.Name) and node.id in _BARE_CONTAINER_NAMES and id(node) not in parameterized:
                hits.append(f"{path}:{node.lineno} {node.id}")
    return hits


def _host_adapter_packages(suffix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in HOST_ADAPTERS_ROOT.iterdir()
            if path.is_dir() and path.name.endswith(suffix) and (path / "__init__.py").exists()
        )
    )


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.level, node.module or ""))
        elif isinstance(node, ast.Import):
            imports.extend((0, alias.name) for alias in node.names)
    return imports


def test_production_code_uses_package_relative_imports() -> None:
    paths = [PROJECT_ROOT / "plugin.py", *_python_sources(SRC_ROOT)]
    for path in paths:
        for level, module in _imports(path):
            assert not (level == 0 and (module == "src" or module.startswith("src."))), path


def test_client_layer_is_self_contained() -> None:
    """Client 层是插件内最底层：只依赖标准库、httpx 和自己，不认识上面任何一层。"""
    forbidden_parts = {
        "config",
        "core",
        "host_adapters",
        "i18n",
        "media",
        "public_api",
        "runtime",
        "schemas",
    }
    for path in _python_sources(CLIENTS_ROOT):
        for _level, module in _imports(path):
            assert forbidden_parts.isdisjoint(module.split(".")), path


def test_host_adapters_do_not_depend_on_public_api() -> None:
    """两条上层通路彼此独立：Host 通路不得把 Public API 当基础设施。"""
    for path in _python_sources(HOST_ADAPTERS_ROOT):
        for _level, module in _imports(path):
            assert "public_api" not in module.split("."), path


def test_public_api_never_depends_on_host_layer() -> None:
    """Public API 通路同样不得反向依赖 Host 通路或宿主 Schema。"""
    for path in _python_sources(PUBLIC_API_ROOT):
        for _level, module in _imports(path):
            parts = set(module.split("."))
            assert "host_adapters" not in parts, path
            assert "schemas" not in parts, path


def test_source_tree_has_no_type_checker_escape_hatches() -> None:
    paths = [PROJECT_ROOT / "plugin.py", *_python_sources(SRC_ROOT)]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "type: ignore" not in source, path
        assert "pyright: ignore" not in source, path


def test_client_families_do_not_depend_on_vendor_clients() -> None:
    vendor_modules = {"anthropic", "ark", "dashscope", "mimo", "openai", "siliconflow"}
    for path in sorted((CLIENTS_ROOT / "families").glob("*.py")):
        for _level, module in _imports(path):
            assert vendor_modules.isdisjoint(module.split(".")), path


def test_only_runtime_ingress_depends_on_sdk_provider_base() -> None:
    ingress_path = SRC_ROOT / "runtime" / "ingress.py"
    for path in _python_sources(SRC_ROOT):
        if "LLMProviderBase" in path.read_text(encoding="utf-8"):
            assert path == ingress_path


def test_legacy_provider_modules_are_removed() -> None:
    assert not (SRC_ROOT / "providers").exists()
    assert not list(HOST_ADAPTERS_ROOT.rglob("provider.py"))


def test_top_level_media_layer_is_removed() -> None:
    assert not (SRC_ROOT / "media").exists()


def test_public_api_common_components_have_no_vendor_dependency() -> None:
    vendor_packages = set(_public_api_vendor_packages())
    assert vendor_packages, "public_api/providers 下至少应有一个供应商包"
    common_paths = [
        PUBLIC_API_ROOT / "catalog.py",
        *_python_sources(PUBLIC_API_ROOT / "api"),
        *_python_sources(PUBLIC_API_ROOT / "application"),
        *_python_sources(PUBLIC_API_ROOT / "domain"),
        *_python_sources(PUBLIC_API_ROOT / "storage"),
    ]
    for path in common_paths:
        for _level, module in _imports(path):
            parts = set(module.split("."))
            assert vendor_packages.isdisjoint(parts), path
            assert "providers" not in parts, path


def test_public_api_providers_are_mutually_isolated() -> None:
    """供应商包之间彼此不可见，任何共享都必须下沉到 providers/common。"""
    providers_root = PUBLIC_API_ROOT / "providers"
    vendor_packages = set(_public_api_vendor_packages())
    assert vendor_packages
    for vendor in sorted(vendor_packages):
        others = vendor_packages - {vendor}
        for path in _python_sources(providers_root / vendor):
            for _level, module in _imports(path):
                assert others.isdisjoint(module.split(".")), path


def test_public_api_provider_common_stays_vendor_agnostic() -> None:
    common_root = PUBLIC_API_ROOT / "providers" / "common"
    assert (common_root / "__init__.py").exists()
    vendor_packages = set(_public_api_vendor_packages())
    for path in _python_sources(common_root):
        for _level, module in _imports(path):
            assert vendor_packages.isdisjoint(module.split(".")), path


def test_public_api_config_and_runtime_catalogs_are_distinct() -> None:
    from src.public_api.providers import PUBLIC_API_CONFIG_CATALOG, PUBLIC_PROVIDER_CONTRIBUTIONS

    assert PUBLIC_API_CONFIG_CATALOG is not PUBLIC_PROVIDER_CONTRIBUTIONS


@pytest.mark.parametrize(
    "root",
    [CORE_ROOT, CLIENTS_ROOT, HOST_ADAPTERS_ROOT, PUBLIC_API_ROOT],
    ids=["core", "clients", "host_adapters", "public_api"],
)
def test_layer_has_no_untyped_escape_hatches(root: Path) -> None:
    """四条业务层都不得用 Any / cast / 宽泛值类型 / 裸容器绕开类型检查。

    这四项是同一个问题的四张面孔：都把某处类型退化成"什么都行"，于是转译链路里
    形状写错也不会在 pyright 阶段暴露。零豁免是有意的——需要窄化就用
    core/json_types.py 里的运行时校验，而不是断言。
    """
    for path in _python_sources(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Name) and node.id == "Any"), path
            # `typing.Any` 这种带模块前缀的写法绕得开上面的 Name 检查，这里补上。
            assert not (isinstance(node, ast.Attribute) and node.attr == "Any" and isinstance(node.value, ast.Name)), (
                path
            )
            if isinstance(node, ast.Call):
                # `typing.cast(...)` 属性调用形态与裸 `cast(...)` 一视同仁：
                # 任何 `.cast(...)` 在这四层都按转义舱口处理，零豁免比放过一个
                # 真正的 cast 更划算——真有同名方法时让它改名。
                func = node.func
                is_cast = (isinstance(func, ast.Name) and func.id == "cast") or (
                    isinstance(func, ast.Attribute) and func.attr == "cast"
                )
                assert not is_cast, path
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "dict"
                and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 2
            ):
                continue
            value_type = node.slice.elts[1]
            assert not (isinstance(value_type, ast.Name) and value_type.id in {"Any", "object"}), path
        assert not _bare_container_hits(path), _bare_container_hits(path)


def test_host_adapter_common_layer_has_no_upward_dependency() -> None:
    """common/** 是两条 family 与六家 provider 的共同下游，不得反向依赖任何一方。"""
    upward = {*_host_adapter_packages("_provider"), *_host_adapter_packages("_family")}
    assert upward, "host_adapters 下应同时存在 *_provider 与 *_family 包"
    for path in _python_sources(HOST_ADAPTERS_ROOT / "common"):
        for _level, module in _imports(path):
            assert upward.isdisjoint(module.split(".")), path


def test_host_adapter_families_do_not_depend_on_vendor_providers() -> None:
    """协议族只描述协议，具体供应商差异留在 provider 包里。"""
    vendor_packages = set(_host_adapter_packages("_provider"))
    assert vendor_packages
    for family in _host_adapter_packages("_family"):
        for path in _python_sources(HOST_ADAPTERS_ROOT / family):
            for _level, module in _imports(path):
                assert vendor_packages.isdisjoint(module.split(".")), path


def test_public_api_literal_error_codes_are_localized() -> None:
    from src.public_api.errors import _PUBLIC_ERROR_KEYS

    error_types = {"MediaApiError", "MediaError", "PublicApiStorageError"}
    codes: set[str] = set()
    for path in _python_sources(PUBLIC_API_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in error_types and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    codes.add(first.value)
            for keyword in node.keywords:
                if (
                    keyword.arg == "error_code"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    codes.add(keyword.value.value)
    assert codes
    unmapped = codes - set(_PUBLIC_ERROR_KEYS)
    assert not unmapped, unmapped


def test_removed_public_api_prototype_paths_do_not_return() -> None:
    for relative_path in (
        "contracts.py",
        "facade.py",
        "job_engine.py",
        "schemas.py",
        "store.py",
        "dashscope",
    ):
        assert not (PUBLIC_API_ROOT / relative_path).exists()


def test_plugin_registers_dynamic_public_api_without_media_adapter() -> None:
    plugin_source = (SRC_ROOT / "plugin.py").read_text(encoding="utf-8")
    assert "@API" not in plugin_source
    assert "public=True" in plugin_source.replace(" ", "")
    assert "register_dynamic_api" in plugin_source
    production_source = "\n".join(path.read_text(encoding="utf-8") for path in _python_sources(SRC_ROOT))
    assert "media_adapter" not in production_source


def test_public_api_has_no_auth_contract_or_password_control() -> None:
    public_source = "\n".join(path.read_text(encoding="utf-8") for path in _python_sources(PUBLIC_API_ROOT))
    assert "access_token" not in public_source.lower()
    assert "unauthorized" not in public_source.lower()
    schema = build_maidock_config_schema()
    dashscope_fields = schema["sections"]["public_api_dashscope"]["fields"]["profiles"]["item_fields"]
    assert dashscope_fields["api_key"]["ui_type"] == "text"
    assert dashscope_fields["api_key"]["input_type"] is None


@pytest.mark.parametrize(
    ("mapper", "tools_module", "adapter_module", "parameter_name"),
    [
        (
            openai_responses.create_responses_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.openai.adapter"),
            ),
            openai_tools,
            openai_responses,
            "apply_openai_responses_parameters",
        ),
        (
            ark_responses.create_responses_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.ark.adapter"),
            ),
            ark_tools,
            ark_responses,
            "apply_ark_responses_parameters",
        ),
        (
            siliconflow_chat._create_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.siliconflow.adapter"),
            ),
            siliconflow_tools,
            siliconflow_chat,
            "apply_siliconflow_chat_parameters",
        ),
        (
            mimo_chat._create_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.mimo.adapter"),
            ),
            mimo_tools,
            mimo_chat,
            "apply_mimo_chat_parameters",
        ),
    ],
)
def test_provider_mapper_uses_tools_and_parameter_facades(
    monkeypatch: pytest.MonkeyPatch,
    mapper: object,
    tools_module: object,
    adapter_module: object,
    parameter_name: str,
) -> None:
    sentinel_tools = [cast(object, {"provider_facade": True})]
    parameter_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(tools_module, "convert_tools", lambda _tools: sentinel_tools)
    monkeypatch.setattr(
        adapter_module,
        parameter_name,
        lambda context, envelope: parameter_calls.append((context, envelope)),
    )

    assert cast(_ToolHooks, mapper)._convert_tools([]) is sentinel_tools
    context = object()
    envelope = object()
    if hasattr(mapper, "_apply_response_parameters"):
        cast(_ResponseParameterHooks, mapper)._apply_response_parameters(context, envelope)
    else:
        cast(_ChatParameterHooks, mapper)._apply_chat_parameters(context, envelope)
    assert parameter_calls == [(context, envelope)]


@pytest.mark.parametrize(
    ("mapper", "multimodal_module", "response_mapper"),
    [
        (
            openai_responses.create_responses_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.openai.multimodal"),
            ),
            openai_multimodal,
            True,
        ),
        (
            ark_responses.create_responses_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.ark.multimodal"),
            ),
            ark_multimodal,
            True,
        ),
        (
            siliconflow_chat._create_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.siliconflow.multimodal"),
            ),
            siliconflow_multimodal,
            False,
        ),
        (
            mimo_chat._create_mapper(
                options=ProviderRuntimeOptions(),
                logger=logging.getLogger("test.mimo.multimodal"),
            ),
            mimo_multimodal,
            False,
        ),
    ],
)
def test_provider_mapper_uses_multimodal_facade(
    monkeypatch: pytest.MonkeyPatch,
    mapper: object,
    multimodal_module: object,
    response_mapper: bool,
) -> None:
    if response_mapper:
        monkeypatch.setattr(
            multimodal_module,
            "extract_text_content",
            lambda _response: "provider-response",
        )
        assert cast(_ResponseMultimodalHooks, mapper)._extract_text_content(object()) == "provider-response"
    else:
        monkeypatch.setattr(multimodal_module, "message_content_text", lambda _content: "provider-chat")
        assert cast(_ChatMultimodalHooks, mapper)._message_content_text(object()) == "provider-chat"
