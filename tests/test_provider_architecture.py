import ast
import logging
from pathlib import Path
from typing import Protocol, cast

import pytest

from src.core.common import ProviderRuntimeOptions
from src.providers.openai_responses_provider import multimodal as openai_multimodal
from src.providers.openai_responses_provider import responses as openai_responses
from src.providers.openai_responses_provider import tools as openai_tools
from src.providers.siliconflow_provider import chat as siliconflow_chat
from src.providers.siliconflow_provider import multimodal as siliconflow_multimodal
from src.providers.siliconflow_provider import tools as siliconflow_tools
from src.providers.volcengine_ark_provider import multimodal as ark_multimodal
from src.providers.volcengine_ark_provider import responses as ark_responses
from src.providers.volcengine_ark_provider import tools as ark_tools
from src.providers.xiaomi_mimo_provider import chat as mimo_chat
from src.providers.xiaomi_mimo_provider import multimodal as mimo_multimodal
from src.providers.xiaomi_mimo_provider import tools as mimo_tools

PROVIDERS_ROOT = Path(__file__).resolve().parents[1] / "src" / "providers"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILY_PROVIDER_NAMES = {
    "openai_responses_provider",
    "siliconflow_provider",
    "volcengine_ark_provider",
    "xiaomi_mimo_provider",
}
FAMILY_NAMES = {
    "chat_completions_family",
    "openai_auxiliary_family",
    "responses_family",
}


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


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.level, node.module or ""))
        elif isinstance(node, ast.Import):
            imports.extend((0, alias.name) for alias in node.names)
    return imports


def _python_files(directory_name: str) -> list[Path]:
    return sorted((PROVIDERS_ROOT / directory_name).glob("*.py"))


def test_production_code_uses_package_relative_imports() -> None:
    paths = [PROJECT_ROOT / "plugin.py", *sorted((PROJECT_ROOT / "src").rglob("*.py"))]
    for path in paths:
        for level, module in _imports(path):
            assert not (level == 0 and (module == "src" or module.startswith("src."))), path


def test_family_provider_dependency_direction() -> None:
    for provider_name in FAMILY_PROVIDER_NAMES:
        for path in _python_files(provider_name):
            for level, module in _imports(path):
                assert not (level == 2 and (module == "common" or module.startswith("common."))), path
                assert not module.startswith("src.providers.common"), path

    for path in _python_files("common"):
        for _level, module in _imports(path):
            assert not any(name in module for name in FAMILY_NAMES | FAMILY_PROVIDER_NAMES), path

    for family_name in FAMILY_NAMES:
        for path in _python_files(family_name):
            for _level, module in _imports(path):
                assert not any(name in module for name in FAMILY_PROVIDER_NAMES), path


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
