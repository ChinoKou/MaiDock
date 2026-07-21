"""测试四个供应商文档同步器的代码常量入口。"""

import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import docs_sync_common as docs_sync_common_module
import httpx
import pytest
import update_bailian_docs as bailian
import update_siliconflow_docs as siliconflow
import update_volcengine_ark_docs as ark
import update_xiaomi_mimo_docs as mimo
from docs_sync_common import PROVIDER_DOCS_ROOT, SyncStats

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SYNC_MODULES: tuple[tuple[ModuleType, str, tuple[str, ...]], ...] = (
    (
        bailian,
        "dashscope",
        (
            "source_url",
            "fetch_origin",
            "output",
            "sections",
            "document_ids",
            "workers",
            "attempts",
            "timeout",
            "prune",
            "dry_run",
            "verbose",
        ),
    ),
    (
        siliconflow,
        "siliconflow",
        (
            "source_url",
            "output",
            "sections",
            "document_paths",
            "workers",
            "attempts",
            "timeout",
            "prune",
            "dry_run",
            "verbose",
        ),
    ),
    (
        ark,
        "volcengine_ark",
        (
            "source_url",
            "output",
            "sections",
            "document_ids",
            "workers",
            "attempts",
            "timeout",
            "prune",
            "dry_run",
            "verbose",
        ),
    ),
    (
        mimo,
        "xiaomi_mimo",
        (
            "source_url",
            "output",
            "documents",
            "workers",
            "attempts",
            "timeout",
            "prune",
            "dry_run",
            "verbose",
        ),
    ),
)


def _module_settings(module: ModuleType) -> dict[str, object]:
    """返回各同步器 main() 应传递给 run() 的模块常量。"""
    if module is bailian:
        return {
            "source_url": bailian.SOURCE_URL,
            "fetch_origin": bailian.FETCH_ORIGIN,
            "output": bailian.OUTPUT_DIRECTORY,
            "sections": bailian.SECTIONS,
            "document_ids": bailian.DOCUMENT_IDS,
            "workers": bailian.WORKERS,
            "attempts": bailian.ATTEMPTS,
            "timeout": bailian.TIMEOUT_SECONDS,
            "prune": bailian.PRUNE,
            "dry_run": bailian.DRY_RUN,
            "verbose": bailian.VERBOSE,
        }
    if module is siliconflow:
        return {
            "source_url": siliconflow.SOURCE_URL,
            "output": siliconflow.OUTPUT_DIRECTORY,
            "sections": siliconflow.SECTIONS,
            "document_paths": siliconflow.DOCUMENT_PATHS,
            "workers": siliconflow.WORKERS,
            "attempts": siliconflow.ATTEMPTS,
            "timeout": siliconflow.TIMEOUT_SECONDS,
            "prune": siliconflow.PRUNE,
            "dry_run": siliconflow.DRY_RUN,
            "verbose": siliconflow.VERBOSE,
        }
    if module is ark:
        return {
            "source_url": ark.SOURCE_URL,
            "output": ark.OUTPUT_DIRECTORY,
            "sections": ark.SECTIONS,
            "document_ids": ark.DOCUMENT_IDS,
            "workers": ark.WORKERS,
            "attempts": ark.ATTEMPTS,
            "timeout": ark.TIMEOUT_SECONDS,
            "prune": ark.PRUNE,
            "dry_run": ark.DRY_RUN,
            "verbose": ark.VERBOSE,
        }
    if module is mimo:
        return {
            "source_url": mimo.SOURCE_URL,
            "output": mimo.OUTPUT_DIRECTORY,
            "documents": mimo.DOCUMENTS,
            "workers": mimo.WORKERS,
            "attempts": mimo.ATTEMPTS,
            "timeout": mimo.TIMEOUT_SECONDS,
            "prune": mimo.PRUNE,
            "dry_run": mimo.DRY_RUN,
            "verbose": mimo.VERBOSE,
        }
    raise AssertionError(f"未知同步器模块：{module.__name__}")


@pytest.mark.parametrize(("module", "provider", "run_parameters"), SYNC_MODULES)
def test_sync_modules_are_direct_python_312_scripts(
    module: ModuleType,
    provider: str,
    run_parameters: tuple[str, ...],
) -> None:
    source = inspect.getsource(module)

    assert "argparse" not in source
    assert "from __future__ import annotations" not in source
    assert "SyncOptions" not in source
    assert "SYNC_OPTIONS" not in source
    assert "frozen=True" not in source
    assert "from .docs_sync_common" not in source
    assert tuple(inspect.signature(module.main).parameters) == ()
    assert tuple(inspect.signature(module.run).parameters) == run_parameters
    assert module.OUTPUT_DIRECTORY == PROVIDER_DOCS_ROOT / provider


def test_scripts_directory_is_not_a_python_package() -> None:
    assert not (SCRIPTS_ROOT / "__init__.py").exists()


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (
            lambda: bailian._validate_sync_settings(workers=0, attempts=1, timeout=1, document_ids=(), prune=False),
            bailian.BailianDocsError,
        ),
        (
            lambda: bailian._validate_sync_settings(workers=1, attempts=11, timeout=1, document_ids=(), prune=False),
            bailian.BailianDocsError,
        ),
        (
            lambda: bailian._validate_sync_settings(workers=1, attempts=1, timeout=0, document_ids=(), prune=False),
            bailian.BailianDocsError,
        ),
        (
            lambda: bailian._validate_sync_settings(workers=1, attempts=1, timeout=1, document_ids=(-1,), prune=False),
            bailian.BailianDocsError,
        ),
        (
            lambda: bailian._validate_sync_settings(workers=1, attempts=1, timeout=1, document_ids=(1,), prune=True),
            bailian.BailianDocsError,
        ),
        (
            lambda: siliconflow._validate_sync_settings(
                workers=9, attempts=1, timeout=1, document_paths=(), prune=False
            ),
            siliconflow.SiliconFlowDocsError,
        ),
        (
            lambda: siliconflow._validate_sync_settings(
                workers=1, attempts=0, timeout=1, document_paths=(), prune=False
            ),
            siliconflow.SiliconFlowDocsError,
        ),
        (
            lambda: siliconflow._validate_sync_settings(
                workers=1, attempts=1, timeout=-1, document_paths=(), prune=False
            ),
            siliconflow.SiliconFlowDocsError,
        ),
        (
            lambda: siliconflow._validate_sync_settings(
                workers=1,
                attempts=1,
                timeout=1,
                document_paths=("api/chat",),
                prune=True,
            ),
            siliconflow.SiliconFlowDocsError,
        ),
        (
            lambda: ark._validate_sync_settings(workers=17, attempts=1, timeout=1, document_ids=(), prune=False),
            ark.ArkDocsError,
        ),
        (
            lambda: ark._validate_sync_settings(workers=1, attempts=1, timeout=1, document_ids=(0,), prune=False),
            ark.ArkDocsError,
        ),
        (
            lambda: ark._validate_sync_settings(workers=1, attempts=1, timeout=1, document_ids=(1,), prune=True),
            ark.ArkDocsError,
        ),
        (
            lambda: mimo._validate_sync_settings(workers=1, attempts=0, timeout=1, documents=(), prune=False),
            mimo.MimoDocsError,
        ),
        (
            lambda: mimo._validate_sync_settings(workers=1, attempts=1, timeout=1, documents=("api/chat",), prune=True),
            mimo.MimoDocsError,
        ),
    ],
)
def test_sync_settings_reject_invalid_combinations(
    factory: Callable[[], object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        factory()


def test_bailian_run_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "dashscope"
    plan = bailian.DocumentPlan(
        node_id=1,
        node_type=1,
        title="测试",
        section="用户指南（模型）",
        alias="/model-studio/test",
        url_path="/zh/model-studio/test",
        source_url="https://help.aliyun.com/zh/model-studio/test",
        relative_path=Path("测试.md"),
    )
    monkeypatch.setattr(bailian, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(bailian, "fetch_catalog", lambda *_: {})
    monkeypatch.setattr(bailian, "build_document_plans", lambda *_: [plan])
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = bailian.run(
        source_url=bailian.SOURCE_URL,
        fetch_origin=bailian.FETCH_ORIGIN,
        output=output,
        sections=(),
        document_ids=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=False,
        dry_run=True,
        verbose=True,
    )

    assert result == 0
    assert not output.exists()


def test_siliconflow_run_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "siliconflow"
    plan = siliconflow.DocumentPlan(
        document_path="api/chat",
        title="测试",
        description="测试文档",
        section="API手册",
        source_url="https://api-docs.siliconflow.cn/docs/api/chat",
        relative_path=Path("测试.md"),
    )
    monkeypatch.setattr(siliconflow, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(siliconflow, "fetch_parsed", lambda **_: "catalog")
    monkeypatch.setattr(siliconflow, "build_document_plans", lambda *_: [plan])
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = siliconflow.run(
        source_url=siliconflow.SOURCE_URL,
        output=output,
        sections=(),
        document_paths=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=False,
        dry_run=True,
        verbose=True,
    )

    assert result == 0
    assert not output.exists()


def test_ark_run_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "volcengine_ark"
    plan = ark.DocumentPlan(document_id=1, title="测试", section="API参考", relative_path=Path("测试.md"))
    monkeypatch.setattr(ark, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(ark, "fetch_loader_data", lambda *_: {})
    monkeypatch.setattr(ark, "build_document_plans", lambda *_: [plan])
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = ark.run(
        source_url=ark.SOURCE_URL,
        output=output,
        sections=(),
        document_ids=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=False,
        dry_run=True,
        verbose=True,
    )

    assert result == 0
    assert not output.exists()


def test_mimo_run_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "xiaomi_mimo"
    plan = mimo.DocumentPlan(
        catalog_path="api/chat",
        catalog_title="Chat API",
        source_url="https://mimo.mi.com/docs/zh-CN/api/chat",
        relative_path=Path("api/chat.md"),
    )
    monkeypatch.setattr(mimo, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(mimo, "fetch_parsed", lambda *_: "catalog")
    monkeypatch.setattr(mimo, "parse_catalog", lambda *_: ([plan], []))
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = mimo.run(
        source_url=mimo.SOURCE_URL,
        output=output,
        documents=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=False,
        dry_run=True,
        verbose=True,
    )

    assert result == 0
    assert not output.exists()


@pytest.mark.parametrize(("module", "_provider", "_run_parameters"), SYNC_MODULES)
def test_main_delegates_to_module_constants(
    module: ModuleType,
    _provider: str,
    _run_parameters: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object]] = []

    def fake_run(**settings: object) -> int:
        received.append(settings)
        return 0

    monkeypatch.setattr(module, "run", fake_run)

    module.main()

    assert received == [_module_settings(module)]


def test_main_propagates_nonzero_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bailian, "run", lambda **_: 7)

    with pytest.raises(SystemExit) as exc_info:
        bailian.main()

    assert exc_info.value.code == 7


def test_bailian_partial_failure_disables_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plans = [
        bailian.DocumentPlan(
            node_id=node_id,
            node_type=1,
            title=f"测试 {node_id}",
            section="用户指南（模型）",
            alias=f"/model-studio/test-{node_id}",
            url_path=f"/zh/model-studio/test-{node_id}",
            source_url=f"https://help.aliyun.com/zh/model-studio/test-{node_id}",
            relative_path=Path(f"测试-{node_id}.md"),
        )
        for node_id in (1, 2)
    ]
    allow_prune: list[object] = []

    def fake_download(
        _client: httpx.Client,
        _location: bailian.SourceLocation,
        plan: bailian.DocumentPlan,
        _attempts: int,
    ) -> bailian.DownloadedDocument:
        if plan.node_id == 2:
            raise BailianTestError("下载失败")
        return bailian.DownloadedDocument(plan, plan.title, 1, "# 测试\n", "digest")

    def fake_sync(**kwargs: object) -> SyncStats:
        allow_prune.append(kwargs["allow_prune"])
        return SyncStats()

    monkeypatch.setattr(bailian, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(bailian, "fetch_catalog", lambda *_: {})
    monkeypatch.setattr(bailian, "build_document_plans", lambda *_: plans)
    monkeypatch.setattr(bailian, "download_document", fake_download)
    monkeypatch.setattr(bailian, "sync_downloads", fake_sync)
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = bailian.run(
        source_url=bailian.SOURCE_URL,
        fetch_origin=bailian.FETCH_ORIGIN,
        output=tmp_path / "dashscope",
        sections=(),
        document_ids=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=True,
        dry_run=False,
        verbose=False,
    )

    assert result == 1
    assert allow_prune == [False]
    assert "下载失败" in capsys.readouterr().err


def test_siliconflow_partial_failure_disables_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plans = [
        siliconflow.DocumentPlan(
            document_path=f"api/test-{index}",
            title=f"测试 {index}",
            description="测试文档",
            section="API手册",
            source_url=f"https://api-docs.siliconflow.cn/docs/api/test-{index}",
            relative_path=Path(f"测试-{index}.md"),
        )
        for index in (1, 2)
    ]
    allow_prune: list[object] = []

    def fake_download(
        _client: httpx.Client,
        plan: siliconflow.DocumentPlan,
        _attempts: int,
    ) -> siliconflow.DownloadedDocument:
        if plan.document_path.endswith("2"):
            raise SiliconFlowTestError("下载失败")
        return siliconflow.DownloadedDocument(plan, plan.title, "# 测试\n", "digest")

    def fake_sync(**kwargs: object) -> SyncStats:
        allow_prune.append(kwargs["allow_prune"])
        return SyncStats()

    monkeypatch.setattr(siliconflow, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(siliconflow, "fetch_parsed", lambda **_: "catalog")
    monkeypatch.setattr(siliconflow, "build_document_plans", lambda *_: plans)
    monkeypatch.setattr(siliconflow, "download_document", fake_download)
    monkeypatch.setattr(siliconflow, "sync_downloads", fake_sync)
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = siliconflow.run(
        source_url=siliconflow.SOURCE_URL,
        output=tmp_path / "siliconflow",
        sections=(),
        document_paths=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=True,
        dry_run=False,
        verbose=False,
    )

    assert result == 1
    assert allow_prune == [False]
    assert "下载失败" in capsys.readouterr().err


def test_ark_partial_failure_disables_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plans = [
        ark.DocumentPlan(
            document_id=document_id,
            title=f"测试 {document_id}",
            section="API参考",
            relative_path=Path(f"测试-{document_id}.md"),
        )
        for document_id in (1, 2)
    ]
    allow_prune: list[object] = []

    def fake_download(
        _client: httpx.Client,
        location: ark.SourceLocation,
        plan: ark.DocumentPlan,
        _attempts: int,
    ) -> ark.DownloadedDocument:
        if plan.document_id == 2:
            raise ArkTestError("下载失败")
        return ark.DownloadedDocument(
            plan,
            "# 测试\n",
            "2026-07-21T00:00:00Z",
            "digest",
            location.document_url(plan.document_id),
        )

    def fake_sync(**kwargs: object) -> SyncStats:
        allow_prune.append(kwargs["allow_prune"])
        return SyncStats()

    monkeypatch.setattr(ark, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(ark, "fetch_loader_data", lambda *_: {})
    monkeypatch.setattr(ark, "build_document_plans", lambda *_: plans)
    monkeypatch.setattr(ark, "download_document", fake_download)
    monkeypatch.setattr(ark, "sync_downloads", fake_sync)
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = ark.run(
        source_url=ark.SOURCE_URL,
        output=tmp_path / "volcengine_ark",
        sections=(),
        document_ids=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=True,
        dry_run=False,
        verbose=False,
    )

    assert result == 1
    assert allow_prune == [False]
    assert "下载失败" in capsys.readouterr().err


def test_mimo_partial_failure_disables_prune(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plans = [
        mimo.DocumentPlan(
            catalog_path=f"api/test-{index}",
            catalog_title=f"测试 {index}",
            source_url=f"https://mimo.mi.com/docs/zh-CN/api/test-{index}",
            relative_path=Path(f"api/test-{index}.md"),
        )
        for index in (1, 2)
    ]
    allow_prune: list[object] = []

    def fake_download(
        _client: httpx.Client,
        plan: mimo.DocumentPlan,
        _attempts: int,
    ) -> mimo.DownloadedDocument:
        if plan.catalog_path.endswith("2"):
            raise MimoTestError("下载失败")
        return mimo.DownloadedDocument(plan, plan.catalog_title, "# 测试\n", "digest")

    def fake_sync(**kwargs: object) -> SyncStats:
        allow_prune.append(kwargs["allow_prune"])
        return SyncStats()

    monkeypatch.setattr(mimo, "create_http_client", lambda *_: httpx.Client())
    monkeypatch.setattr(mimo, "fetch_parsed", lambda *_: "catalog")
    monkeypatch.setattr(mimo, "parse_catalog", lambda *_: (plans, []))
    monkeypatch.setattr(mimo, "download_document", fake_download)
    monkeypatch.setattr(mimo, "sync_downloads", fake_sync)
    monkeypatch.setattr(docs_sync_common_module, "PROJECT_ROOT", tmp_path)

    result = mimo.run(
        source_url=mimo.SOURCE_URL,
        output=tmp_path / "xiaomi_mimo",
        documents=(),
        workers=1,
        attempts=1,
        timeout=1,
        prune=True,
        dry_run=False,
        verbose=False,
    )

    assert result == 1
    assert allow_prune == [False]
    assert "下载失败" in capsys.readouterr().err


class BailianTestError(RuntimeError):
    """模拟百炼单篇文档下载失败。"""


class SiliconFlowTestError(RuntimeError):
    """模拟 SiliconFlow 单篇文档下载失败。"""


class ArkTestError(RuntimeError):
    """模拟火山方舟单篇文档下载失败。"""


class MimoTestError(RuntimeError):
    """模拟 MiMo 单篇文档下载失败。"""
