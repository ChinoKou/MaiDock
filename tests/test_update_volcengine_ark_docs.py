"""测试火山方舟文档同步脚本。"""

import json
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from update_volcengine_ark_docs import (
    MANIFEST_NAME,
    ArkDocsError,
    DocumentPlan,
    DownloadedDocument,
    SourceLocation,
    build_document_plans,
    extract_downloaded_document,
    extract_loader_data,
    fetch_loader_data,
    parse_source_url,
    safe_path_component,
    sync_downloads,
)


def _catalog_data() -> dict[str, object]:
    return {
        "curLib": {
            "SecondNav": [
                {"ID": 969, "Name": "API参考"},
                {"ID": 747, "Name": "文档指南"},
            ]
        },
        "docListMap": {
            "969": {
                "0": {"children": [100, 103]},
                "100": {
                    "value": {
                        "DocumentID": 100,
                        "Title": "Responses/API",
                        "Type": 1,
                        "Status": 1,
                    },
                    "children": [101, 102],
                },
                "101": {
                    "value": {
                        "DocumentID": 101,
                        "Title": "创建：响应？",
                        "Type": 0,
                        "Status": 2,
                    },
                    "children": [],
                },
                "102": {
                    "value": {
                        "DocumentID": 102,
                        "Title": "草稿",
                        "Type": 0,
                        "Status": 5,
                    },
                    "children": [],
                },
                "103": {
                    "value": {
                        "DocumentID": 103,
                        "Title": "错误码",
                        "Type": 0,
                        "Status": 2,
                    },
                    "children": [],
                },
            },
            "747": {
                "0": {"children": [201]},
                "201": {
                    "value": {
                        "DocumentID": 201,
                        "Title": "快速入门",
                        "Type": 0,
                        "Status": 2,
                    },
                    "children": [],
                },
            },
        },
    }


def _document_data(document_id: int = 101, title: str = "创建：响应？") -> dict[str, object]:
    return {
        "curDoc": {
            "DocumentID": document_id,
            "Title": title,
            "MDContent": "# 示例\n\n正文\n\n",
            "UpdatedTime": "2026-07-18T00:00:00Z",
        }
    }


def _downloaded_document(plan: DocumentPlan, markdown: str) -> DownloadedDocument:
    normalized = markdown.rstrip() + "\n"
    return DownloadedDocument(
        plan=plan,
        markdown=normalized,
        updated_time="2026-07-18T00:00:00Z",
        digest=sha256(normalized.encode()).hexdigest(),
        source_url=f"https://docs.volcengine.com/docs/82379/{plan.document_id}?lang=zh",
    )


def test_parse_source_url_normalizes_library_url() -> None:
    location = parse_source_url("https://docs.volcengine.com/docs/82379/1099455?lang=zh")

    assert location == SourceLocation(origin="https://docs.volcengine.com", library_id=82379, lang="zh")
    assert location.catalog_url == "https://docs.volcengine.com/docs/82379?lang=zh"
    assert location.document_url(101) == "https://docs.volcengine.com/docs/82379/101?lang=zh"
    assert "__loader=docs%2F%28libid%29%2Flayout" in location.catalog_data_url
    assert "__loader=docs%2F%28libid%29%2F%28docid%24%29%2Fpage" in location.document_data_url(101)


def test_extract_loader_data_uses_json_decoder() -> None:
    expected = _catalog_data()

    assert extract_loader_data(json.dumps(expected, ensure_ascii=False), "测试") == expected


def test_extract_loader_data_rejects_html_shell() -> None:
    with pytest.raises(ArkDocsError, match="未返回有效 JSON"):
        extract_loader_data("<html>空壳页面</html>", "测试")


def test_build_document_plans_preserves_tree_order_and_skips_unpublished() -> None:
    plans = build_document_plans(_catalog_data())

    assert [(plan.document_id, plan.relative_path.as_posix()) for plan in plans] == [
        (101, "API参考/1.Responses／API/1.创建：响应？.md"),
        (103, "API参考/2.错误码.md"),
        (201, "文档指南/1.快速入门.md"),
    ]


def test_build_document_plans_filters_sections() -> None:
    plans = build_document_plans(_catalog_data(), {"API参考"})

    assert {plan.section for plan in plans} == {"API参考"}
    assert [plan.document_id for plan in plans] == [101, 103]


def test_build_document_plans_rejects_unknown_section() -> None:
    with pytest.raises(ArkDocsError, match="找不到栏目"):
        build_document_plans(_catalog_data(), {"不存在"})


def test_extract_downloaded_document_normalizes_trailing_newline() -> None:
    plan = DocumentPlan(
        document_id=101,
        title="创建：响应？",
        section="API参考",
        relative_path=Path("API参考/1.创建：响应？.md"),
    )

    document = extract_downloaded_document(_document_data(), plan, "https://example.test/101")

    assert document.markdown == "# 示例\n\n正文\n"
    assert document.updated_time == "2026-07-18T00:00:00Z"
    assert document.digest == sha256(document.markdown.encode()).hexdigest()


def test_extract_downloaded_document_rejects_stale_catalog_title() -> None:
    plan = DocumentPlan(101, "旧标题", "API参考", Path("旧标题.md"))

    with pytest.raises(ArkDocsError, match="目录标题"):
        extract_downloaded_document(_document_data(), plan, "https://example.test/101")


def test_fetch_loader_data_retries_html_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, text="<html>空壳页面</html>", request=request)
        return httpx.Response(200, json=_catalog_data(), request=request)

    monkeypatch.setattr("update_volcengine_ark_docs.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_loader_data(client, "https://example.test/docs/82379", attempts=2, context="测试")

    assert result == _catalog_data()
    assert request_count == 2


def test_fetch_loader_data_retries_transient_405(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(405, request=request)
        return httpx.Response(200, json=_catalog_data(), request=request)

    monkeypatch.setattr("update_volcengine_ark_docs.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_loader_data(client, "https://example.test/docs/82379", attempts=2, context="测试")

    assert result == _catalog_data()
    assert request_count == 2


def test_sync_downloads_covers_create_unchanged_update_and_prune(
    tmp_path: Path,
) -> None:
    location = SourceLocation("https://docs.volcengine.com", 82379, "zh")
    plan = DocumentPlan(101, "示例", "API参考", Path("API参考/1.示例.md"))

    first = sync_downloads(
        output_root=tmp_path,
        location=location,
        plans=[plan],
        downloads=[_downloaded_document(plan, "第一版")],
        selected_sections={"API参考"},
        prune=False,
        allow_prune=True,
    )
    first_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    second = sync_downloads(
        output_root=tmp_path,
        location=location,
        plans=[plan],
        downloads=[_downloaded_document(plan, "第一版")],
        selected_sections={"API参考"},
        prune=False,
        allow_prune=True,
    )
    second_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    third = sync_downloads(
        output_root=tmp_path,
        location=location,
        plans=[plan],
        downloads=[_downloaded_document(plan, "第二版")],
        selected_sections={"API参考"},
        prune=False,
        allow_prune=True,
    )
    fourth = sync_downloads(
        output_root=tmp_path,
        location=location,
        plans=[],
        downloads=[],
        selected_sections={"API参考"},
        prune=True,
        allow_prune=True,
    )

    assert first.created == 1
    assert second.unchanged == 1
    assert second_manifest == first_manifest
    assert third.updated == 1
    assert fourth.removed == 1
    assert not (tmp_path / plan.relative_path).exists()


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ('A<B>C:D"E/F\\G|H?I*J', "A＜B＞C：D＂E／F＼G｜H？I＊J"),
        ("CON", "_CON"),
        (" . ", "未命名"),
    ],
)
def test_safe_path_component(title: str, expected: str) -> None:
    assert safe_path_component(title) == expected
