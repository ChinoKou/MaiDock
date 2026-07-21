"""测试小米 MiMo 中文文档同步脚本。"""

import json
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from update_xiaomi_mimo_docs import (
    MANIFEST_NAME,
    DocumentPlan,
    DownloadedDocument,
    MimoDocsError,
    SourceLocation,
    convert_document_html,
    fetch_parsed,
    parse_catalog,
    parse_source_url,
    sync_downloads,
)

CATALOG = """\
# MiMo

- [First API Call](https://mimo.mi.com/static/docs/quick-start/summary/first-api-call.md)
- [Models](https://mimo.mi.com/static/docs/quick-start/summary/model.md)
- [Service Agreement](https://mimo.mi.com/static/docs/quick-start/terms/user-agreement.md)
"""


def _plan() -> DocumentPlan:
    return DocumentPlan(
        catalog_path="quick-start/summary/first-api-call",
        catalog_title="First API Call",
        source_url="https://mimo.mi.com/docs/zh-CN/quick-start/summary/first-api-call",
        relative_path=Path("quick-start/summary/first-api-call.md"),
    )


def _page_html() -> str:
    schema = json.dumps(
        [
            {
                "name": "messages",
                "type": "array",
                "required": True,
                "description": "消息列表。",
                "children": [{"name": "role", "type": "string", "description": "角色。"}],
            }
        ],
        ensure_ascii=False,
    )
    return f"""
    <html><body><div class="mdxContent">
      <h1>首次调用 API</h1>
      <div class="mdx-highlight mdx-highlight-info"><p>请妥善保管密钥。</p></div>
      <mimo-tab><mimo-tab-item label="Python SDK">
        <mimo-code-block><pre class="language-python" raw="print%28%27%E4%BD%A0%E5%A5%BD%27%29"><code>高亮内容</code></pre></mimo-code-block>
      </mimo-tab-item></mimo-tab>
      <inline-schema-v2 schema='{schema}'></inline-schema-v2>
      <table><thead><tr><th>字段</th><th>说明</th></tr></thead><tbody><tr><td>model</td><td>模型</td></tr></tbody></table>
      <p><a href="/docs/zh-CN/api">接口</a><img alt="图" src="/static/example.png" /></p>
      <mimo-audio src="/static/example.mp3" title="示例.wav"></mimo-audio>
      <video src="/static/example.mp4" title="示例.mp4"></video>
    </div></body></html>
    """


def _downloaded(plan: DocumentPlan, markdown: str) -> DownloadedDocument:
    normalized = markdown.rstrip() + "\n"
    return DownloadedDocument(
        plan=plan,
        title="首次调用 API",
        markdown=normalized,
        digest=sha256(normalized.encode()).hexdigest(),
    )


def test_parse_source_url() -> None:
    location = parse_source_url("https://mimo.mi.com/docs/zh-CN")

    assert location == SourceLocation(origin="https://mimo.mi.com", lang="zh-CN")
    assert location.catalog_url == "https://mimo.mi.com/llms.txt"
    assert location.document_url("tokenplan/Token Plan/subscription").endswith(
        "/docs/zh-CN/tokenplan/Token%20Plan/subscription"
    )


def test_parse_catalog_builds_paths_and_excludes_unavailable_terms() -> None:
    plans, excluded = parse_catalog(CATALOG, SourceLocation("https://mimo.mi.com", "zh-CN"))

    assert [plan.catalog_path for plan in plans] == [
        "quick-start/summary/first-api-call",
        "quick-start/summary/model",
    ]
    assert plans[0].relative_path.as_posix() == "quick-start/summary/first-api-call.md"
    assert excluded == ["quick-start/terms/user-agreement"]


def test_parse_catalog_rejects_non_mimo_document() -> None:
    catalog = "- [Bad](https://example.com/static/docs/bad.md)"

    with pytest.raises(MimoDocsError, match="非本站文档"):
        parse_catalog(catalog, SourceLocation("https://mimo.mi.com", "zh-CN"))


def test_convert_document_html_preserves_complex_content() -> None:
    document = convert_document_html(_page_html(), _plan())

    assert document.title == "首次调用 API"
    assert document.markdown.startswith("# 首次调用 API")
    assert "> 请妥善保管密钥。" in document.markdown
    assert "**Python SDK**" in document.markdown
    assert "```python\nprint('你好')\n```" in document.markdown
    assert "`messages` `array` **必填**" in document.markdown
    assert "`role` `string`" in document.markdown
    assert "| 字段 | 说明 |" in document.markdown
    assert "https://mimo.mi.com/docs/zh-CN/api" in document.markdown
    assert "https://mimo.mi.com/static/example.png" in document.markdown
    assert "[音频：示例.wav](https://mimo.mi.com/static/example.mp3)" in document.markdown
    assert "[视频：示例.mp4](https://mimo.mi.com/static/example.mp4)" in document.markdown
    assert "<inline-schema-v2" not in document.markdown
    assert "<mimo-code-block" not in document.markdown


def test_convert_document_html_rejects_home_page() -> None:
    with pytest.raises(MimoDocsError, match="mdxContent"):
        convert_document_html("<html><body>首页</body></html>", _plan())


def test_fetch_parsed_retries_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(200, text="<html>首页</html>", request=request)
        return httpx.Response(200, text=_page_html(), request=request)

    monkeypatch.setattr("update_xiaomi_mimo_docs.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_parsed(
            client,
            _plan().source_url,
            attempts=2,
            context="测试",
            parser=lambda page_html: convert_document_html(page_html, _plan()),
        )

    assert result.title == "首次调用 API"
    assert request_count == 2


def test_sync_downloads_covers_create_unchanged_update_and_prune(
    tmp_path: Path,
) -> None:
    location = SourceLocation("https://mimo.mi.com", "zh-CN")
    plan = _plan()

    first = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第一版")],
        [],
        prune=False,
        allow_prune=True,
    )
    first_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    second = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第一版")],
        [],
        prune=False,
        allow_prune=True,
    )
    second_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    third = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第二版")],
        [],
        prune=False,
        allow_prune=True,
    )
    fourth = sync_downloads(
        tmp_path,
        location,
        [],
        [],
        [],
        prune=True,
        allow_prune=True,
    )

    assert first.created == 1
    assert second.unchanged == 1
    assert second_manifest == first_manifest
    assert third.updated == 1
    assert fourth.removed == 1
    assert not (tmp_path / plan.relative_path).exists()
