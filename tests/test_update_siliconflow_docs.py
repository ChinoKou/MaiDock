"""测试 SiliconFlow 中文文档同步脚本。"""

import json
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from update_siliconflow_docs import (
    MANIFEST_NAME,
    DocumentPlan,
    DownloadedDocument,
    SiliconFlowDocsError,
    SourceLocation,
    build_document_plans,
    convert_document_html,
    extract_catalog_tree,
    fetch_parsed,
    parse_source_url,
    safe_path_component,
    sync_downloads,
)


def _page_node(path: str, name: str, description: str = "") -> dict[str, object]:
    return {
        "$id": f"{path}.mdx",
        "type": "page",
        "name": name,
        "description": description,
        "url": f"/docs/{path}",
        "$ref": {"file": path},
    }


def _catalog_html() -> str:
    tree = {
        "$id": "cn",
        "name": "",
        "children": [
            {
                "type": "folder",
                "name": "使用指南",
                "children": [
                    _page_node("userguide/introduction", "平台简介"),
                    {"type": "separator", "name": "对话模型"},
                    _page_node("userguide/capabilities/stream-mode", "流式输出"),
                ],
            },
            {
                "type": "folder",
                "name": "API手册",
                "children": [
                    {"type": "separator", "name": "文本系列"},
                    _page_node("api/chat-completions-post", "$L14", "创建模型响应。"),
                ],
            },
            _page_node("release-notes/overview", "更新公告"),
            {"type": "separator", "name": "更多"},
            {
                "type": "page",
                "name": "SiliconFlow 官网",
                "url": "https://www.siliconflow.cn/",
                "external": True,
            },
        ],
    }
    chunk = f'5:["$",null,null,{{"tree":{json.dumps(tree, ensure_ascii=False)}}}]'
    flight = json.dumps([1, chunk], ensure_ascii=False)
    return f"""
    <html><body>
      <aside id="nd-sidebar">
        <a href="/docs/api/chat-completions-post">创建对话请求(OpenAI) <span>POST</span></a>
      </aside>
      <script>self.__next_f.push({flight})</script>
    </body></html>
    """


def _plan() -> DocumentPlan:
    return DocumentPlan(
        document_path="api/chat-completions-post",
        title="创建对话请求(OpenAI)",
        description="创建模型响应。",
        section="API手册",
        source_url="https://api-docs.siliconflow.cn/docs/api/chat-completions-post",
        relative_path=Path("2.API手册/1.文本系列/1.创建对话请求(OpenAI).md"),
    )


def _page_html() -> str:
    return """
    <html><body><article>
      <h1>创建对话请求(OpenAI)</h1>
      <div class="prose">
        <h2>Request Body</h2>
        <div role="tablist">
          <button role="tab" id="tab-llm">LLM</button>
          <button role="tab" id="tab-vlm">VLM</button>
        </div>
        <div role="tabpanel" aria-labelledby="tab-llm">
          <div class="text-sm border-t p-3 border-x bg-fd-card">
            <div class="flex flex-wrap items-center gap-2 not-prose">
              <span class="font-medium font-mono text-fd-primary">model</span>
              <span class="text-fd-muted-foreground px-2">string</span>
              <span><span>required</span></span>
            </div>
            <div>
              <p>对应的模型名称。</p>
              <div class="bg-fd-secondary rounded-lg text-xs">
                <span>Example</span><code>"deepseek-ai/DeepSeek-V3"</code>
              </div>
            </div>
          </div>
          <figure>
            <button aria-label="Copy Text"><svg></svg></button>
            <pre><code class="language-python"><span>print("ok")</span></code></pre>
          </figure>
        </div>
        <div role="tabpanel" aria-labelledby="tab-vlm" hidden="">
          <p>视觉模型参数。</p>
        </div>
        <div data-callout="info"><p>请妥善保管 API Key。</p></div>
        <details><summary>更多说明</summary><p>折叠内容。</p></details>
        <p>
          <a href="/docs/userguide/quickstart">快速上手</a>
          <img alt="示例图" src="/images/example.png" />
        </p>
        <video src="/media/example.mp4"></video>
        <form><select><option>界面选项</option></select><input value="secret" /></form>
      </div>
    </article></body></html>
    """


def _downloaded(plan: DocumentPlan, markdown: str) -> DownloadedDocument:
    normalized = markdown.rstrip() + "\n"
    return DownloadedDocument(
        plan=plan,
        document_title=plan.title,
        markdown=normalized,
        digest=sha256(normalized.encode()).hexdigest(),
    )


def test_parse_source_url() -> None:
    location = parse_source_url("https://api-docs.siliconflow.cn/docs/")

    assert location == SourceLocation(origin="https://api-docs.siliconflow.cn", docs_path="/docs")
    assert location.catalog_url == "https://api-docs.siliconflow.cn/docs"
    assert location.document_url("api/chat-completions-post").endswith("/docs/api/chat-completions-post")


def test_parse_source_url_rejects_other_paths() -> None:
    with pytest.raises(SiliconFlowDocsError, match="只支持"):
        parse_source_url("https://api-docs.siliconflow.cn/en/docs")


def test_extract_catalog_tree_and_build_numbered_paths() -> None:
    tree = extract_catalog_tree(_catalog_html())
    plans = build_document_plans(_catalog_html(), SourceLocation("https://api-docs.siliconflow.cn", "/docs"))

    assert tree["$id"] == "cn"
    assert [plan.document_path for plan in plans] == [
        "userguide/introduction",
        "userguide/capabilities/stream-mode",
        "api/chat-completions-post",
        "release-notes/overview",
    ]
    assert [plan.relative_path.as_posix() for plan in plans] == [
        "1.使用指南/1.平台简介.md",
        "1.使用指南/2.对话模型/1.流式输出.md",
        "2.API手册/1.文本系列/1.创建对话请求(OpenAI).md",
        "3.更新公告.md",
    ]
    assert plans[2].description == "创建模型响应。"


def test_build_document_plans_validates_sections() -> None:
    location = SourceLocation("https://api-docs.siliconflow.cn", "/docs")

    selected = build_document_plans(_catalog_html(), location, {"API手册"})
    assert [plan.document_path for plan in selected] == ["api/chat-completions-post"]
    with pytest.raises(SiliconFlowDocsError, match="不存在指定栏目"):
        build_document_plans(_catalog_html(), location, {"不存在"})


def test_convert_document_html_preserves_complex_content() -> None:
    document = convert_document_html(_page_html(), _plan())
    tick = chr(96)
    fence = tick * 3

    assert document.document_title == "创建对话请求(OpenAI)"
    assert document.markdown.startswith("# 创建对话请求(OpenAI)")
    assert "**选项：LLM**" in document.markdown
    assert "**选项：VLM**" in document.markdown
    assert f"{tick}model{tick} {tick}string{tick} **必填**" in document.markdown
    assert f'**Example：**{tick}"deepseek-ai/DeepSeek-V3"{tick}' in document.markdown
    assert f'{fence}python\nprint("ok")\n{fence}' in document.markdown
    assert "> 请妥善保管 API Key。" in document.markdown
    assert "**更多说明**" in document.markdown
    assert "https://api-docs.siliconflow.cn/docs/userguide/quickstart" in document.markdown
    assert "https://api-docs.siliconflow.cn/images/example.png" in document.markdown
    assert "[视频](https://api-docs.siliconflow.cn/media/example.mp4)" in document.markdown
    assert "<button" not in document.markdown
    assert "<svg" not in document.markdown
    assert "界面选项" not in document.markdown
    assert "secret" not in document.markdown


def test_convert_document_html_rejects_non_document() -> None:
    with pytest.raises(SiliconFlowDocsError, match="未找到 article"):
        convert_document_html("<html><body>首页</body></html>", _plan())


def test_convert_document_html_uses_longer_fence_for_nested_backticks() -> None:
    nested_fence = chr(96) * 3
    page_html = _page_html().replace('print("ok")', f'print("{nested_fence}python")')

    document = convert_document_html(page_html, _plan())

    outer_fence = chr(96) * 4
    assert f'{outer_fence}python\nprint("{nested_fence}python")\n{outer_fence}' in document.markdown


def test_fetch_parsed_retries_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        page_html = "<html><body>首页</body></html>" if request_count == 1 else _page_html()
        return httpx.Response(
            200,
            text=page_html,
            headers={"Content-Type": "text/html; charset=utf-8"},
            request=request,
        )

    monkeypatch.setattr("update_siliconflow_docs.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_parsed(
            client,
            _plan().source_url,
            attempts=2,
            context="测试",
            parser=lambda page_html: convert_document_html(page_html, _plan()),
        )

    assert result.document_title == "创建对话请求(OpenAI)"
    assert request_count == 2


def test_sync_downloads_covers_create_update_prune_and_local_change(
    tmp_path: Path,
) -> None:
    location = SourceLocation("https://api-docs.siliconflow.cn", "/docs")
    plan = _plan()

    first = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第一版")],
        {"API手册"},
        prune=False,
        allow_prune=True,
    )
    first_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    second = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第一版")],
        {"API手册"},
        prune=False,
        allow_prune=True,
    )
    second_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    third = sync_downloads(
        tmp_path,
        location,
        [plan],
        [_downloaded(plan, "第二版")],
        {"API手册"},
        prune=False,
        allow_prune=True,
    )
    target_path = tmp_path / plan.relative_path
    target_path.write_text("本地修改\n", encoding="utf-8")
    fourth = sync_downloads(
        tmp_path,
        location,
        [],
        [],
        {"API手册"},
        prune=True,
        allow_prune=True,
    )

    assert first.created == 1
    assert second.unchanged == 1
    assert second_manifest == first_manifest
    assert third.updated == 1
    assert fourth.preserved == 1
    assert target_path.read_text(encoding="utf-8") == "本地修改\n"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("创建:请求?", "创建：请求？"),
        ("CON", "_CON"),
        ("  多   空格  ", "多 空格"),
    ],
)
def test_safe_path_component(title: str, expected: str) -> None:
    assert safe_path_component(title) == expected
