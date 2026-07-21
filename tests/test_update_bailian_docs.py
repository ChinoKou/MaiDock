"""测试阿里云百炼中文文档同步脚本。"""

import copy
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from update_bailian_docs import (
    MANIFEST_NAME,
    SECTION_SPECS,
    BailianDocsError,
    DocumentPlan,
    DownloadedDocument,
    SourceLocation,
    build_document_plans,
    convert_document_html,
    extract_downloaded_document,
    fetch_parsed,
    parse_source_url,
    safe_path_component,
    sync_downloads,
)


def _node(
    node_id: int,
    title: str,
    node_type: int,
    valid_document: bool,
    slug: str,
    children: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    suffix = "/" if node_type == 8 else ""
    node: dict[str, object] = {
        "id": node_id,
        "nodeType": node_type,
        "title": title,
        "validDocument": valid_document,
        "alias": f"/model-studio/{slug}",
        "url": f"/zh/model-studio/{slug}{suffix}",
    }
    if children is not None:
        node["children"] = children
    return node


def _catalog() -> dict[str, object]:
    model_guide_children = [
        _node(
            10,
            "开始/准备",
            8,
            False,
            "start",
            [
                _node(101, "产品:简介?", 1, True, "introduction"),
                _node(
                    102,
                    "地域",
                    8,
                    True,
                    "regions",
                    [_node(103, "接入域名", 1, True, "region-endpoint")],
                ),
                _node(104, "未发布", 1, False, "unpublished"),
            ],
        )
    ]
    section_children = {
        "用户指南（模型）": model_guide_children,
        "用户指南（应用）": [_node(201, "应用入门", 1, True, "application-start")],
        "API参考（模型）": [_node(301, "文本生成", 1, True, "text-generation-api")],
        "API参考（应用）": [_node(401, "应用调用", 1, True, "application-call-api")],
    }
    return {
        "id": 2400256,
        "alias": "/model-studio",
        "children": [
            {
                "id": section_id,
                "nodeType": 8,
                "title": title,
                "validDocument": False,
                "alias": f"/model-studio/{section_id}",
                "url": f"/zh/model-studio/{section_id}/",
                "children": section_children[title],
            }
            for section_id, title in SECTION_SPECS
        ],
    }


def _location() -> SourceLocation:
    return SourceLocation(
        canonical_origin="https://help.aliyun.com",
        fetch_origin="https://help.aliyun.cn",
        language="zh",
        product_alias="model-studio",
    )


def _plan() -> DocumentPlan:
    return DocumentPlan(
        node_id=101,
        node_type=1,
        title="产品简介",
        section="用户指南（模型）",
        alias="/model-studio/introduction",
        url_path="/zh/model-studio/introduction",
        source_url="https://help.aliyun.com/zh/model-studio/introduction",
        relative_path=Path("用户指南（模型）/1.开始/1.产品简介.md"),
    )


def _content_html() -> str:
    return r"""
    <div lang="zh" class="icms-help-docs-content">
      <main>
        <p>这是正文。</p>
        <div class="note note-important">
          <div class="note-icon-wrapper"><i class="icon-note"></i></div>
          <div class="noteContentSpan"><strong>重要</strong><p>请保管好密钥。</p></div>
        </div>
        <div class="tabbed-codeblock-box">
          <div class="tab-box"></div>
          <input checked type="radio" id="python-tab" />
          <label for="python-tab">Python</label>
          <div class="codeblock-item">
            <pre class="pre codeblock language-python"><code>print(&quot;你好&quot;)</code></pre>
          </div>
          <input type="radio" id="bash-tab" />
          <label for="bash-tab"></label>
          <div class="codeblock-item">
            <pre syntax="bash" outputclass="language-bash"><code>echo ok</code></pre>
          </div>
        </div>
        <table><thead><tr><th>字段</th><th>说明</th></tr></thead>
          <tbody><tr><td>model</td><td>模型</td></tr></tbody></table>
        <p><a href="/zh/model-studio/models">模型列表</a>
          <img alt="架构图" src="/assets/architecture.png" /></p>
        <p><hetu formula="\frac{1}{2}" style="display:inline-block"></hetu></p>
        <hetu type="flowchart"><img alt="流程图" src="/assets/flow.svg" /></hetu>
        <hetu type="flowchart"></hetu>
        <label for="model"><strong>选择模型</strong></label>
        <select id="model"><option selected>qwen-plus</option><option>qwen-max</option></select>
        <textarea>第一行
第二行</textarea><button>生成</button>
        <p><input type="checkbox" checked />已完成</p>
        <video src="/assets/demo.mp4" title="演示"></video>
      </main>
    </div>
    """


def _payload(plan: DocumentPlan | None = None) -> dict[str, object]:
    active_plan = plan or _plan()
    return {
        "code": 200,
        "success": True,
        "data": {
            "nodeId": active_plan.node_id,
            "alias": active_plan.alias,
            "url": active_plan.url_path,
            "docTitle": "产品简介正文",
            "content": _content_html(),
            "lastModifiedTime": 1783669235000,
        },
    }


def _downloaded(plan: DocumentPlan, markdown: str) -> DownloadedDocument:
    normalized = markdown.rstrip() + "\n"
    return DownloadedDocument(
        plan=plan,
        document_title="产品简介正文",
        last_modified=1783669235000,
        markdown=normalized,
        digest=sha256(normalized.encode()).hexdigest(),
    )


def test_parse_source_url_keeps_canonical_and_fetch_origins_separate() -> None:
    location = parse_source_url("https://help.aliyun.com/zh/model-studio")

    assert location == _location()
    assert location.catalog_url == "https://help.aliyun.cn/help/json/menupath.json"
    assert location.document_url == "https://help.aliyun.cn/help/json/document_detail.json"
    assert location.source_url == "https://help.aliyun.com/zh/model-studio"


def test_parse_source_url_rejects_other_products() -> None:
    with pytest.raises(BailianDocsError, match="只支持"):
        parse_source_url("https://help.aliyun.com/zh/ecs")


def test_build_document_plans_preserves_sections_order_and_directory_documents() -> None:
    plans = build_document_plans(_catalog(), _location())

    assert [plan.node_id for plan in plans] == [101, 102, 103, 201, 301, 401]
    assert plans[0].relative_path.as_posix() == "用户指南（模型）/1.开始／准备/1.产品：简介？.md"
    assert plans[1].relative_path.as_posix() == "用户指南（模型）/1.开始／准备/2.地域/index.md"
    assert plans[2].relative_path.as_posix() == "用户指南（模型）/1.开始／准备/2.地域/1.接入域名.md"
    assert plans[0].source_url == "https://help.aliyun.com/zh/model-studio/introduction"


def test_build_document_plans_filters_and_validates_sections() -> None:
    plans = build_document_plans(_catalog(), _location(), {"API参考（模型）"})

    assert [plan.node_id for plan in plans] == [301]
    with pytest.raises(BailianDocsError, match="可用栏目"):
        build_document_plans(_catalog(), _location(), {"不存在"})


def test_build_document_plans_rejects_duplicate_node_id() -> None:
    catalog = copy.deepcopy(_catalog())
    sections = cast_sections(catalog)
    second_section_children = cast_children(sections[1])
    second_section_children[0]["id"] = 101

    with pytest.raises(BailianDocsError, match="重复声明文档节点"):
        build_document_plans(catalog, _location())


def test_extract_downloaded_document_preserves_complex_content() -> None:
    document = extract_downloaded_document(_payload(), _plan())

    assert document.document_title == "产品简介正文"
    assert document.last_modified == 1783669235000
    assert document.markdown.startswith("# 产品简介正文\n")
    assert "> **重要**" in document.markdown
    assert "请保管好密钥。" in document.markdown
    assert "**Python**" in document.markdown
    assert '```python\nprint("你好")\n```' in document.markdown
    assert "**bash**" in document.markdown
    assert "```bash\necho ok\n```" in document.markdown
    assert "| 字段 | 说明 |" in document.markdown
    assert "https://help.aliyun.com/zh/model-studio/models" in document.markdown
    assert "https://help.aliyun.com/assets/architecture.png" in document.markdown
    assert "$\\frac{1}{2}$" in document.markdown
    assert "![流程图](https://help.aliyun.com/assets/flow.svg)" in document.markdown
    assert "[流程图：查看官网原文](https://help.aliyun.com/zh/model-studio/introduction)" in document.markdown
    assert "**选择模型**" in document.markdown
    assert "**qwen-plus（默认）**" in document.markdown
    assert "- qwen-max" in document.markdown
    assert "```\n第一行\n第二行\n```" in document.markdown
    assert "生成" not in document.markdown
    assert "[x] 已完成" in document.markdown
    assert "[视频：演示](https://help.aliyun.com/assets/demo.mp4)" in document.markdown
    assert "<input" not in document.markdown
    assert "<hetu" not in document.markdown


def test_convert_document_html_rejects_missing_content_root() -> None:
    with pytest.raises(BailianDocsError, match="icms-help-docs-content"):
        convert_document_html("<main>错误页面</main>", "标题", _plan().source_url, 101)


def test_extract_downloaded_document_rejects_mismatched_node() -> None:
    payload = _payload()
    data = payload["data"]
    assert isinstance(data, dict)
    data["nodeId"] = 999

    with pytest.raises(BailianDocsError, match="节点不匹配"):
        extract_downloaded_document(payload, _plan())


def test_fetch_parsed_retries_verification_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(
                200,
                text='<script>window._config_={"action":"captcha"};_____tmd_____/punish</script>',
                headers={"Content-Type": "text/html"},
                request=request,
            )
        return httpx.Response(
            200,
            json={"code": 200, "success": True, "data": {"value": "ok"}},
            request=request,
        )

    monkeypatch.setattr("update_bailian_docs.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_parsed(
            client=client,
            url="https://help.aliyun.cn/help/json/test.json",
            params={},
            attempts=2,
            context="测试接口",
            parser=lambda payload: payload["data"],
        )

    assert result == {"value": "ok"}
    assert request_count == 2


def test_sync_downloads_covers_create_unchanged_update_and_prune(
    tmp_path: Path,
) -> None:
    plan = _plan()
    sections = {plan.section}
    first = sync_downloads(
        tmp_path,
        _location(),
        [plan],
        [_downloaded(plan, "第一版")],
        sections,
        prune=False,
        allow_prune=True,
    )
    first_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    second = sync_downloads(
        tmp_path,
        _location(),
        [plan],
        [_downloaded(plan, "第一版")],
        sections,
        prune=False,
        allow_prune=True,
    )
    second_manifest = (tmp_path / MANIFEST_NAME).read_bytes()
    third = sync_downloads(
        tmp_path,
        _location(),
        [plan],
        [_downloaded(plan, "第二版")],
        sections,
        prune=False,
        allow_prune=True,
    )
    fourth = sync_downloads(
        tmp_path,
        _location(),
        [],
        [],
        sections,
        prune=True,
        allow_prune=True,
    )

    assert first.created == 1
    assert second.unchanged == 1
    assert second_manifest == first_manifest
    assert third.updated == 1
    assert fourth.removed == 1
    assert not (tmp_path / plan.relative_path).exists()


def test_sync_downloads_preserves_local_modification_during_prune(
    tmp_path: Path,
) -> None:
    plan = _plan()
    sections = {plan.section}
    sync_downloads(
        tmp_path,
        _location(),
        [plan],
        [_downloaded(plan, "远端版本")],
        sections,
        prune=False,
        allow_prune=True,
    )
    target = tmp_path / plan.relative_path
    target.write_text("本地修改\n", encoding="utf-8")

    result = sync_downloads(
        tmp_path,
        _location(),
        [],
        [],
        sections,
        prune=True,
        allow_prune=True,
    )

    assert result.preserved == 1
    assert target.read_text(encoding="utf-8") == "本地修改\n"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("调用：API？", "调用：API？"),
        ("CON", "_CON"),
        ("路径/名称", "路径／名称"),
    ],
)
def test_safe_path_component(title: str, expected: str) -> None:
    assert safe_path_component(title) == expected


def cast_sections(catalog: dict[str, object]) -> list[dict[str, object]]:
    """将测试目录根节点 children 窄化为对象列表。"""
    sections = catalog["children"]
    assert isinstance(sections, list)
    assert all(isinstance(section, dict) for section in sections)
    return sections


def cast_children(node: dict[str, object]) -> list[dict[str, object]]:
    """将测试目录节点 children 窄化为对象列表。"""
    children = node["children"]
    assert isinstance(children, list)
    assert all(isinstance(child, dict) for child in children)
    return children
