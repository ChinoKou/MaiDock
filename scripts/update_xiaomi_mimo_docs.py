"""下载小米 MiMo 中文文档，并将服务端 HTML 转换为 Markdown。"""

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Comment, Tag
from docs_sync_common import (
    PROVIDER_DOCS_ROOT,
    DocsSyncError,
    SyncStats,
    atomic_write,
    encode_stable_manifest,
    project_output_path,
    remove_empty_directories,
    remove_tracked_file,
)
from markdownify import markdownify

SOURCE_URL = "https://mimo.mi.com/docs/zh-CN"
OUTPUT_DIRECTORY = PROVIDER_DOCS_ROOT / "xiaomi_mimo"
DOCUMENTS: tuple[str, ...] = ()
WORKERS = 6
ATTEMPTS = 3
TIMEOUT_SECONDS = 30.0
PRUNE = False
DRY_RUN = False
VERBOSE = False
MANIFEST_NAME = ".xiaomi_mimo_docs.json"
MANIFEST_VERSION = 1
RETRYABLE_STATUS_CODES = {405, 408, 425, 429}
STATIC_DOCS_PREFIX = "/static/docs/"
UNAVAILABLE_ZH_PATHS = {
    "quick-start/terms/privacy-policy",
    "quick-start/terms/user-agreement",
}
MARKDOWN_LINK_PATTERN = re.compile(r"^\s*-\s+\[([^\]]+)]\((https?://[^)]+\.md)\)\s*$", re.MULTILINE)
INVALID_FILENAME_TRANSLATION = str.maketrans(
    {
        "<": "＜",
        ">": "＞",
        ":": "：",
        '"': "＂",
        "/": "／",
        "\\": "＼",
        "|": "｜",
        "?": "？",
        "*": "＊",
    }
)
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class MimoDocsError(DocsSyncError):
    """表示 MiMo 文档同步过程中发生了可定位的错误。"""


@dataclass(slots=True)
class SourceLocation:
    """保存 MiMo 文档站点地址及语言。"""

    origin: str
    lang: str

    @property
    def catalog_url(self) -> str:
        """返回官方 llms.txt 地址。"""
        return f"{self.origin}/llms.txt"

    def document_url(self, catalog_path: str) -> str:
        """返回一篇中文文档的页面地址。"""
        encoded_path = quote(catalog_path, safe="/")
        return f"{self.origin}/docs/{quote(self.lang)}/{encoded_path}"


@dataclass(slots=True)
class DocumentPlan:
    """描述一篇 MiMo 文档的远端身份和本地路径。"""

    catalog_path: str
    catalog_title: str
    source_url: str
    relative_path: Path


@dataclass(slots=True)
class DownloadedDocument:
    """保存转换后的 Markdown 与元数据。"""

    plan: DocumentPlan
    title: str
    markdown: str
    digest: str


def parse_source_url(source_url: str) -> SourceLocation:
    """解析 MiMo 文档首页 URL。"""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MimoDocsError(f"文档地址无效：{source_url}")
    match = re.fullmatch(r"/docs/([^/]+)/?", parsed.path)
    if match is None:
        raise MimoDocsError("文档地址路径必须是 /docs/<language>")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    return SourceLocation(origin=origin, lang=unquote(match.group(1)))


def parse_catalog(llms_text: str, location: SourceLocation) -> tuple[list[DocumentPlan], list[str]]:
    """从 llms.txt 提取文档清单，并排除当前无中文正文的条款。"""
    llms_text = llms_text.lstrip("\ufeff")
    plans: list[DocumentPlan] = []
    excluded: list[str] = []
    seen_paths: set[str] = set()
    seen_local_paths: set[str] = set()

    for title, raw_url in MARKDOWN_LINK_PATTERN.findall(llms_text):
        parsed = urlsplit(raw_url)
        expected_origin = urlsplit(location.origin)
        if (parsed.scheme, parsed.netloc) != (
            expected_origin.scheme,
            expected_origin.netloc,
        ):
            raise MimoDocsError(f"llms.txt 包含非本站文档：{raw_url}")
        if not parsed.path.startswith(STATIC_DOCS_PREFIX) or not parsed.path.endswith(".md"):
            raise MimoDocsError(f"llms.txt 包含未知文档路径：{raw_url}")

        catalog_path = unquote(parsed.path[len(STATIC_DOCS_PREFIX) : -3]).strip("/")
        path_parts = PurePosixPath(catalog_path).parts
        if not catalog_path or any(part in {"", ".", ".."} for part in path_parts):
            raise MimoDocsError(f"llms.txt 包含不安全路径：{raw_url}")
        if catalog_path in seen_paths:
            raise MimoDocsError(f"llms.txt 重复声明文档：{catalog_path}")
        seen_paths.add(catalog_path)

        if catalog_path in UNAVAILABLE_ZH_PATHS:
            excluded.append(catalog_path)
            continue

        safe_parts = [safe_path_component(part) for part in path_parts]
        relative_path = Path(*safe_parts[:-1], f"{safe_parts[-1]}.md")
        normalized_local_path = relative_path.as_posix().casefold()
        if normalized_local_path in seen_local_paths:
            raise MimoDocsError(f"多个文档映射到了同一路径：{relative_path.as_posix()}")
        seen_local_paths.add(normalized_local_path)
        plans.append(
            DocumentPlan(
                catalog_path=catalog_path,
                catalog_title=title.strip(),
                source_url=location.document_url(catalog_path),
                relative_path=relative_path,
            )
        )

    if not plans:
        raise MimoDocsError("llms.txt 中没有找到可下载的 Markdown 文档链接")
    return plans, excluded


def safe_path_component(value: str, max_length: int = 100) -> str:
    """将 URL 路径段转换成跨平台可用的文件名。"""
    component = value.translate(INVALID_FILENAME_TRANSLATION)
    component = re.sub(r"[\x00-\x1f]+", " ", component)
    component = re.sub(r"\s+", " ", component).strip(" .")
    if not component:
        component = "未命名"
    if component.upper() in WINDOWS_RESERVED_NAMES:
        component = f"_{component}"
    component = component[:max_length].rstrip(" .")
    return component or "未命名"


def convert_document_html(page_html: str, plan: DocumentPlan) -> DownloadedDocument:
    """提取中文正文 DOM，并转换成适合离线阅读的 Markdown。"""
    soup = BeautifulSoup(page_html, "html.parser")
    content = soup.select_one("div.mdxContent")
    if not isinstance(content, Tag):
        raise MimoDocsError(f"文档 {plan.catalog_path} 未找到 div.mdxContent，页面可能返回了首页或结构已变化")

    title_node = content.find("h1")
    if not isinstance(title_node, Tag):
        raise MimoDocsError(f"文档 {plan.catalog_path} 缺少 h1 标题")
    title = title_node.get_text(" ", strip=True)
    if not title:
        raise MimoDocsError(f"文档 {plan.catalog_path} 的 h1 标题为空")

    for comment in content.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()
    _normalize_code_blocks(soup, content)
    _normalize_tabs(soup, content)
    _normalize_schemas(soup, content, plan.catalog_path)
    _normalize_media(soup, content, plan.source_url)
    _normalize_callouts(content)
    _normalize_links(content, plan.source_url)
    _normalize_dynamic_components(soup, content, plan.source_url)

    converted = markdownify(
        str(content),
        heading_style="ATX",
        bullets="-",
        code_language_callback=_code_language,
    )
    markdown = converted.strip().lstrip("\ufeff") + "\n"
    if not markdown.startswith("# "):
        raise MimoDocsError(f"文档 {plan.catalog_path} 转换后没有一级标题")
    forbidden_tags = ("<inline-schema-v2", "<mimo-code-block", "<mimo-tab")
    if any(tag in markdown for tag in forbidden_tags):
        raise MimoDocsError(f"文档 {plan.catalog_path} 转换后仍包含未处理的 MiMo 组件")
    digest = sha256(markdown.encode("utf-8")).hexdigest()
    return DownloadedDocument(plan=plan, title=title, markdown=markdown, digest=digest)


def fetch_parsed[ParsedValue](
    client: httpx.Client,
    url: str,
    attempts: int,
    context: str,
    parser: Callable[[str], ParsedValue],
) -> ParsedValue:
    """请求远端内容，并对暂时性响应或解析失败进行重试。"""
    if attempts < 1:
        raise MimoDocsError("attempts 必须大于等于 1")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(url)
            if _is_retryable_status(response.status_code):
                raise MimoDocsError(f"服务端暂时返回 HTTP {response.status_code}")
            response.raise_for_status()
            return parser(response.text)
        except (httpx.RequestError, httpx.HTTPStatusError, MimoDocsError) as exc:
            last_error = exc
            retryable = not isinstance(exc, httpx.HTTPStatusError) or _is_retryable_status(exc.response.status_code)
            if attempt >= attempts or not retryable:
                break
            time.sleep(_retry_delay(response, attempt))
    raise MimoDocsError(f"请求失败（{context}，{url}）：{last_error}") from last_error


def download_document(client: httpx.Client, plan: DocumentPlan, attempts: int) -> DownloadedDocument:
    """下载并转换一篇 MiMo 中文文档。"""
    return fetch_parsed(
        client,
        plan.source_url,
        attempts,
        f"文档 {plan.catalog_path}",
        lambda page_html: convert_document_html(page_html, plan),
    )


def load_manifest(output_root: Path) -> dict[str, object]:
    """读取既有清单；首次运行时返回空清单。"""
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.exists():
        return {"format_version": MANIFEST_VERSION, "documents": {}}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MimoDocsError(f"清单读取失败：{manifest_path}：{exc}") from exc
    manifest = _expect_mapping(parsed, str(manifest_path))
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise MimoDocsError(f"不支持的清单版本：{manifest.get('format_version')}")
    _mapping_value(manifest, "documents", str(manifest_path))
    return manifest


def sync_downloads(
    output_root: Path,
    location: SourceLocation,
    all_plans: Sequence[DocumentPlan],
    downloads: Sequence[DownloadedDocument],
    excluded: Sequence[str],
    prune: bool,
    allow_prune: bool,
) -> SyncStats:
    """增量写入 Markdown 和清单，并按需清理已失效文件。"""
    stats = SyncStats()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_root)
    existing_documents = _mapping_value(manifest, "documents", "同步清单")
    manifest_documents: dict[str, object] = dict(existing_documents)

    for document in downloads:
        target_path = output_root / document.plan.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = document.markdown.encode("utf-8")
        if not target_path.exists():
            atomic_write(target_path, encoded)
            stats.created += 1
        elif target_path.read_bytes() == encoded:
            stats.unchanged += 1
        else:
            atomic_write(target_path, encoded)
            stats.updated += 1

        key = document.plan.catalog_path
        old_record_raw = manifest_documents.get(key)
        if isinstance(old_record_raw, dict):
            old_record = cast(dict[str, object], old_record_raw)
            old_path = old_record.get("path")
            if isinstance(old_path, str) and old_path != document.plan.relative_path.as_posix():
                if remove_tracked_file(output_root, old_path, str(old_record.get("sha256", ""))):
                    stats.removed += 1
                else:
                    stats.preserved += 1

        manifest_documents[key] = {
            "catalog_title": document.plan.catalog_title,
            "title": document.title,
            "path": document.plan.relative_path.as_posix(),
            "source_url": document.plan.source_url,
            "sha256": document.digest,
        }

    if prune and allow_prune:
        planned_keys = {plan.catalog_path for plan in all_plans}
        for key, raw_record in list(manifest_documents.items()):
            if key in planned_keys or not isinstance(raw_record, dict):
                continue
            record = cast(dict[str, object], raw_record)
            old_path = record.get("path")
            old_digest = record.get("sha256")
            if isinstance(old_path, str) and isinstance(old_digest, str):
                if remove_tracked_file(output_root, old_path, old_digest):
                    stats.removed += 1
                else:
                    stats.preserved += 1
            del manifest_documents[key]

    manifest_payload = {
        "format_version": MANIFEST_VERSION,
        "source_url": f"{location.origin}/docs/{location.lang}",
        "catalog_url": location.catalog_url,
        "language": location.lang,
        "excluded": list(excluded),
        "documents": manifest_documents,
    }
    manifest_bytes = encode_stable_manifest(manifest, manifest_payload)
    atomic_write(output_root / MANIFEST_NAME, manifest_bytes)
    remove_empty_directories(output_root)
    return stats


def create_http_client(workers: int, timeout: float) -> httpx.Client:
    """创建可供下载线程共享的 HTTP 客户端。"""
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        headers={
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "User-Agent": "MaiDock-MiMo-Docs-Sync/1.0",
        },
    )


def _validate_sync_settings(
    *,
    workers: int,
    attempts: int,
    timeout: float,
    documents: tuple[str, ...],
    prune: bool,
) -> None:
    """校验 MiMo 同步参数。"""
    if not 1 <= workers <= 16:
        raise MimoDocsError("workers 必须在 1 到 16 之间")
    if not 1 <= attempts <= 10:
        raise MimoDocsError("attempts 必须在 1 到 10 之间")
    if timeout <= 0:
        raise MimoDocsError("timeout 必须大于 0")
    if prune and documents:
        raise MimoDocsError("指定 documents 时不能同时启用 prune")


def run(
    *,
    source_url: str,
    output: Path,
    documents: tuple[str, ...],
    workers: int,
    attempts: int,
    timeout: float,
    prune: bool,
    dry_run: bool,
    verbose: bool,
) -> int:
    """按显式参数执行同步流程并返回进程退出码。"""
    _validate_sync_settings(
        workers=workers,
        attempts=attempts,
        timeout=timeout,
        documents=documents,
        prune=prune,
    )
    location = parse_source_url(source_url)
    output_root = project_output_path(output)
    selected_documents = set(documents)

    with create_http_client(workers, timeout) as client:
        llms_text = fetch_parsed(
            client,
            location.catalog_url,
            attempts,
            "llms.txt",
            lambda value: value,
        )
        all_plans, excluded = parse_catalog(llms_text, location)
        if selected_documents:
            plans = [plan for plan in all_plans if plan.catalog_path in selected_documents]
            missing = selected_documents - {plan.catalog_path for plan in plans}
            if missing:
                raise MimoDocsError(f"目录中找不到中文文档：{'、'.join(sorted(missing))}")
        else:
            plans = all_plans

        print(f"发现 {len(plans)} 篇中文文档；跳过 {len(excluded)} 篇当前没有中文正文的条款。")
        if dry_run:
            if verbose:
                for plan in plans:
                    print(f"{plan.catalog_path}\t{plan.relative_path.as_posix()}")
            print("dry-run 完成，未下载正文或修改文件。")
            return 0

        downloads, failures = _download_all(
            client=client,
            plans=plans,
            attempts=attempts,
            workers=workers,
            verbose=verbose,
        )

    stats = sync_downloads(
        output_root=output_root,
        location=location,
        all_plans=all_plans,
        downloads=downloads,
        excluded=excluded,
        prune=prune,
        allow_prune=not failures and not selected_documents,
    )
    print(
        f"同步结果：新增 {stats.created}，更新 {stats.updated}，未变化 {stats.unchanged}，"
        f"删除旧文件 {stats.removed}，保留本地修改 {stats.preserved}。"
    )
    if failures:
        print(
            f"以下 {len(failures)} 篇文档下载或转换失败，本次未执行清理：",
            file=sys.stderr,
        )
        for plan, error in failures:
            print(f"- {plan.catalog_path}：{error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    """使用模块级代码常量执行文档同步。"""
    exit_code = run(
        source_url=SOURCE_URL,
        output=OUTPUT_DIRECTORY,
        documents=DOCUMENTS,
        workers=WORKERS,
        attempts=ATTEMPTS,
        timeout=TIMEOUT_SECONDS,
        prune=PRUNE,
        dry_run=DRY_RUN,
        verbose=VERBOSE,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def _normalize_code_blocks(soup: BeautifulSoup, content: Tag) -> None:
    """使用 raw 属性恢复未被语法高亮 span 拆散的原始代码。"""
    for pre in list(content.find_all("pre", attrs={"raw": True})):
        raw_value = pre.get("raw")
        if not isinstance(raw_value, str):
            continue
        language = _language_from_classes(pre.get("class"))
        clean_pre = soup.new_tag("pre")
        clean_code = soup.new_tag("code")
        if language:
            clean_code["class"] = f"language-{language}"
        clean_code.string = unquote(raw_value)
        clean_pre.append(clean_code)
        wrapper = pre.parent
        if isinstance(wrapper, Tag) and wrapper.name == "mimo-code-block":
            wrapper.replace_with(clean_pre)
        else:
            pre.replace_with(clean_pre)


def _normalize_tabs(soup: BeautifulSoup, content: Tag) -> None:
    """将网页 Tabs 展开为带粗体标签的连续 Markdown 小节。"""
    for tab in list(content.find_all("mimo-tab")):
        for item in list(tab.find_all("mimo-tab-item", recursive=False)):
            label = item.get("label")
            label_text = str(label).strip() if label is not None else "选项"
            paragraph = soup.new_tag("p")
            strong = soup.new_tag("strong")
            strong.string = label_text
            paragraph.append(strong)
            item.insert(0, paragraph)
            item.unwrap()
        tab.unwrap()


def _normalize_schemas(soup: BeautifulSoup, content: Tag, catalog_path: str) -> None:
    """将 API schema JSON 递归展开为嵌套字段列表。"""
    for schema_tag in list(content.find_all("inline-schema-v2")):
        raw_schema = schema_tag.get("schema")
        if not isinstance(raw_schema, str):
            raise MimoDocsError(f"文档 {catalog_path} 的 inline-schema-v2 缺少 schema")
        try:
            schema_data = json.loads(raw_schema)
        except json.JSONDecodeError as exc:
            raise MimoDocsError(f"文档 {catalog_path} 的 API schema JSON 无效：{exc}") from exc
        if not isinstance(schema_data, list):
            raise MimoDocsError(f"文档 {catalog_path} 的 API schema 根节点不是数组")
        schema_tag.replace_with(_build_schema_list(soup, cast(list[object], schema_data), catalog_path))
    for wrapper in list(content.find_all("inlineschema")):
        wrapper.unwrap()


def _build_schema_list(soup: BeautifulSoup, items: Sequence[object], catalog_path: str) -> Tag:
    """递归构造 schema 的 HTML 列表，交给 markdownify 生成 Markdown。"""
    unordered_list = soup.new_tag("ul")
    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise MimoDocsError(f"文档 {catalog_path} 的 API schema 包含非对象字段")
        item = cast(dict[str, object], raw_item)
        name = item.get("name")
        field_type = item.get("type")
        if not isinstance(name, str) or not name:
            raise MimoDocsError(f"文档 {catalog_path} 的 API schema 字段缺少 name")

        list_item = soup.new_tag("li")
        name_code = soup.new_tag("code")
        name_code.string = name
        list_item.append(name_code)
        if isinstance(field_type, list):
            type_text = " / ".join(str(value) for value in cast(list[object], field_type))
        elif field_type is None:
            type_text = "unknown"
        else:
            type_text = str(field_type)
        type_code = soup.new_tag("code")
        type_code.string = type_text
        list_item.append(" ")
        list_item.append(type_code)
        if item.get("required") is True:
            required = soup.new_tag("strong")
            required.string = "必填"
            list_item.append(" ")
            list_item.append(required)
        if "defaultValue" in item:
            default_code = soup.new_tag("code")
            default_code.string = str(item["defaultValue"])
            list_item.append(" 默认值：")
            list_item.append(default_code)

        description = item.get("description")
        if isinstance(description, str) and description:
            list_item.append(" - ")
            fragment = BeautifulSoup(description, "html.parser")
            for child in list(fragment.contents):
                list_item.append(child.extract())

        children = item.get("children")
        if children is not None:
            if not isinstance(children, list):
                raise MimoDocsError(f"文档 {catalog_path} 的 API schema children 不是数组")
            list_item.append(_build_schema_list(soup, cast(list[object], children), catalog_path))
        unordered_list.append(list_item)
    return unordered_list


def _normalize_media(soup: BeautifulSoup, content: Tag, source_url: str) -> None:
    """将音视频组件转换成可离线识别的绝对链接。"""
    media_tags = [("mimo-audio", "音频"), ("video", "视频")]
    for tag_name, label in media_tags:
        for media in list(content.find_all(tag_name)):
            source = media.get("src")
            if not isinstance(source, str) or not source:
                continue
            title = media.get("title")
            title_text = str(title).strip() if title is not None else label
            paragraph = soup.new_tag("p")
            link = soup.new_tag("a", href=urljoin(source_url, source))
            link.string = f"{label}：{title_text}"
            paragraph.append(link)
            media.replace_with(paragraph)


def _normalize_callouts(content: Tag) -> None:
    """把 MiMo 提示框转换成 Markdown 引用块。"""
    for callout in content.find_all("div"):
        classes = callout.get("class")
        if isinstance(classes, list) and "mdx-highlight" in classes:
            callout.name = "blockquote"
            callout.attrs = {}


def _normalize_links(content: Tag, source_url: str) -> None:
    """将站内链接和资源地址转换成绝对 URL。"""
    for link in content.find_all("a", href=True):
        href = link.get("href")
        if isinstance(href, str) and not href.startswith("#"):
            link["href"] = urljoin(source_url, href)
    for image in content.find_all("img", src=True):
        source = image.get("src")
        if isinstance(source, str):
            image["src"] = urljoin(source_url, source)


def _normalize_dynamic_components(soup: BeautifulSoup, content: Tag, source_url: str) -> None:
    """为无法静态展开的交互组件保留原文入口。"""
    labels = {
        "mimo-tool-grid": "交互式工具列表",
        "mimo-banner": "交互式公告",
        "mimo-quick-start-cards": "交互式快速开始卡片",
    }
    for tag_name, label in labels.items():
        for component in list(content.find_all(tag_name)):
            paragraph = soup.new_tag("p")
            paragraph.append(f"{label}：")
            link = soup.new_tag("a", href=source_url)
            link.string = "查看官网原文"
            paragraph.append(link)
            component.replace_with(paragraph)


def _code_language(element: Tag) -> str:
    """为 markdownify 的 fenced code block 提取语言。"""
    code = element.find("code")
    if not isinstance(code, Tag):
        return ""
    return _language_from_classes(code.get("class"))


def _language_from_classes(classes: object) -> str:
    """从 BeautifulSoup class 属性中提取 language-*。"""
    if not isinstance(classes, list):
        return ""
    for class_name in classes:
        if isinstance(class_name, str) and class_name.startswith("language-"):
            return class_name.removeprefix("language-")
    return ""


def _download_all(
    client: httpx.Client,
    plans: Sequence[DocumentPlan],
    attempts: int,
    workers: int,
    verbose: bool,
) -> tuple[list[DownloadedDocument], list[tuple[DocumentPlan, Exception]]]:
    """并发下载全部计划，并保留精确的失败信息。"""
    downloaded_by_path: dict[str, DownloadedDocument] = {}
    failures: list[tuple[DocumentPlan, Exception]] = []
    future_map: dict[Future[DownloadedDocument], DocumentPlan] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mimo-docs") as executor:
        for plan in plans:
            future = executor.submit(download_document, client, plan, attempts)
            future_map[future] = plan
        for completed, future in enumerate(as_completed(future_map), start=1):
            plan = future_map[future]
            try:
                downloaded_by_path[plan.catalog_path] = future.result()
                if verbose or completed % 10 == 0 or completed == len(plans):
                    print(f"下载进度：{completed}/{len(plans)}")
            except Exception as exc:
                failures.append((plan, exc))
    downloads = [downloaded_by_path[plan.catalog_path] for plan in plans if plan.catalog_path in downloaded_by_path]
    return downloads, failures


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """优先采用 Retry-After，否则使用有上限的指数退避。"""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, min(float(retry_after), 60.0))
            except ValueError:
                pass
    return min(float(2 ** (attempt - 1)), 8.0)


def _is_retryable_status(status_code: int) -> bool:
    """识别文档站实测会瞬时返回的状态码和服务端错误。"""
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


def _expect_mapping(value: object, context: str) -> dict[str, object]:
    """校验动态 JSON 值是字符串键映射。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MimoDocsError(f"{context} 应为 JSON 对象")
    return cast(dict[str, object], value)


def _mapping_value(container: Mapping[str, object], key: str, context: str) -> dict[str, object]:
    """读取并校验映射字段。"""
    if key not in container:
        raise MimoDocsError(f"{context} 缺少字段 {key}")
    return _expect_mapping(container[key], f"{context}.{key}")


if __name__ == "__main__":
    main()
