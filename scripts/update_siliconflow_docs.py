"""下载 SiliconFlow 中文文档，并将服务器渲染正文转换为 Markdown。"""

import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

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

SOURCE_URL = "https://api-docs.siliconflow.cn/docs"
OUTPUT_DIRECTORY = PROVIDER_DOCS_ROOT / "siliconflow"
SECTIONS: tuple[str, ...] = ()
DOCUMENT_PATHS: tuple[str, ...] = ()
WORKERS = 4
ATTEMPTS = 3
TIMEOUT_SECONDS = 30.0
PRUNE = False
DRY_RUN = False
VERBOSE = False
MANIFEST_NAME = ".siliconflow_docs.json"
MANIFEST_VERSION = 1
RETRYABLE_STATUS_CODES = {408, 425, 429}
VERIFICATION_MARKERS = ("<title>just a moment", "cf-chl-", "captcha", "访问验证")
HTTP_METHODS = {"DELETE", "GET", "PATCH", "POST", "PUT"}
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


class SiliconFlowDocsError(DocsSyncError):
    """表示 SiliconFlow 文档同步过程中发生了可定位的错误。"""


class RetryableSiliconFlowDocsError(SiliconFlowDocsError):
    """表示远端暂时性响应，可以按退避策略重试。"""


@dataclass(slots=True)
class SourceLocation:
    """保存 SiliconFlow 文档站来源信息。"""

    origin: str
    docs_path: str

    @property
    def catalog_url(self) -> str:
        """返回会跳转到首篇文档的目录入口。"""
        return f"{self.origin}{self.docs_path}"

    def document_url(self, document_path: str) -> str:
        """根据目录中的站内路径生成正文 URL。"""
        return f"{self.origin}{self.docs_path}/{document_path.lstrip('/')}"


@dataclass(slots=True)
class DocumentPlan:
    """描述一篇 SiliconFlow 文档的远端身份和本地路径。"""

    document_path: str
    title: str
    description: str
    section: str
    source_url: str
    relative_path: Path


@dataclass(slots=True)
class DownloadedDocument:
    """保存转换后的 Markdown 及其摘要。"""

    plan: DocumentPlan
    document_title: str
    markdown: str
    digest: str


def parse_source_url(source_url: str) -> SourceLocation:
    """解析并校验 SiliconFlow 中文文档入口。"""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SiliconFlowDocsError(f"文档地址无效：{source_url}")
    if parsed.query or parsed.fragment:
        raise SiliconFlowDocsError("文档地址不能包含查询参数或锚点")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path != "/docs":
        raise SiliconFlowDocsError("当前脚本只支持 SiliconFlow 中文文档入口 /docs")
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    return SourceLocation(origin=origin, docs_path=normalized_path)


def safe_path_component(value: str, max_length: int = 100) -> str:
    """将官方标题转换成 Windows 可用的路径段。"""
    component = value.translate(INVALID_FILENAME_TRANSLATION)
    component = re.sub(r"[\x00-\x1f]+", " ", component)
    component = re.sub(r"\s+", " ", component).strip(" .")
    if not component:
        component = "未命名"
    if component.upper() in WINDOWS_RESERVED_NAMES:
        component = f"_{component}"
    component = component[:max_length].rstrip(" .")
    return component or "未命名"


def extract_catalog_tree(page_html: str) -> dict[str, object]:
    """从 Next.js Flight 数据中提取 Fumadocs 完整目录树。"""
    soup = BeautifulSoup(page_html, "html.parser")
    marker = '"tree":'
    prefix = "self.__next_f.push("
    parse_errors: list[str] = []
    for script in soup.find_all("script"):
        script_text = script.get_text()
        if not script_text.startswith("self.__next_f.push([1,") or not script_text.endswith(")"):
            continue
        try:
            flight_entry = json.loads(script_text[len(prefix) : -1])
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        if (
            not isinstance(flight_entry, list)
            or len(flight_entry) < 2
            or not isinstance(flight_entry[1], str)
            or marker not in flight_entry[1]
        ):
            continue
        chunk = flight_entry[1]
        offset = chunk.index(marker) + len(marker)
        try:
            raw_tree = json.JSONDecoder().raw_decode(chunk, offset)[0]
        except json.JSONDecodeError as exc:
            raise SiliconFlowDocsError(f"目录树 JSON 无法解析：{exc}") from exc
        tree = _expect_mapping(raw_tree, "目录树")
        _sequence_value(tree, "children", "目录树")
        return tree
    detail = f"；Flight JSON 错误：{parse_errors[-1]}" if parse_errors else ""
    raise SiliconFlowDocsError(f"页面未包含可识别的 Fumadocs 目录树{detail}")


def build_document_plans(
    page_html: str,
    location: SourceLocation,
    selected_sections: set[str] | None = None,
) -> list[DocumentPlan]:
    """按官方顺序将完整目录树转换为本地下载计划。"""
    soup = BeautifulSoup(page_html, "html.parser")
    tree = extract_catalog_tree(page_html)
    labels = _sidebar_labels(soup)
    raw_children = _sequence_value(tree, "children", "目录树")
    available_sections = {
        _node_name(_expect_mapping(node, "目录根节点"), "目录根节点")
        for node in raw_children
        if isinstance(node, dict)
        and (
            node.get("type") == "folder"
            or (
                node.get("type") == "page"
                and node.get("external") is not True
                and isinstance(node.get("url"), str)
                and str(node["url"]).startswith(f"{location.docs_path}/")
            )
        )
    }
    if selected_sections is not None:
        unknown = selected_sections - available_sections
        if unknown:
            raise SiliconFlowDocsError(f"目录中不存在指定栏目：{'、'.join(sorted(unknown))}")

    plans: list[DocumentPlan] = []
    seen_paths: set[str] = set()
    root_position = 0
    for raw_node in raw_children:
        node = _expect_mapping(raw_node, "目录根节点")
        node_type = node.get("type")
        if node_type == "separator" or node.get("external") is True:
            continue
        if node_type == "folder":
            section = _node_name(node, "目录栏目")
            root_position += 1
            if selected_sections is not None and section not in selected_sections:
                continue
            _walk_folder(
                node=node,
                section=section,
                section_root=Path(f"{root_position}.{safe_path_component(section)}"),
                location=location,
                labels=labels,
                plans=plans,
                seen_paths=seen_paths,
            )
            continue
        if node_type == "page":
            url = _node_url(node, "根目录文档")
            if not url.startswith(f"{location.docs_path}/"):
                continue
            section = _resolved_title(node, url, labels)
            root_position += 1
            if selected_sections is not None and section not in selected_sections:
                continue
            plans.append(
                _make_plan(
                    node=node,
                    title=section,
                    section=section,
                    relative_path=Path(f"{root_position}.{safe_path_component(section)}.md"),
                    location=location,
                    seen_paths=seen_paths,
                )
            )
            continue
        raise SiliconFlowDocsError(f"目录根节点包含不支持的类型：{node_type}")
    if not plans:
        raise SiliconFlowDocsError("目录树中没有可下载的站内文档")
    return plans


def convert_document_html(page_html: str, plan: DocumentPlan) -> DownloadedDocument:
    """把服务器渲染正文转换成适合离线阅读的 Markdown。"""
    soup = BeautifulSoup(page_html, "html.parser")
    article = soup.select_one("article")
    if not isinstance(article, Tag):
        raise SiliconFlowDocsError(f"文档 {plan.document_path} 未找到 article")
    title_tag = article.find("h1")
    if not isinstance(title_tag, Tag):
        raise SiliconFlowDocsError(f"文档 {plan.document_path} 未找到 h1")
    document_title = title_tag.get_text(" ", strip=True)
    if not document_title:
        raise SiliconFlowDocsError(f"文档 {plan.document_path} 的标题为空")
    content = article.find("div", class_="prose", recursive=False)
    if not isinstance(content, Tag):
        raise SiliconFlowDocsError(f"文档 {plan.document_path} 未找到 article 直属 div.prose")

    for comment in content.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()
    for unwanted in list(content.find_all(["script", "style", "noscript", "template"])):
        unwanted.decompose()
    _normalize_tabs(soup, content)
    _normalize_field_cards(soup, content)
    _normalize_metadata_chips(soup, content)
    code_blocks = _normalize_code_blocks(soup, content)
    _normalize_callouts(content)
    _normalize_details(soup, content)
    _normalize_media(soup, content, plan.source_url)
    _normalize_links(content, plan.source_url)
    _remove_interface_controls(content)

    converted = markdownify(
        str(content),
        heading_style="ATX",
        bullets="-",
    ).strip()
    for placeholder, code_block in code_blocks:
        if converted.count(placeholder) != 1:
            raise SiliconFlowDocsError(f"文档 {plan.document_path} 的代码块占位符数量异常：{placeholder}")
        converted = converted.replace(placeholder, code_block)
    if not converted:
        raise SiliconFlowDocsError(f"文档 {plan.document_path} 转换后的正文为空")
    markdown = f"# {document_title}\n\n{converted}\n"
    return DownloadedDocument(
        plan=plan,
        document_title=document_title,
        markdown=markdown,
        digest=sha256(markdown.encode("utf-8")).hexdigest(),
    )


def fetch_parsed[ParsedValue](
    client: httpx.Client,
    url: str,
    attempts: int,
    context: str,
    parser: Callable[[str], ParsedValue],
) -> ParsedValue:
    """请求 HTML，并只对网络、限流和页面结构异常执行有限重试。"""
    if attempts < 1:
        raise SiliconFlowDocsError("attempts 必须大于等于 1")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(url)
            if _is_retryable_status(response.status_code):
                raise RetryableSiliconFlowDocsError(f"服务端暂时返回 HTTP {response.status_code}")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").casefold()
            if "text/html" not in content_type:
                raise RetryableSiliconFlowDocsError(f"站点返回了非 HTML 内容：{content_type or '未知类型'}")
            if _is_verification_page(response.text):
                raise RetryableSiliconFlowDocsError("站点返回了访问验证页")
            return parser(response.text)
        except (
            httpx.RequestError,
            RetryableSiliconFlowDocsError,
            SiliconFlowDocsError,
        ) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(_retry_delay(response, attempt))
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if attempt >= attempts or not _is_retryable_status(exc.response.status_code):
                break
            time.sleep(_retry_delay(response, attempt))
    raise SiliconFlowDocsError(f"请求失败（{context}，{url}）：{last_error}") from last_error


def download_document(client: httpx.Client, plan: DocumentPlan, attempts: int) -> DownloadedDocument:
    """下载并转换一篇 SiliconFlow 文档。"""
    return fetch_parsed(
        client=client,
        url=plan.source_url,
        attempts=attempts,
        context=f"文档 {plan.document_path} {plan.title}",
        parser=lambda page_html: convert_document_html(page_html, plan),
    )


def load_manifest(output_root: Path) -> dict[str, object]:
    """读取既有清单；首次运行时返回空清单。"""
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.exists():
        return {"format_version": MANIFEST_VERSION, "documents": {}}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiliconFlowDocsError(f"清单读取失败：{manifest_path}：{exc}") from exc
    manifest = _expect_mapping(parsed, str(manifest_path))
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise SiliconFlowDocsError(f"不支持的清单版本：{manifest.get('format_version')}")
    _mapping_value(manifest, "documents", str(manifest_path))
    return manifest


def sync_downloads(
    output_root: Path,
    location: SourceLocation,
    plans: Sequence[DocumentPlan],
    downloads: Sequence[DownloadedDocument],
    selected_sections: set[str],
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

        document_key = document.plan.document_path
        old_record_raw = manifest_documents.get(document_key)
        if isinstance(old_record_raw, dict):
            old_record = cast(dict[str, object], old_record_raw)
            old_path = old_record.get("path")
            if isinstance(old_path, str) and old_path != document.plan.relative_path.as_posix():
                if remove_tracked_file(output_root, old_path, str(old_record.get("sha256", ""))):
                    stats.removed += 1
                else:
                    stats.preserved += 1
        manifest_documents[document_key] = {
            "title": document.plan.title,
            "document_title": document.document_title,
            "description": document.plan.description,
            "section": document.plan.section,
            "path": document.plan.relative_path.as_posix(),
            "source_url": document.plan.source_url,
            "sha256": document.digest,
        }

    if prune and allow_prune:
        planned_paths = {plan.document_path for plan in plans}
        for document_key, raw_record in list(manifest_documents.items()):
            if document_key in planned_paths or not isinstance(raw_record, dict):
                continue
            record = cast(dict[str, object], raw_record)
            if record.get("section") not in selected_sections:
                continue
            old_path = record.get("path")
            old_digest = record.get("sha256")
            if isinstance(old_path, str) and isinstance(old_digest, str):
                if remove_tracked_file(output_root, old_path, old_digest):
                    stats.removed += 1
                else:
                    stats.preserved += 1
            del manifest_documents[document_key]

    manifest_payload = {
        "format_version": MANIFEST_VERSION,
        "source_url": location.catalog_url,
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
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "User-Agent": "MaiDock-SiliconFlow-Docs-Sync/1.0",
        },
    )


def _validate_sync_settings(
    *,
    workers: int,
    attempts: int,
    timeout: float,
    document_paths: tuple[str, ...],
    prune: bool,
) -> None:
    """校验 SiliconFlow 同步参数。"""
    if not 1 <= workers <= 8:
        raise SiliconFlowDocsError("workers 必须在 1 到 8 之间")
    if not 1 <= attempts <= 10:
        raise SiliconFlowDocsError("attempts 必须在 1 到 10 之间")
    if timeout <= 0:
        raise SiliconFlowDocsError("timeout 必须大于 0")
    if prune and document_paths:
        raise SiliconFlowDocsError("指定 document_paths 时不能同时启用 prune")


def run(
    *,
    source_url: str,
    output: Path,
    sections: tuple[str, ...],
    document_paths: tuple[str, ...],
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
        document_paths=document_paths,
        prune=prune,
    )
    location = parse_source_url(source_url)
    output_root = project_output_path(output)
    selected_sections = set(sections)
    selected_document_paths = {_normalize_document_selector(value, location) for value in document_paths}

    with create_http_client(workers, timeout) as client:
        page_html = fetch_parsed(
            client=client,
            url=location.catalog_url,
            attempts=attempts,
            context="完整目录",
            parser=_validate_catalog_html,
        )
        all_plans = build_document_plans(page_html, location, selected_sections or None)
        active_sections = selected_sections or {plan.section for plan in all_plans}
        if selected_document_paths:
            plans = [plan for plan in all_plans if plan.document_path in selected_document_paths]
            missing_paths = selected_document_paths - {plan.document_path for plan in plans}
            if missing_paths:
                raise SiliconFlowDocsError(f"目录中找不到有效文档：{'、'.join(sorted(missing_paths))}")
        else:
            plans = all_plans

        section_counts: dict[str, int] = {}
        for plan in plans:
            section_counts[plan.section] = section_counts.get(plan.section, 0) + 1
        count_text = "，".join(f"{name} {count} 篇" for name, count in section_counts.items())
        print(f"发现 {len(plans)} 篇有效文档：{count_text}")
        if dry_run:
            if verbose:
                for plan in plans:
                    print(f"{plan.document_path}\t{plan.relative_path.as_posix()}")
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
        plans=all_plans,
        downloads=downloads,
        selected_sections=active_sections,
        prune=prune,
        allow_prune=not failures and not selected_document_paths,
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
            print(f"- {plan.document_path} {plan.title}：{error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    """使用模块级代码常量执行文档同步。"""
    exit_code = run(
        source_url=SOURCE_URL,
        output=OUTPUT_DIRECTORY,
        sections=SECTIONS,
        document_paths=DOCUMENT_PATHS,
        workers=WORKERS,
        attempts=ATTEMPTS,
        timeout=TIMEOUT_SECONDS,
        prune=PRUNE,
        dry_run=DRY_RUN,
        verbose=VERBOSE,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def _validate_catalog_html(page_html: str) -> str:
    """验证目录页后原样返回 HTML。"""
    extract_catalog_tree(page_html)
    return page_html


def _walk_folder(
    node: Mapping[str, object],
    section: str,
    section_root: Path,
    location: SourceLocation,
    labels: Mapping[str, str],
    plans: list[DocumentPlan],
    seen_paths: set[str],
) -> None:
    """把栏目中的分隔标题映射成子目录，并保留同组文档顺序。"""
    current_parent = section_root
    section_position = 0
    group_document_position = 0
    for raw_child in _sequence_value(node, "children", f"栏目 {section}"):
        child = _expect_mapping(raw_child, f"栏目 {section} 子节点")
        child_type = child.get("type")
        if child_type == "separator":
            section_position += 1
            group_document_position = 0
            separator_name = _node_name(child, f"栏目 {section} 分组")
            current_parent = section_root / f"{section_position}.{safe_path_component(separator_name)}"
            continue
        if child_type != "page":
            raise SiliconFlowDocsError(f"栏目 {section} 包含不支持的节点类型：{child_type}")
        if child.get("external") is True:
            continue
        url = _node_url(child, f"栏目 {section} 文档")
        if not url.startswith(f"{location.docs_path}/"):
            continue
        title = _resolved_title(child, url, labels)
        if current_parent == section_root:
            section_position += 1
            document_position = section_position
        else:
            group_document_position += 1
            document_position = group_document_position
        relative_path = current_parent / f"{document_position}.{safe_path_component(title)}.md"
        plans.append(
            _make_plan(
                node=child,
                title=title,
                section=section,
                relative_path=relative_path,
                location=location,
                seen_paths=seen_paths,
            )
        )


def _make_plan(
    node: Mapping[str, object],
    title: str,
    section: str,
    relative_path: Path,
    location: SourceLocation,
    seen_paths: set[str],
) -> DocumentPlan:
    """校验单篇目录节点并构造下载计划。"""
    url = _node_url(node, f"文档 {title}")
    prefix = f"{location.docs_path}/"
    if not url.startswith(prefix):
        raise SiliconFlowDocsError(f"文档 {title} 不是站内文档：{url}")
    document_path = unquote(url.removeprefix(prefix)).strip("/")
    if not document_path or document_path in seen_paths:
        raise SiliconFlowDocsError(f"文档路径为空或重复：{document_path}")
    seen_paths.add(document_path)
    description_raw = node.get("description")
    description = description_raw if isinstance(description_raw, str) and description_raw != "$undefined" else ""
    return DocumentPlan(
        document_path=document_path,
        title=title,
        description=description,
        section=section,
        source_url=location.document_url(document_path),
        relative_path=relative_path,
    )


def _sidebar_labels(soup: BeautifulSoup) -> dict[str, str]:
    """读取已展开侧栏中的动态 API 标题。"""
    labels: dict[str, str] = {}
    for link in soup.select("aside#nd-sidebar a[href]"):
        href = link.get("href")
        if not isinstance(href, str) or not href.startswith("/docs/"):
            continue
        parts = list(link.stripped_strings)
        if parts and parts[-1].upper() in HTTP_METHODS:
            parts.pop()
        label = " ".join(parts).strip()
        if label:
            labels[href] = label
    return labels


def _resolved_title(node: Mapping[str, object], url: str, labels: Mapping[str, str]) -> str:
    """解析普通标题，并用侧栏文本还原 React 引用形式的 API 标题。"""
    name = _node_name(node, f"目录节点 {url}")
    if not re.fullmatch(r"\$L[0-9a-z]+", name, flags=re.IGNORECASE):
        return name
    if url not in labels:
        raise SiliconFlowDocsError(f"动态文档标题无法从侧栏还原：{url}（{name}）")
    return labels[url]


def _normalize_tabs(soup: BeautifulSoup, content: Tag) -> None:
    """把 Radix 标签页的所有面板展开，并为每项补充标签。"""
    tab_labels: dict[str, str] = {}
    for tab in content.select('[role="tab"]'):
        tab_id = tab.get("id")
        if isinstance(tab_id, str):
            label = tab.get_text(" ", strip=True)
            if label:
                tab_labels[tab_id] = label
    for panel in content.select('[role="tabpanel"]'):
        labelled_by = panel.get("aria-labelledby")
        if isinstance(labelled_by, str) and labelled_by in tab_labels:
            paragraph = soup.new_tag("p")
            strong = soup.new_tag("strong")
            strong.string = f"选项：{tab_labels[labelled_by]}"
            paragraph.append(strong)
            panel.insert(0, paragraph)
        panel.attrs.pop("hidden", None)
        panel.attrs.pop("role", None)
    for tab_list in list(content.select('[role="tablist"]')):
        tab_list.decompose()


def _normalize_field_cards(soup: BeautifulSoup, content: Tag) -> None:
    """把 OpenAPI 参数卡片标题转换为字段、类型和必填标记。"""
    for card in list(content.find_all("div")):
        direct_children = [child for child in card.children if isinstance(child, Tag)]
        if not direct_children:
            continue
        header = direct_children[0]
        if header.name != "div" or "not-prose" not in _class_names(header):
            continue
        name_span = header.find("span", class_=lambda value: _class_value_contains(value, "text-fd-primary"))
        if not isinstance(name_span, Tag):
            continue
        name = name_span.get_text(" ", strip=True)
        spans = header.find_all("span", recursive=False)
        if not name or len(spans) < 2:
            continue
        field_type = spans[1].get_text(" ", strip=True)
        required = "required" in header.get_text(" ", strip=True).casefold()
        paragraph = soup.new_tag("p")
        name_code = soup.new_tag("code")
        name_code.string = name
        paragraph.append(name_code)
        if field_type:
            type_code = soup.new_tag("code")
            type_code.string = field_type
            paragraph.append(" ")
            paragraph.append(type_code)
        if required:
            strong = soup.new_tag("strong")
            strong.string = "必填"
            paragraph.append(" ")
            paragraph.append(strong)
        header.replace_with(paragraph)


def _normalize_metadata_chips(soup: BeautifulSoup, content: Tag) -> None:
    """把范围、默认值和示例等视觉标签转换为普通段落。"""
    for chip in list(content.find_all("div")):
        classes = _class_names(chip)
        if not {"bg-fd-secondary", "rounded-lg", "text-xs"}.issubset(classes):
            continue
        direct_spans = chip.find_all("span", recursive=False)
        direct_codes = chip.find_all("code", recursive=False)
        if not direct_spans or not direct_codes:
            continue
        label = direct_spans[0].get_text(" ", strip=True)
        value = direct_codes[0].get_text("", strip=True)
        if not label or not value:
            continue
        paragraph = soup.new_tag("p")
        strong = soup.new_tag("strong")
        strong.string = f"{label}："
        code = soup.new_tag("code")
        code.string = value
        paragraph.append(strong)
        paragraph.append(code)
        chip.replace_with(paragraph)


def _normalize_code_blocks(soup: BeautifulSoup, content: Tag) -> list[tuple[str, str]]:
    """提取代码块，并使用不会与代码内容冲突的 Markdown 围栏。"""
    code_blocks: list[tuple[str, str]] = []
    for index, pre in enumerate(list(content.find_all("pre")), start=1):
        code = pre.find("code")
        source = code if isinstance(code, Tag) else pre
        raw_code = source.get_text("").strip("\n")
        language = _language_from_classes(source.get("class"))
        fence_character = chr(96)
        pattern = f"{re.escape(fence_character)}+"
        longest_run = max((len(match.group(0)) for match in re.finditer(pattern, raw_code)), default=0)
        fence = fence_character * max(3, longest_run + 1)
        placeholder = f"MAIDOCKSILICONFLOWCODEBLOCK{index:04d}TOKEN"
        replacement = soup.new_tag("p")
        replacement.string = placeholder
        pre.replace_with(replacement)
        code_blocks.append((placeholder, f"{fence}{language}\n{raw_code}\n{fence}"))
    return code_blocks


def _normalize_callouts(content: Tag) -> None:
    """把 Fumadocs 提示框转换成 Markdown 引用块。"""
    for callout in content.find_all("div"):
        classes = _class_names(callout)
        if callout.has_attr("data-callout") or "fd-callout" in classes:
            callout.name = "blockquote"
            callout.attrs = {}


def _normalize_details(soup: BeautifulSoup, content: Tag) -> None:
    """静态展开 details，避免离线 Markdown 丢失折叠内容。"""
    for details in list(content.find_all("details")):
        summary = details.find("summary", recursive=False)
        if isinstance(summary, Tag):
            paragraph = soup.new_tag("p")
            strong = soup.new_tag("strong")
            strong.string = summary.get_text(" ", strip=True)
            paragraph.append(strong)
            summary.replace_with(paragraph)
        details.unwrap()


def _normalize_media(soup: BeautifulSoup, content: Tag, source_url: str) -> None:
    """把音视频元素转换为可离线识别的绝对链接。"""
    for tag_name, label in (("audio", "音频"), ("video", "视频")):
        for media in list(content.find_all(tag_name)):
            source = media.get("src")
            if not isinstance(source, str) or not source:
                source_tag = media.find("source", src=True)
                source = source_tag.get("src") if isinstance(source_tag, Tag) else None
            if not isinstance(source, str) or not source:
                media.decompose()
                continue
            paragraph = soup.new_tag("p")
            link = soup.new_tag("a", href=urljoin(source_url, source))
            link.string = label
            paragraph.append(link)
            media.replace_with(paragraph)


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


def _remove_interface_controls(content: Tag) -> None:
    """删除复制按钮、下拉框和仅用于界面的图标。"""
    for unwanted in list(content.find_all(["button", "select", "input", "textarea", "svg"])):
        unwanted.decompose()
    for form in list(content.find_all("form")):
        form.unwrap()


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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="siliconflow-docs") as executor:
        for plan in plans:
            future = executor.submit(download_document, client, plan, attempts)
            future_map[future] = plan
        for completed, future in enumerate(as_completed(future_map), start=1):
            plan = future_map[future]
            try:
                downloaded_by_path[plan.document_path] = future.result()
                if verbose or completed % 10 == 0 or completed == len(plans):
                    print(f"下载进度：{completed}/{len(plans)}")
            except Exception as exc:
                failures.append((plan, exc))
    downloads = [downloaded_by_path[plan.document_path] for plan in plans if plan.document_path in downloaded_by_path]
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
    """识别限流、瞬时请求错误和服务端错误。"""
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


def _is_verification_page(response_text: str) -> bool:
    """识别访问验证页面。"""
    normalized = response_text.casefold()
    return any(marker in normalized for marker in VERIFICATION_MARKERS)


def _normalize_document_selector(value: str, location: SourceLocation) -> str:
    """统一命令行中的 URL、/docs 路径和纯文档路径。"""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or parsed.netloc != urlsplit(location.origin).netloc:
            raise SiliconFlowDocsError(f"非本站文档地址：{value}")
        path = parsed.path
    else:
        path = value
    prefix = f"{location.docs_path}/"
    if path.startswith(prefix):
        path = path.removeprefix(prefix)
    normalized = unquote(path).strip("/")
    if not normalized:
        raise SiliconFlowDocsError(f"文档路径无效：{value}")
    return normalized


def _node_name(node: Mapping[str, object], context: str) -> str:
    """读取目录节点的非空名称。"""
    name = node.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SiliconFlowDocsError(f"{context} 缺少有效 name")
    return name.strip()


def _node_url(node: Mapping[str, object], context: str) -> str:
    """读取目录节点的站内 URL。"""
    url = node.get("url")
    if not isinstance(url, str) or not url.startswith("/"):
        raise SiliconFlowDocsError(f"{context} 缺少有效 url")
    return url


def _class_names(tag: Tag) -> set[str]:
    """读取标签的 class 集合。"""
    classes = tag.get("class")
    if not isinstance(classes, list):
        return set()
    return {value for value in classes if isinstance(value, str)}


def _class_value_contains(value: object, expected: str) -> bool:
    """适配 BeautifulSoup class_ 回调的字符串或列表值。"""
    if isinstance(value, str):
        return expected in value.split()
    if isinstance(value, list):
        return expected in value
    return False


def _expect_mapping(value: object, context: str) -> dict[str, object]:
    """校验动态 JSON 值是字符串键映射。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SiliconFlowDocsError(f"{context} 应为 JSON 对象")
    return cast(dict[str, object], value)


def _mapping_value(container: Mapping[str, object], key: str, context: str) -> dict[str, object]:
    """读取并校验映射字段。"""
    if key not in container:
        raise SiliconFlowDocsError(f"{context} 缺少字段 {key}")
    return _expect_mapping(container[key], f"{context}.{key}")


def _sequence_value(container: Mapping[str, object], key: str, context: str) -> list[object]:
    """读取并校验数组字段。"""
    if key not in container or not isinstance(container[key], list):
        raise SiliconFlowDocsError(f"{context}.{key} 应为 JSON 数组")
    return cast(list[object], container[key])


if __name__ == "__main__":
    main()
