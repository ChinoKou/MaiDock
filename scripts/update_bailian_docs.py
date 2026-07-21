"""下载阿里云百炼中文文档，并将官方正文 HTML 转换为 Markdown。"""

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
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Comment, Tag
from bs4.element import NavigableString
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

SOURCE_URL = "https://help.aliyun.com/zh/model-studio"
FETCH_ORIGIN = "https://help.aliyun.cn"
OUTPUT_DIRECTORY = PROVIDER_DOCS_ROOT / "dashscope"
SECTIONS: tuple[str, ...] = ()
DOCUMENT_IDS: tuple[int, ...] = ()
WORKERS = 2
ATTEMPTS = 5
TIMEOUT_SECONDS = 30.0
PRUNE = False
DRY_RUN = False
VERBOSE = False
MANIFEST_NAME = ".bailian_docs.json"
MANIFEST_VERSION = 1
PRODUCT_NODE_ID = 2400256
RETRYABLE_STATUS_CODES = {405, 408, 425, 429}
VERIFICATION_MARKERS = (
    "_____tmd_____/punish",
    '"action":"captcha"',
    '"action": "captcha"',
    "window._config_",
)
SECTION_SPECS: tuple[tuple[int, str], ...] = (
    (2400262, "用户指南（模型）"),
    (2840916, "用户指南（应用）"),
    (2400264, "API参考（模型）"),
    (2863247, "API参考（应用）"),
)
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


class BailianDocsError(DocsSyncError):
    """表示百炼文档同步过程中发生了可定位的错误。"""


class RetryableBailianDocsError(BailianDocsError):
    """表示远端暂时性响应，可以按退避策略重试。"""


@dataclass(slots=True)
class SourceLocation:
    """保存百炼文档的公开来源和实际抓取端。"""

    canonical_origin: str
    fetch_origin: str
    language: str
    product_alias: str
    website: str = "cn"

    @property
    def catalog_url(self) -> str:
        """返回完整目录树接口。"""
        return f"{self.fetch_origin}/help/json/menupath.json"

    @property
    def document_url(self) -> str:
        """返回文档正文接口。"""
        return f"{self.fetch_origin}/help/json/document_detail.json"

    @property
    def catalog_alias(self) -> str:
        """返回用于定位产品目录树的稳定入口别名。"""
        return f"/{self.product_alias}/what-is-model-studio"

    @property
    def source_url(self) -> str:
        """返回用户可访问的产品文档首页。"""
        return f"{self.canonical_origin}/{self.language}/{self.product_alias}"

    def canonical_document_url(self, url_path: str) -> str:
        """把目录树中的路径转换成公开来源 URL。"""
        return urljoin(f"{self.canonical_origin}/", url_path.lstrip("/"))


@dataclass(slots=True)
class DocumentPlan:
    """描述一篇百炼文档的远端身份和本地路径。"""

    node_id: int
    node_type: int
    title: str
    section: str
    alias: str
    url_path: str
    source_url: str
    relative_path: Path


@dataclass(slots=True)
class DownloadedDocument:
    """保存转换后的 Markdown 与远端元数据。"""

    plan: DocumentPlan
    document_title: str
    last_modified: int
    markdown: str
    digest: str


def parse_source_url(source_url: str, fetch_origin: str = FETCH_ORIGIN) -> SourceLocation:
    """解析百炼文档首页和抓取端地址。"""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BailianDocsError(f"文档地址无效：{source_url}")
    match = re.fullmatch(r"/([^/]+)/([^/]+)/?", parsed.path)
    if match is None:
        raise BailianDocsError("文档地址路径必须是 /<language>/<product>")
    language, product_alias = match.groups()
    if language != "zh" or product_alias != "model-studio":
        raise BailianDocsError("当前脚本只支持 /zh/model-studio 中文文档")

    fetch = urlsplit(fetch_origin)
    if fetch.scheme not in {"http", "https"} or not fetch.netloc or fetch.path not in {"", "/"}:
        raise BailianDocsError(f"抓取端地址无效：{fetch_origin}")
    canonical_origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    normalized_fetch_origin = urlunsplit((fetch.scheme, fetch.netloc, "", "", "")).rstrip("/")
    return SourceLocation(
        canonical_origin=canonical_origin,
        fetch_origin=normalized_fetch_origin,
        language=language,
        product_alias=product_alias,
    )


def safe_path_component(value: str, max_length: int = 100) -> str:
    """将文档标题转换成 Windows 可用的路径段。"""
    component = value.translate(INVALID_FILENAME_TRANSLATION)
    component = re.sub(r"[\x00-\x1f]+", " ", component)
    component = re.sub(r"\s+", " ", component).strip(" .")
    if not component:
        component = "未命名"
    if component.upper() in WINDOWS_RESERVED_NAMES:
        component = f"_{component}"
    component = component[:max_length].rstrip(" .")
    return component or "未命名"


def fetch_catalog(client: httpx.Client, location: SourceLocation, attempts: int) -> dict[str, object]:
    """获取并校验百炼完整目录树。"""
    params = {
        "alias": location.catalog_alias,
        "website": location.website,
        "language": location.language,
    }
    return fetch_parsed(
        client=client,
        url=location.catalog_url,
        params=params,
        attempts=attempts,
        context="百炼完整目录树",
        parser=lambda payload: _api_data(payload, "百炼完整目录树"),
    )


def build_document_plans(
    catalog: Mapping[str, object],
    location: SourceLocation,
    selected_sections: set[str] | None = None,
) -> list[DocumentPlan]:
    """按官方树顺序生成四章的本地文件计划。"""
    root_id = _int_value(catalog, "id", "目录树根节点")
    if root_id != PRODUCT_NODE_ID:
        raise BailianDocsError(f"目录树根节点应为 {PRODUCT_NODE_ID}，实际为 {root_id}")
    root_alias = _string_value(catalog, "alias", "目录树根节点")
    if root_alias.rstrip("/") != f"/{location.product_alias}":
        raise BailianDocsError(f"目录树产品别名不匹配：{root_alias}")

    raw_sections = _sequence_value(catalog, "children", "目录树根节点")
    actual_sections: list[tuple[int, str]] = []
    section_nodes: dict[str, dict[str, object]] = {}
    for index, raw_section in enumerate(raw_sections, start=1):
        section = _expect_mapping(raw_section, f"目录树第 {index} 个章节")
        section_id = _int_value(section, "id", f"目录树第 {index} 个章节")
        section_title = _string_value(section, "title", f"目录树第 {index} 个章节")
        actual_sections.append((section_id, section_title))
        section_nodes[section_title] = section
    if tuple(actual_sections) != SECTION_SPECS:
        raise BailianDocsError(f"百炼四章结构已变化：{actual_sections}")

    available_sections = {title for _, title in SECTION_SPECS}
    if selected_sections is not None:
        unknown = selected_sections - available_sections
        if unknown:
            missing = "、".join(sorted(unknown))
            available = "、".join(title for _, title in SECTION_SPECS)
            raise BailianDocsError(f"目录中找不到栏目：{missing}；可用栏目：{available}")
    chosen_sections = selected_sections or available_sections

    plans: list[DocumentPlan] = []
    seen_ids: set[int] = set()
    seen_urls: set[str] = set()
    seen_paths: set[str] = set()
    for _, section_title in SECTION_SPECS:
        if section_title not in chosen_sections:
            continue
        section = section_nodes[section_title]
        section_directory = Path(safe_path_component(section_title))
        _walk_nodes(
            raw_nodes=_sequence_value(section, "children", f"章节 {section_title}"),
            parent_path=section_directory,
            section=section_title,
            location=location,
            plans=plans,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_paths=seen_paths,
        )
    if not plans:
        raise BailianDocsError("百炼目录树中没有找到可下载的有效文档")
    return plans


def extract_downloaded_document(payload: Mapping[str, object], plan: DocumentPlan) -> DownloadedDocument:
    """校验正文接口数据并转换成 Markdown。"""
    data = _api_data(payload, f"文档 {plan.node_id} {plan.title}")
    node_id = _int_value(data, "nodeId", f"文档 {plan.node_id}")
    if node_id != plan.node_id:
        raise BailianDocsError(f"文档节点不匹配：请求 {plan.node_id}，返回 {node_id}")
    alias = _string_value(data, "alias", f"文档 {plan.node_id}")
    if alias.rstrip("/") != plan.alias.rstrip("/"):
        raise BailianDocsError(f"文档 {plan.node_id} 别名不匹配：{alias}")
    url_path = _string_value(data, "url", f"文档 {plan.node_id}")
    if url_path.rstrip("/") != plan.url_path.rstrip("/"):
        raise BailianDocsError(f"文档 {plan.node_id} URL 不匹配：{url_path}")

    document_title = _string_value(data, "docTitle", f"文档 {plan.node_id}").strip()
    content_html = _string_value(data, "content", f"文档 {plan.node_id}")
    last_modified = _int_value(data, "lastModifiedTime", f"文档 {plan.node_id}")
    if not document_title:
        raise BailianDocsError(f"文档 {plan.node_id} 的标题为空")
    if not content_html.strip():
        raise BailianDocsError(f"文档 {plan.node_id} 的正文为空")

    markdown = convert_document_html(content_html, document_title, plan.source_url, plan.node_id)
    digest = sha256(markdown.encode("utf-8")).hexdigest()
    return DownloadedDocument(
        plan=plan,
        document_title=document_title,
        last_modified=last_modified,
        markdown=markdown,
        digest=digest,
    )


def convert_document_html(content_html: str, title: str, source_url: str, node_id: int) -> str:
    """把百炼正文 HTML 转换成适合离线阅读的 Markdown。"""
    soup = BeautifulSoup(content_html, "html.parser")
    content = soup.select_one("div.icms-help-docs-content")
    if not isinstance(content, Tag):
        raise BailianDocsError(f"文档 {node_id} 未找到 div.icms-help-docs-content")

    for comment in content.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()
    for unwanted in list(content.find_all(["script", "style", "noscript", "template"])):
        unwanted.decompose()
    _normalize_tabbed_code_blocks(soup, content, node_id)
    _normalize_code_blocks(soup, content)
    _normalize_callouts(content)
    _normalize_hetu(soup, content, source_url, node_id)
    _normalize_media(soup, content, source_url)
    _normalize_forms(soup, content)
    _normalize_inputs(content)
    _normalize_links(content, source_url)

    unprocessed = content.find(["input", "label", "select", "textarea", "button", "hetu", "script", "style"])
    if isinstance(unprocessed, Tag):
        raise BailianDocsError(f"文档 {node_id} 仍包含未处理的 {unprocessed.name} 组件")

    converted = markdownify(
        str(content),
        heading_style="ATX",
        bullets="-",
        code_language_callback=_code_language,
    ).strip()
    markdown = f"# {title}\n"
    if converted:
        markdown += f"\n{converted}\n"
    return markdown


def fetch_parsed[ParsedValue](
    client: httpx.Client,
    url: str,
    params: Mapping[str, str | int],
    attempts: int,
    context: str,
    parser: Callable[[Mapping[str, object]], ParsedValue],
) -> ParsedValue:
    """请求 JSON 接口，并只对明确的暂时性故障进行重试。"""
    if attempts < 1:
        raise BailianDocsError("attempts 必须大于等于 1")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        verification_page = False
        try:
            response = client.get(url, params=params)
            verification_page = _is_verification_page(response.text)
            if verification_page:
                raise RetryableBailianDocsError("触发阿里云访问验证页")
            if _is_retryable_status(response.status_code):
                raise RetryableBailianDocsError(f"服务端暂时返回 HTTP {response.status_code}")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").casefold()
            if "application/json" not in content_type:
                raise RetryableBailianDocsError(f"接口返回了非 JSON 内容：{content_type or '未知类型'}")
            try:
                raw_payload = response.json()
            except json.JSONDecodeError as exc:
                raise RetryableBailianDocsError(f"接口返回的 JSON 无法解析：{exc}") from exc
            payload = _expect_mapping(raw_payload, context)
            return parser(payload)
        except (httpx.RequestError, RetryableBailianDocsError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(_retry_delay(response, attempt, verification_page))
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if attempt >= attempts or not _is_retryable_status(exc.response.status_code):
                break
            time.sleep(_retry_delay(response, attempt, verification_page))
    raise BailianDocsError(f"请求失败（{context}，{url}）：{last_error}") from last_error


def download_document(
    client: httpx.Client, location: SourceLocation, plan: DocumentPlan, attempts: int
) -> DownloadedDocument:
    """下载并转换一篇百炼文档。"""
    params = {
        "nodeId": plan.node_id,
        "website": location.website,
        "language": location.language,
        "pageNum": 1,
        "pageSize": 20,
    }
    return fetch_parsed(
        client=client,
        url=location.document_url,
        params=params,
        attempts=attempts,
        context=f"文档 {plan.node_id} {plan.title}",
        parser=lambda payload: extract_downloaded_document(payload, plan),
    )


def load_manifest(output_root: Path) -> dict[str, object]:
    """读取既有清单；首次运行时返回空清单。"""
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.exists():
        return {"format_version": MANIFEST_VERSION, "documents": {}}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BailianDocsError(f"清单读取失败：{manifest_path}：{exc}") from exc
    manifest = _expect_mapping(parsed, str(manifest_path))
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise BailianDocsError(f"不支持的清单版本：{manifest.get('format_version')}")
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

        document_key = str(document.plan.node_id)
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
            "section": document.plan.section,
            "node_type": document.plan.node_type,
            "path": document.plan.relative_path.as_posix(),
            "source_url": document.plan.source_url,
            "last_modified": document.last_modified,
            "sha256": document.digest,
        }

    if prune and allow_prune:
        planned_ids = {str(plan.node_id) for plan in plans}
        for document_key, raw_record in list(manifest_documents.items()):
            if document_key in planned_ids or not isinstance(raw_record, dict):
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
        "source_url": location.source_url,
        "fetch_origin": location.fetch_origin,
        "language": location.language,
        "documents": manifest_documents,
    }
    manifest_bytes = encode_stable_manifest(manifest, manifest_payload)
    atomic_write(output_root / MANIFEST_NAME, manifest_bytes)
    remove_empty_directories(output_root)
    return stats


def create_http_client(location: SourceLocation, workers: int, timeout: float) -> httpx.Client:
    """创建可供下载线程共享的 HTTP 客户端。"""
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        headers={
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": f"{location.fetch_origin}/{location.language}/{location.product_alias}/",
            "User-Agent": "MaiDock-Bailian-Docs-Sync/1.0",
        },
    )


def _validate_sync_settings(
    *,
    workers: int,
    attempts: int,
    timeout: float,
    document_ids: tuple[int, ...],
    prune: bool,
) -> None:
    """校验百炼同步参数。"""
    if not 1 <= workers <= 8:
        raise BailianDocsError("workers 必须在 1 到 8 之间")
    if not 1 <= attempts <= 10:
        raise BailianDocsError("attempts 必须在 1 到 10 之间")
    if timeout <= 0:
        raise BailianDocsError("timeout 必须大于 0")
    if any(document_id <= 0 for document_id in document_ids):
        raise BailianDocsError("document_ids 必须全部为正整数")
    if prune and document_ids:
        raise BailianDocsError("指定 document_ids 时不能同时启用 prune")


def run(
    *,
    source_url: str,
    fetch_origin: str,
    output: Path,
    sections: tuple[str, ...],
    document_ids: tuple[int, ...],
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
        document_ids=document_ids,
        prune=prune,
    )
    location = parse_source_url(source_url, fetch_origin)
    output_root = project_output_path(output)
    selected_sections = set(sections)
    selected_document_ids = set(document_ids)

    with create_http_client(location, workers, timeout) as client:
        catalog = fetch_catalog(client, location, attempts)
        all_plans = build_document_plans(catalog, location, selected_sections or None)
        active_sections = selected_sections or {plan.section for plan in all_plans}
        if selected_document_ids:
            plans = [plan for plan in all_plans if plan.node_id in selected_document_ids]
            missing_ids = selected_document_ids - {plan.node_id for plan in plans}
            if missing_ids:
                missing = "、".join(str(value) for value in sorted(missing_ids))
                raise BailianDocsError(f"目录中找不到有效文档：{missing}")
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
                    print(f"{plan.node_id}\t{plan.relative_path.as_posix()}")
            print("dry-run 完成，未下载正文或修改文件。")
            return 0

        downloads, failures = _download_all(
            client=client,
            location=location,
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
        allow_prune=not failures and not selected_document_ids,
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
            print(f"- {plan.node_id} {plan.title}：{error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    """使用模块级代码常量执行文档同步。"""
    exit_code = run(
        source_url=SOURCE_URL,
        fetch_origin=FETCH_ORIGIN,
        output=OUTPUT_DIRECTORY,
        sections=SECTIONS,
        document_ids=DOCUMENT_IDS,
        workers=WORKERS,
        attempts=ATTEMPTS,
        timeout=TIMEOUT_SECONDS,
        prune=PRUNE,
        dry_run=DRY_RUN,
        verbose=VERBOSE,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


def _walk_nodes(
    raw_nodes: Sequence[object],
    parent_path: Path,
    section: str,
    location: SourceLocation,
    plans: list[DocumentPlan],
    seen_ids: set[int],
    seen_urls: set[str],
    seen_paths: set[str],
) -> None:
    """递归遍历官方目录节点，并保留同级排序编号。"""
    for index, raw_node in enumerate(raw_nodes, start=1):
        context = f"章节 {section} 的第 {index} 个节点"
        node = _expect_mapping(raw_node, context)
        node_id = _int_value(node, "id", context)
        node_type = _int_value(node, "nodeType", context)
        if node_type not in {1, 8}:
            raise BailianDocsError(f"{context} 包含未知 nodeType：{node_type}")
        title = _string_value(node, "title", context).strip()
        if not title:
            raise BailianDocsError(f"{context} 的标题为空")
        valid_document = _bool_value(node, "validDocument", context)
        raw_children = node.get("children", [])
        if not isinstance(raw_children, list):
            raise BailianDocsError(f"{context}.children 应为 JSON 数组")
        children = cast(list[object], raw_children)
        if node_type == 1 and children:
            raise BailianDocsError(f"普通文档节点 {node_id} 意外包含子节点")

        numbered_title = safe_path_component(f"{index}.{title}")
        if node_type == 8:
            node_path = parent_path / numbered_title
            relative_path = node_path / "index.md"
            child_parent = node_path
        else:
            relative_path = parent_path / f"{numbered_title}.md"
            child_parent = parent_path

        if valid_document:
            alias = _string_value(node, "alias", context)
            url_path = _string_value(node, "url", context)
            expected_alias_prefix = f"/{location.product_alias}/"
            expected_url_prefix = f"/{location.language}/{location.product_alias}/"
            if not alias.startswith(expected_alias_prefix):
                raise BailianDocsError(f"文档 {node_id} 别名越界：{alias}")
            if not url_path.startswith(expected_url_prefix):
                raise BailianDocsError(f"文档 {node_id} URL 越界：{url_path}")
            normalized_url = url_path.rstrip("/").casefold()
            normalized_path = relative_path.as_posix().casefold()
            if node_id in seen_ids:
                raise BailianDocsError(f"目录树重复声明文档节点：{node_id}")
            if normalized_url in seen_urls:
                raise BailianDocsError(f"目录树重复声明文档 URL：{url_path}")
            if normalized_path in seen_paths:
                raise BailianDocsError(f"多个文档映射到了同一路径：{relative_path.as_posix()}")
            seen_ids.add(node_id)
            seen_urls.add(normalized_url)
            seen_paths.add(normalized_path)
            plans.append(
                DocumentPlan(
                    node_id=node_id,
                    node_type=node_type,
                    title=title,
                    section=section,
                    alias=alias,
                    url_path=url_path,
                    source_url=location.canonical_document_url(url_path),
                    relative_path=relative_path,
                )
            )
        if children:
            _walk_nodes(
                raw_nodes=children,
                parent_path=child_parent,
                section=section,
                location=location,
                plans=plans,
                seen_ids=seen_ids,
                seen_urls=seen_urls,
                seen_paths=seen_paths,
            )


def _normalize_tabbed_code_blocks(soup: BeautifulSoup, content: Tag, node_id: int) -> None:
    """将代码 Tabs 展开为带粗体语言标签的连续代码块。"""
    for container in list(content.select("div.tabbed-codeblock-box")):
        labels = [tag for tag in container.find_all("label", recursive=False) if isinstance(tag, Tag)]
        items = [
            tag
            for tag in container.find_all("div", recursive=False)
            if isinstance(tag, Tag) and "codeblock-item" in (tag.get("class") or [])
        ]
        if len(labels) != len(items) or not items:
            raise BailianDocsError(f"文档 {node_id} 的代码 Tabs 标签与代码块数量不一致")
        for label, item in zip(labels, items, strict=True):
            label_text = label.get_text(" ", strip=True)
            if not label_text:
                pre = item.find("pre")
                label_text = _language_from_element(pre) if isinstance(pre, Tag) else ""
            if not label_text:
                raise BailianDocsError(f"文档 {node_id} 的代码 Tab 标签为空且代码块缺少 syntax")
            paragraph = soup.new_tag("p")
            strong = soup.new_tag("strong")
            strong.string = label_text
            paragraph.append(strong)
            item.insert_before(paragraph)
            item.unwrap()
        for label in labels:
            label.decompose()
        for input_tag in list(container.find_all("input", recursive=False)):
            input_tag.decompose()
        for tab_box in list(container.select(":scope > div.tab-box")):
            tab_box.decompose()
        container.unwrap()


def _normalize_code_blocks(soup: BeautifulSoup, content: Tag) -> None:
    """重建代码块，避免高亮节点破坏原始缩进和语言。"""
    for pre in list(content.find_all("pre")):
        language = _language_from_element(pre)
        code = pre.find("code")
        code_text = code.get_text("", strip=False) if isinstance(code, Tag) else pre.get_text("", strip=False)
        clean_pre = soup.new_tag("pre")
        clean_code = soup.new_tag("code")
        if language:
            clean_code["class"] = f"language-{language}"
        clean_code.string = code_text
        clean_pre.append(clean_code)
        pre.replace_with(clean_pre)


def _normalize_callouts(content: Tag) -> None:
    """把说明、警告等提示框转换成 Markdown 引用块。"""
    for callout in list(content.select("div.note")):
        for icon in list(callout.select(".note-icon-wrapper")):
            icon.decompose()
        for wrapper in list(callout.select(".noteContentSpan")):
            wrapper.unwrap()
        callout.name = "blockquote"
        callout.attrs = {}


def _normalize_hetu(soup: BeautifulSoup, content: Tag, source_url: str, node_id: int) -> None:
    """将 Hetu 公式和流程图转换成静态 Markdown 内容。"""
    for hetu_node in list(content.find_all("hetu")):
        formula = hetu_node.get("formula")
        if isinstance(formula, str) and formula.strip():
            clean_formula = formula.strip()
            style = str(hetu_node.get("style", "")).casefold()
            if "display:block" in style:
                replacement = f"\n\n$\n{clean_formula}\n$\n\n"
            else:
                replacement = f"${clean_formula}$"
            hetu_node.replace_with(NavigableString(replacement))
            continue

        component_type = hetu_node.get("type")
        if component_type != "flowchart":
            raise BailianDocsError(f"文档 {node_id} 包含未知 Hetu 组件：{component_type}")
        image = hetu_node.find("img", src=True)
        if isinstance(image, Tag):
            hetu_node.unwrap()
            continue
        paragraph = soup.new_tag("p")
        link = soup.new_tag("a", href=source_url)
        link.string = "流程图：查看官网原文"
        paragraph.append(link)
        hetu_node.replace_with(paragraph)


def _normalize_media(soup: BeautifulSoup, content: Tag, source_url: str) -> None:
    """把音视频和嵌入内容转换成可离线识别的链接。"""
    labels = {"audio": "音频", "video": "视频", "iframe": "嵌入内容"}
    for tag_name, label in labels.items():
        for media in list(content.find_all(tag_name)):
            raw_source = media.get("src")
            if not isinstance(raw_source, str) or not raw_source:
                source_tag = media.find("source", src=True)
                raw_source = source_tag.get("src") if isinstance(source_tag, Tag) else None
            target_url = urljoin(source_url, raw_source) if isinstance(raw_source, str) and raw_source else source_url
            raw_title = media.get("title") or media.get("aria-label")
            title = str(raw_title).strip() if raw_title is not None else "查看官网原文"
            paragraph = soup.new_tag("p")
            link = soup.new_tag("a", href=target_url)
            link.string = f"{label}：{title}"
            paragraph.append(link)
            media.replace_with(paragraph)


def _normalize_forms(soup: BeautifulSoup, content: Tag) -> None:
    """将内嵌交互表单展开成可离线阅读的静态内容。"""
    for select in list(content.find_all("select")):
        option_list = soup.new_tag("ul")
        for option in select.find_all("option"):
            option_text = option.get_text(" ", strip=True)
            if not option_text:
                continue
            item = soup.new_tag("li")
            if option.has_attr("selected"):
                strong = soup.new_tag("strong")
                strong.string = f"{option_text}（默认）"
                item.append(strong)
            else:
                item.string = option_text
            option_list.append(item)
        if option_list.contents:
            select.replace_with(option_list)
        else:
            select.decompose()

    for textarea in list(content.find_all("textarea")):
        value = textarea.get_text("", strip=False).strip()
        if not value:
            textarea.decompose()
            continue
        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = value
        pre.append(code)
        textarea.replace_with(pre)

    for label in list(content.find_all("label")):
        label.name = "p"
        label.attrs = {}
    for button in list(content.find_all("button")):
        button.decompose()


def _normalize_inputs(content: Tag) -> None:
    """保留任务列表状态，并删除其余仅用于网页交互的输入框。"""
    for input_tag in list(content.find_all("input")):
        input_type = str(input_tag.get("type", "")).casefold()
        if input_type == "checkbox":
            marker = "[x] " if input_tag.has_attr("checked") else "[ ] "
            input_tag.replace_with(NavigableString(marker))
        else:
            input_tag.decompose()


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


def _code_language(element: Tag) -> str:
    """为 markdownify 的 fenced code block 提取语言。"""
    language = _language_from_element(element)
    if language:
        return language
    code = element.find("code")
    return _language_from_element(code) if isinstance(code, Tag) else ""


def _language_from_element(element: Tag) -> str:
    """从 syntax、class 或 outputclass 中提取代码语言。"""
    syntax = element.get("syntax")
    if isinstance(syntax, str) and syntax.strip():
        return syntax.strip()
    classes = element.get("class")
    if isinstance(classes, list):
        for class_name in classes:
            if isinstance(class_name, str) and class_name.startswith("language-"):
                return class_name.removeprefix("language-")
    output_class = element.get("outputclass")
    if isinstance(output_class, str):
        for class_name in output_class.split():
            if class_name.startswith("language-"):
                return class_name.removeprefix("language-")
    return ""


def _download_all(
    client: httpx.Client,
    location: SourceLocation,
    plans: Sequence[DocumentPlan],
    attempts: int,
    workers: int,
    verbose: bool,
) -> tuple[list[DownloadedDocument], list[tuple[DocumentPlan, Exception]]]:
    """并发下载全部计划，并保留精确的失败信息。"""
    downloaded_by_id: dict[int, DownloadedDocument] = {}
    failures: list[tuple[DocumentPlan, Exception]] = []
    future_map: dict[Future[DownloadedDocument], DocumentPlan] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bailian-docs") as executor:
        for plan in plans:
            future = executor.submit(download_document, client, location, plan, attempts)
            future_map[future] = plan
        for completed, future in enumerate(as_completed(future_map), start=1):
            plan = future_map[future]
            try:
                downloaded_by_id[plan.node_id] = future.result()
                if verbose or completed % 25 == 0 or completed == len(plans):
                    print(f"下载进度：{completed}/{len(plans)}")
            except Exception as exc:
                failures.append((plan, exc))
    downloads = [downloaded_by_id[plan.node_id] for plan in plans if plan.node_id in downloaded_by_id]
    return downloads, failures


def _api_data(payload: Mapping[str, object], context: str) -> dict[str, object]:
    """校验阿里云帮助中心 JSON 信封并返回 data。"""
    code = payload.get("code")
    success = payload.get("success")
    if code != 200 or success is not True:
        message = payload.get("msg")
        raise BailianDocsError(f"{context} 接口失败：code={code}，success={success}，msg={message}")
    return _mapping_value(payload, "data", context)


def _is_verification_page(response_text: str) -> bool:
    """识别 HTTP 200 但实际要求验证码的阿里云拦截页。"""
    lowered = response_text.casefold()
    return any(marker.casefold() in lowered for marker in VERIFICATION_MARKERS)


def _retry_delay(response: httpx.Response | None, attempt: int, verification_page: bool) -> float:
    """优先采用 Retry-After；验证码页使用更长的指数退避。"""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, min(float(retry_after), 60.0))
            except ValueError:
                pass
    multiplier = 5 if verification_page else 1
    return min(float(multiplier * (2 ** (attempt - 1))), 60.0)


def _is_retryable_status(status_code: int) -> bool:
    """识别文档站实测会瞬时返回的状态码和服务端错误。"""
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


def _expect_mapping(value: object, context: str) -> dict[str, object]:
    """校验动态 JSON 值是字符串键映射。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BailianDocsError(f"{context} 应为 JSON 对象")
    return cast(dict[str, object], value)


def _mapping_value(container: Mapping[str, object], key: str, context: str) -> dict[str, object]:
    """读取并校验映射字段。"""
    if key not in container:
        raise BailianDocsError(f"{context} 缺少字段 {key}")
    return _expect_mapping(container[key], f"{context}.{key}")


def _sequence_value(container: Mapping[str, object], key: str, context: str) -> list[object]:
    """读取并校验数组字段。"""
    if key not in container:
        raise BailianDocsError(f"{context} 缺少字段 {key}")
    value = container[key]
    if not isinstance(value, list):
        raise BailianDocsError(f"{context}.{key} 应为 JSON 数组")
    return cast(list[object], value)


def _string_value(container: Mapping[str, object], key: str, context: str) -> str:
    """读取并校验字符串字段。"""
    if key not in container:
        raise BailianDocsError(f"{context} 缺少字段 {key}")
    value = container[key]
    if not isinstance(value, str):
        raise BailianDocsError(f"{context}.{key} 应为字符串")
    return value


def _int_value(container: Mapping[str, object], key: str, context: str) -> int:
    """读取并校验整数字段，拒绝把 bool 当作整数。"""
    if key not in container:
        raise BailianDocsError(f"{context} 缺少字段 {key}")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise BailianDocsError(f"{context}.{key} 应为整数")
    return value


def _bool_value(container: Mapping[str, object], key: str, context: str) -> bool:
    """读取并校验布尔字段。"""
    if key not in container:
        raise BailianDocsError(f"{context} 缺少字段 {key}")
    value = container[key]
    if not isinstance(value, bool):
        raise BailianDocsError(f"{context}.{key} 应为布尔值")
    return value


if __name__ == "__main__":
    main()
