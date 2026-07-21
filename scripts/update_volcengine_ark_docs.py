"""下载并增量更新火山方舟文档中心的官方 Markdown。"""

import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
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

SOURCE_URL = "https://docs.volcengine.com/docs/82379?lang=zh"
OUTPUT_DIRECTORY = PROVIDER_DOCS_ROOT / "volcengine_ark"
SECTIONS: tuple[str, ...] = ()
DOCUMENT_IDS: tuple[int, ...] = ()
WORKERS = 4
ATTEMPTS = 3
TIMEOUT_SECONDS = 30.0
PRUNE = False
DRY_RUN = False
VERBOSE = False
MANIFEST_NAME = ".volcengine_ark_docs.json"
MANIFEST_VERSION = 1
LAYOUT_KEY = "docs/(libid)/layout"
PAGE_KEY = "docs/(libid)/(docid$)/page"
RETRYABLE_STATUS_CODES = {405, 408, 425, 429}
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


class ArkDocsError(DocsSyncError):
    """表示文档同步过程中发生了可定位的错误。"""


@dataclass(slots=True)
class SourceLocation:
    """保存文档库地址及语言参数。"""

    origin: str
    library_id: int
    lang: str

    @property
    def catalog_url(self) -> str:
        """返回文档库首页地址。"""
        return f"{self.origin}/docs/{self.library_id}?{urlencode({'lang': self.lang})}"

    @property
    def catalog_data_url(self) -> str:
        """返回文档库目录 loader 地址。"""
        return self._loader_url(f"/docs/{self.library_id}", LAYOUT_KEY)

    def document_url(self, document_id: int) -> str:
        """返回单篇文档地址。"""
        return f"{self.origin}/docs/{self.library_id}/{document_id}?{urlencode({'lang': self.lang})}"

    def document_data_url(self, document_id: int) -> str:
        """返回单篇文档 loader 地址。"""
        return self._loader_url(f"/docs/{self.library_id}/{document_id}", PAGE_KEY)

    def _loader_url(self, path: str, loader: str) -> str:
        query = urlencode({"lang": self.lang, "__loader": loader, "__ssrDirect": "true"})
        return f"{self.origin}{path}?{query}"


@dataclass(slots=True)
class DocumentPlan:
    """描述一篇文档的远端身份和本地目标路径。"""

    document_id: int
    title: str
    section: str
    relative_path: Path


@dataclass(slots=True)
class DownloadedDocument:
    """保存已解析的官方 Markdown 与元数据。"""

    plan: DocumentPlan
    markdown: str
    updated_time: str
    digest: str
    source_url: str


def parse_source_url(source_url: str) -> SourceLocation:
    """解析文档库 URL，并提取库 ID 与语言。"""
    parsed = urlsplit(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ArkDocsError(f"文档地址无效：{source_url}")

    match = re.fullmatch(r"/docs/(\d+)(?:/\d+)?/?", parsed.path)
    if match is None:
        raise ArkDocsError("文档地址路径必须是 /docs/<library_id> 或 /docs/<library_id>/<document_id>")

    lang_values = parse_qs(parsed.query).get("lang", ["zh"])
    lang = lang_values[0].strip()
    if not lang:
        raise ArkDocsError("文档地址的 lang 参数不能为空")

    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    return SourceLocation(origin=origin, library_id=int(match.group(1)), lang=lang)


def extract_loader_data(response_text: str, context: str) -> dict[str, object]:
    """解析官网路由 loader 返回的 JSON 对象。"""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ArkDocsError(f"{context} loader 未返回有效 JSON：{exc}") from exc
    return _expect_mapping(parsed, f"{context} loader")


def safe_path_component(title: str, max_length: int = 100) -> str:
    """将远端标题转换成 Windows 与 POSIX 均可用的路径段。"""
    component = title.translate(INVALID_FILENAME_TRANSLATION)
    component = re.sub(r"[\x00-\x1f]+", " ", component)
    component = re.sub(r"\s+", " ", component).strip(" .")
    if not component:
        component = "未命名"
    if component.upper() in WINDOWS_RESERVED_NAMES:
        component = f"_{component}"
    component = component[:max_length].rstrip(" .")
    return component or "未命名"


def build_document_plans(layout: Mapping[str, object], selected_sections: set[str] | None = None) -> list[DocumentPlan]:
    """根据扁平目录映射构造有序的本地文档路径。"""
    current_library = _mapping_value(layout, "curLib", LAYOUT_KEY)
    second_nav = _sequence_value(current_library, "SecondNav", "curLib")
    doc_list_map = _mapping_value(layout, "docListMap", LAYOUT_KEY)

    available_sections: dict[str, Mapping[str, object]] = {}
    for raw_section in second_nav:
        section = _expect_mapping(raw_section, "SecondNav 项")
        name = _string_value(section, "Name", "SecondNav 项")
        available_sections[name] = section

    requested_sections = selected_sections or set(available_sections)
    missing_sections = requested_sections - set(available_sections)
    if missing_sections:
        missing = "、".join(sorted(missing_sections))
        available = "、".join(available_sections)
        raise ArkDocsError(f"找不到栏目：{missing}；可用栏目：{available}")

    plans: list[DocumentPlan] = []
    relative_paths: set[str] = set()
    for section_name, section in available_sections.items():
        if section_name not in requested_sections:
            continue
        section_id = _int_value(section, "ID", f"栏目 {section_name}")
        node_map = _mapping_value(doc_list_map, str(section_id), "docListMap")
        root_node = _mapping_value(node_map, "0", f"栏目 {section_name}")
        root_children = _sequence_value(root_node, "children", f"栏目 {section_name} 根节点")
        _walk_nodes(
            node_map=node_map,
            child_ids=root_children,
            section_name=section_name,
            parent_parts=[safe_path_component(section_name)],
            ancestors=set(),
            plans=plans,
            relative_paths=relative_paths,
        )
    return plans


def extract_downloaded_document(
    page_data: Mapping[str, object], plan: DocumentPlan, source_url: str
) -> DownloadedDocument:
    """从单篇文档页面数据中提取并校验官方 Markdown。"""
    current_doc = _mapping_value(page_data, "curDoc", PAGE_KEY)

    document_id = _int_value(current_doc, "DocumentID", "curDoc")
    if document_id != plan.document_id:
        raise ArkDocsError(f"请求文档 {plan.document_id}，页面却返回了文档 {document_id}")

    title = _string_value(current_doc, "Title", "curDoc")
    if title != plan.title:
        raise ArkDocsError(f"文档 {plan.document_id} 的目录标题为“{plan.title}”，页面标题为“{title}”，请重新同步目录")

    markdown = _string_value(current_doc, "MDContent", "curDoc")
    if not markdown.strip():
        raise ArkDocsError(f"文档 {plan.document_id} 的 MDContent 为空")
    normalized_markdown = markdown.rstrip() + "\n"
    updated_time = _string_value(current_doc, "UpdatedTime", "curDoc")
    digest = sha256(normalized_markdown.encode("utf-8")).hexdigest()
    return DownloadedDocument(
        plan=plan,
        markdown=normalized_markdown,
        updated_time=updated_time,
        digest=digest,
        source_url=source_url,
    )


def fetch_loader_data(client: httpx.Client, url: str, attempts: int, context: str) -> dict[str, object]:
    """请求 loader 并重试暂时性 HTTP、网络或 JSON 结构错误。"""
    if attempts < 1:
        raise ArkDocsError("attempts 必须大于等于 1")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(url)
            if _is_retryable_status(response.status_code):
                raise ArkDocsError(f"服务端暂时返回 HTTP {response.status_code}")
            response.raise_for_status()
            return extract_loader_data(response.text, context)
        except (httpx.RequestError, httpx.HTTPStatusError, ArkDocsError) as exc:
            last_error = exc
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (_is_retryable_status(exc.response.status_code))
            if attempt >= attempts or not retryable:
                break
            time.sleep(_retry_delay(response, attempt))

    raise ArkDocsError(f"请求失败（{url}）：{last_error}") from last_error


def download_document(
    client: httpx.Client, location: SourceLocation, plan: DocumentPlan, attempts: int
) -> DownloadedDocument:
    """下载并解析一篇文档。"""
    source_url = location.document_url(plan.document_id)
    page_data = fetch_loader_data(
        client,
        location.document_data_url(plan.document_id),
        attempts,
        f"文档 {plan.document_id}",
    )
    return extract_downloaded_document(page_data, plan, source_url)


def load_manifest(output_root: Path) -> dict[str, object]:
    """读取既有清单；首次运行时返回空清单。"""
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.exists():
        return {"format_version": MANIFEST_VERSION, "documents": {}}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArkDocsError(f"清单读取失败：{manifest_path}：{exc}") from exc
    manifest = _expect_mapping(parsed, str(manifest_path))
    version = manifest.get("format_version")
    if version != MANIFEST_VERSION:
        raise ArkDocsError(f"不支持的清单版本：{version}")
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
    """将成功下载的内容增量写入磁盘并更新清单。"""
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

        document_key = str(document.plan.document_id)
        old_record_raw = manifest_documents.get(document_key)
        if isinstance(old_record_raw, dict):
            old_record = cast(dict[str, object], old_record_raw)
            old_path_raw = old_record.get("path")
            if isinstance(old_path_raw, str) and old_path_raw != document.plan.relative_path.as_posix():
                if remove_tracked_file(output_root, old_path_raw, str(old_record.get("sha256", ""))):
                    stats.removed += 1
                else:
                    stats.preserved += 1

        manifest_documents[document_key] = {
            "title": document.plan.title,
            "section": document.plan.section,
            "path": document.plan.relative_path.as_posix(),
            "source_url": document.source_url,
            "updated_time": document.updated_time,
            "sha256": document.digest,
        }

    if prune and allow_prune:
        planned_ids = {str(plan.document_id) for plan in plans}
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
        "library_id": location.library_id,
        "lang": location.lang,
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
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "User-Agent": "MaiDock-Ark-Docs-Sync/1.0",
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
    """校验火山方舟同步参数。"""
    if not 1 <= workers <= 16:
        raise ArkDocsError("workers 必须在 1 到 16 之间")
    if not 1 <= attempts <= 10:
        raise ArkDocsError("attempts 必须在 1 到 10 之间")
    if timeout <= 0:
        raise ArkDocsError("timeout 必须大于 0")
    if any(document_id <= 0 for document_id in document_ids):
        raise ArkDocsError("document_ids 必须全部为正整数")
    if prune and document_ids:
        raise ArkDocsError("指定 document_ids 时不能同时启用 prune")


def run(
    *,
    source_url: str,
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
    location = parse_source_url(source_url)
    output_root = project_output_path(output)
    selected_sections = set(sections)
    selected_document_ids = set(document_ids)

    with create_http_client(workers, timeout) as client:
        catalog_data = fetch_loader_data(client, location.catalog_data_url, attempts, "文档目录")
        all_plans = build_document_plans(catalog_data, selected_sections or None)
        available_sections = {plan.section for plan in all_plans}
        selected_sections = selected_sections or available_sections

        if selected_document_ids:
            plans = [plan for plan in all_plans if plan.document_id in selected_document_ids]
            missing_ids = selected_document_ids - {plan.document_id for plan in plans}
            if missing_ids:
                missing = "、".join(str(value) for value in sorted(missing_ids))
                raise ArkDocsError(f"目录中找不到已发布文档：{missing}")
        else:
            plans = all_plans

        section_counts: dict[str, int] = {}
        for plan in plans:
            section_counts[plan.section] = section_counts.get(plan.section, 0) + 1
        count_text = "，".join(f"{name} {count} 篇" for name, count in section_counts.items())
        print(f"发现 {len(plans)} 篇已发布文档：{count_text}")
        if dry_run:
            if verbose:
                for plan in plans:
                    print(f"{plan.document_id}\t{plan.relative_path.as_posix()}")
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
        selected_sections=selected_sections,
        prune=prune,
        allow_prune=not failures and not selected_document_ids,
    )
    print(
        f"同步结果：新增 {stats.created}，更新 {stats.updated}，未变化 {stats.unchanged}，"
        f"删除旧文件 {stats.removed}，保留本地修改 {stats.preserved}。"
    )
    if failures:
        print(f"以下 {len(failures)} 篇文档下载失败，本次未执行清理：", file=sys.stderr)
        for plan, error in failures:
            print(f"- {plan.document_id} {plan.title}：{error}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    """使用模块级代码常量执行文档同步。"""
    exit_code = run(
        source_url=SOURCE_URL,
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
    node_map: Mapping[str, object],
    child_ids: Sequence[object],
    section_name: str,
    parent_parts: list[str],
    ancestors: set[int],
    plans: list[DocumentPlan],
    relative_paths: set[str],
) -> None:
    """按官网顺序递归展开扁平目录树。"""
    for position, raw_node_id in enumerate(child_ids, start=1):
        if not isinstance(raw_node_id, int) or isinstance(raw_node_id, bool):
            raise ArkDocsError(f"栏目 {section_name} 包含无效节点 ID：{raw_node_id}")
        node_id = raw_node_id
        if node_id in ancestors:
            raise ArkDocsError(f"栏目 {section_name} 的目录树存在循环：{node_id}")

        node = _mapping_value(node_map, str(node_id), f"栏目 {section_name}")
        value = _mapping_value(node, "value", f"节点 {node_id}")
        document_id = _int_value(value, "DocumentID", f"节点 {node_id}")
        title = _string_value(value, "Title", f"节点 {node_id}")
        node_type = _int_value(value, "Type", f"节点 {node_id}")
        status = _int_value(value, "Status", f"节点 {node_id}")
        component = f"{position}.{safe_path_component(title)}"
        children = _sequence_value(node, "children", f"节点 {node_id}")

        if node_type == 1:
            _walk_nodes(
                node_map=node_map,
                child_ids=children,
                section_name=section_name,
                parent_parts=[*parent_parts, component],
                ancestors=ancestors | {node_id},
                plans=plans,
                relative_paths=relative_paths,
            )
            continue
        if node_type != 0:
            raise ArkDocsError(f"节点 {node_id} 使用了未知 Type={node_type}")
        if children:
            raise ArkDocsError(f"文档节点 {node_id} 意外包含子节点")
        if status != 2:
            continue

        relative_path = Path(*parent_parts, f"{component}.md")
        normalized_path = relative_path.as_posix().casefold()
        if normalized_path in relative_paths:
            raise ArkDocsError(f"多个文档映射到了同一路径：{relative_path.as_posix()}")
        relative_paths.add(normalized_path)
        plans.append(
            DocumentPlan(
                document_id=document_id,
                title=title,
                section=section_name,
                relative_path=relative_path,
            )
        )


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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ark-docs") as executor:
        for plan in plans:
            future = executor.submit(download_document, client, location, plan, attempts)
            future_map[future] = plan
        for completed, future in enumerate(as_completed(future_map), start=1):
            plan = future_map[future]
            try:
                downloaded_by_id[plan.document_id] = future.result()
                if verbose or completed % 25 == 0 or completed == len(plans):
                    print(f"下载进度：{completed}/{len(plans)}")
            except Exception as exc:
                failures.append((plan, exc))

    downloads = [downloaded_by_id[plan.document_id] for plan in plans if plan.document_id in downloaded_by_id]
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
    """识别 loader 实测会瞬时返回的状态码和服务端错误。"""
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


def _expect_mapping(value: object, context: str) -> dict[str, object]:
    """校验动态 JSON 值是字符串键映射。"""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArkDocsError(f"{context} 应为 JSON 对象")
    return cast(dict[str, object], value)


def _mapping_value(container: Mapping[str, object], key: str, context: str) -> dict[str, object]:
    """读取并校验映射字段。"""
    if key not in container:
        raise ArkDocsError(f"{context} 缺少字段 {key}")
    return _expect_mapping(container[key], f"{context}.{key}")


def _sequence_value(container: Mapping[str, object], key: str, context: str) -> list[object]:
    """读取并校验数组字段。"""
    if key not in container:
        raise ArkDocsError(f"{context} 缺少字段 {key}")
    value = container[key]
    if not isinstance(value, list):
        raise ArkDocsError(f"{context}.{key} 应为 JSON 数组")
    return cast(list[object], value)


def _string_value(container: Mapping[str, object], key: str, context: str) -> str:
    """读取并校验字符串字段。"""
    if key not in container or not isinstance(container[key], str):
        raise ArkDocsError(f"{context}.{key} 应为字符串")
    return cast(str, container[key])


def _int_value(container: Mapping[str, object], key: str, context: str) -> int:
    """读取并校验整数字段。"""
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArkDocsError(f"{context}.{key} 应为整数")
    return value


if __name__ == "__main__":
    main()
