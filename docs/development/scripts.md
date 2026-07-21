# 开发脚本使用说明

在 MaiDock 根目录执行以下命令，即可将对应供应商的最新文档下载到 `docs/provider_docs/`，供协议实现和问题排查时在本地辅助参考：

```powershell
uv run scripts/update_bailian_docs.py
uv run scripts/update_siliconflow_docs.py
uv run scripts/update_volcengine_ark_docs.py
uv run scripts/update_xiaomi_mimo_docs.py
```

| 脚本 | 默认输出目录 |
| --- | --- |
| `update_bailian_docs.py` | `docs/provider_docs/dashscope/` |
| `update_siliconflow_docs.py` | `docs/provider_docs/siliconflow/` |
| `update_volcengine_ark_docs.py` | `docs/provider_docs/volcengine_ark/` |
| `update_xiaomi_mimo_docs.py` | `docs/provider_docs/xiaomi_mimo/` |

`docs/provider_docs/` 已被 Git 忽略，不应提交。默认配置会同步该供应商目录中的全部文档；同步器只会在正文或清单元数据发生变化时更新时间戳，因此重复执行不会反复改写未变化的清单。

## 调整同步范围

需要调整范围或网络参数时，编辑目标脚本顶部的模块常量：

| 常量 | 用途 |
| --- | --- |
| `SOURCE_URL` | 供应商文档入口 |
| `OUTPUT_DIRECTORY` | 输出目录，必须位于 MaiDock 仓库内 |
| `WORKERS` | 并发下载数 |
| `ATTEMPTS` | 单次请求的最大尝试次数 |
| `TIMEOUT_SECONDS` | 请求超时秒数 |
| `PRUNE` | 是否清理清单中已失效的旧文档 |
| `DRY_RUN` | 是否只检查目录和选择结果 |
| `VERBOSE` | 是否打印每篇文档的远端标识和本地路径 |

选择部分资料时使用各脚本对应的选择器：

| 脚本 | 栏目选择器 | 单篇文档选择器 |
| --- | --- | --- |
| 百炼 | `SECTIONS` | `DOCUMENT_IDS` |
| SiliconFlow | `SECTIONS` | `DOCUMENT_PATHS` |
| 火山方舟 | `SECTIONS` | `DOCUMENT_IDS` |
| Xiaomi MiMo | 无 | `DOCUMENTS` |

选择器为空元组时同步全部文档。单篇文档选择器与 `PRUNE = True` 不能同时使用，参数不合法时脚本会在发起请求前报错。

## Dry-run 与清理

将 `DRY_RUN` 改为 `True` 后，脚本只读取远端目录并输出选中的文档，不下载正文、不创建目录，也不修改现有清单。

将 `PRUNE` 改为 `True` 后，脚本会清理远端目录中已经不存在的旧文档。删除前会核对清单中的 SHA-256；本地内容被修改过的文件会保留。只要有文档下载失败，本次执行就不会清理旧文件。

执行清理前应先检查 Git 工作树，并通过 dry-run 确认选择范围。

## 生成配置模板

修改配置 Schema 后，在 MaiDock 根目录执行：

```powershell
uv run scripts/generate_config.py
```

脚本会根据当前 Schema 重新生成 `config.toml`。提交前检查模板 diff，避免手工调整生成器负责的默认值和排版。

## 开发依赖与质量检查

首次使用时安装开发依赖：

```powershell
uv sync
```

该命令会安装 `dev` 和 `sdk` 两个默认依赖组。`beautifulsoup4` 和 `markdownify` 用于文档同步；`sdk` 组包含 `anthropic`、`openai`、`dashscope` 和 `volcengine-python-sdk`，用于在本地环境中检索和阅读供应商 SDK 源码。这些包都不属于插件运行依赖，生产代码不得导入供应商 SDK。

修改脚本或测试后运行：

```powershell
uv run ruff format --check plugin.py src scripts tests
uv run ruff check --select I scripts tests
uv run ruff check plugin.py src scripts tests
uv run pyright
uv run pytest -q
git diff --check
```

`downloads/test/` 仅用于人工测试下载，不是 Pytest 临时目录，也不应提交。Pytest 的一次性文件统一使用内置 `tmp_path`，由系统临时目录管理。
