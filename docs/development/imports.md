# Python 导入规范

本文集中说明 MaiDock 生产代码、测试和开发脚本的 Python 导入边界。

## 生产代码

MaiBot Core 会使用动态包名加载插件，并且运行环境已经存在顶层 `src` 包。为了避免 MaiDock 错误导入 Core 的 `src`，生产代码必须使用包相对导入。

根入口 `plugin.py` 固定为：

```python
from .src.plugin import create_plugin

__all__ = ["create_plugin"]
```

`src/` 内按包层级使用相对导入：

```python
# 同一目录
from .config import MaiDockConfig

# 父目录或相邻包
from ...schemas import ResponseRequestSnapshot
from ..responses_family.responses import ResponsesMapper
```

生产代码禁止使用顶层导入：

```python
# 错误：动态加载时可能命中 Core 的 src 包
from src.schemas import ResponseRequestSnapshot
```

## Client 与 Host Adapter 依赖边界

导入路径必须体现架构依赖方向：

- `clients/common` 不得导入 `schemas`、`i18n`、`host_adapters`、`public_api` 或插件配置。
- `clients/families` 只接收 wire DTO，不得导入具体供应商 Client 或 Host Adapter。
- 具体供应商 Client 可以导入 `clients/common` 和 `clients/families`，不得导入 Host Schema。
- Host Adapter 可以依赖 Client、Host Schema、Core 和 i18n；Client 不得反向依赖 Host Adapter。
- `public_api` 的 Facade、Application、Store、Domain 和公共 API schema 不得导入具体供应商。
- 具体 Public Driver 可以依赖供应商 Client，但不得导入 Host Schema、Host Adapter 或 Host 参数策略。
- 供应商配置和 Driver 通过 Public API config catalog/registry 加入调用链，Facade、Job Engine 与 Store 不增加供应商分支。
- 只有 `src/runtime/ingress.py` 可以导入并继承 `LLMProviderBase`。

例如，Host Adapter 可以导入精确供应商 Client：

```python
from ...clients.mimo import MimoClient, MimoConnection
```

Client 不得反向读取 Host 请求快照：

```python
from ...schemas import ResponseRequestSnapshot
```

这些边界由 `tests/test_provider_architecture.py` 的 AST 测试持续检查。

## 测试代码

测试从 MaiDock 仓库根目录运行，Pytest 和 Pyright 已把仓库根目录配置为搜索路径。因此测试可以使用顶层 `src` 导入：

```python
from src.core.common import ProviderRuntimeOptions
from src.host_adapters.openai_responses_provider.adapter import OpenAIHostAdapter
```

`tests/` 是显式包。`tests/support/` 中跨多个测试文件使用的 helper 优先采用包绝对导入：

```python
from tests.support.assertions import as_json_object
from tests.support.http import TrackingByteStream
```

显式包相对导入同样有效，现有测试在只取少量局部 helper 时也会使用：

```python
from .support.http import make_api_provider
```

不要写成 `from support...`；这种顶层导入依赖偶然的搜索路径，无法表达 helper 属于 `tests` 包。

与测试文件同属 `tests/` 根包的相邻 helper 使用相对导入：

```python
from .plugin_test_support import bind_plugin_context
```

不要让生产代码反向导入 `tests`，也不要为了测试调整生产包的导入方式。

## 开发脚本

`scripts/` 按可直接执行的独立脚本组织，没有 `scripts/__init__.py`。从 MaiDock 根目录运行脚本时，脚本所在目录会进入模块搜索路径，因此同目录共享模块使用普通导入：

```python
from docs_sync_common import atomic_write, project_output_path
```

不要在独立脚本中使用包相对导入：

```python
# 错误：直接执行时 scripts 不是包
from .docs_sync_common import atomic_write_text
```

需要导入 `src` 的脚本只允许做必要的项目根路径初始化。当前 `generate_config.py` 使用这一方式生成配置；供应商文档同步器不应无故依赖生产包。

脚本统一从 MaiDock 根目录直接执行：

```powershell
uv run scripts/generate_config.py
uv run scripts/update_bailian_docs.py
```

## 导入分组与排序

生产包通常按以下块分组，块之间空一行：

1. 标准库。
2. 第三方库。
3. 本地模块。

项目默认 Ruff lint 启用 `E`、`F`、`B` 和 `UP`，没有全局启用 `I`。`scripts/` 与 `tests/` 通过单独的 `--select I` 命令整理导入；生产代码保留现有导入布局，不在无关改动中做全仓排序。

独立脚本不是 Python 包，Ruff 会把 `docs_sync_common` 视为普通绝对导入，并与第三方绝对导入放在同一块。这是直接执行脚本所需的预期布局，不要为了制造“本地模块块”改成不可执行的相对导入。

检查命令：

```powershell
uv run --locked ruff check --select I scripts tests
uv run --locked ruff check plugin.py src scripts tests
uv run --locked pytest -q tests/test_provider_architecture.py
```
