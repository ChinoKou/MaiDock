# Repository Guidelines

## 仓库范围与基本原则

MaiDock 是 MaiBot 的独立 LLM Provider 插件，为 OpenAI、Anthropic、Volcengine ARK、DashScope、百炼 Responses、SiliconFlow 和 Xiaomi Mimo 提供原生 `httpx` 后端，运行时不依赖供应商 SDK。本指南作用于整个 MaiDock 仓库。

- 只修改本仓库。父级 MaiBot Core 及其 Git 历史只读；需要 Core 改动时先说明原因并请求许可。
- 保留用户现有工作树，不使用 `git checkout --`、`git reset --hard` 等命令覆盖未知修改。
- 文件内容只能通过 Edit、Write 或 `apply_patch` 修改。禁止使用重定向、`Set-Content`、`Out-File`、`WriteAllText` 或临时脚本覆写源码。
- 未经明确要求，不执行 `git add`、commit、push、merge，不修改版本号或 `_manifest.json`。禁止手工编辑 `uv.lock`；依赖解析与锁文件更新必须通过 `uv add`、`uv remove`、`uv lock`、`uv sync` 等 UV CLI 完成。manifest 始终需要用户明确确认。
- `.claude/`、`.omc/`、`config_back/`、`downloads/`、`docs/provider_docs/` 和本地缓存不得进入提交。

## Project Structure & Module Organization

- `plugin.py`：Core 动态加载入口，仅导出 `create_plugin`。
- `src/plugin.py`：`MaiDockPlugin`、七个 `@LLMProvider` 注册入口、12 个动态 Public API 注册与插件生命周期。
- `src/config.py`、`src/config_schema.py`：运行配置模型、规范化及 WebUI Schema。
- `src/clients/`：**供应商 Client 层**。`common/` 是协议无关的 HTTP 原语与 `RetryPolicy`；`families/` 是 JSON/multipart 资源模板；`<vendor>.py` 是各家的 Connection / Session / Client。
- `src/host_adapters/`：**Host 通路**。`common/` 是 RPC 边界、参数管线与 httpx 配置；`*_family/` 是协议级复用层；`*_provider/` 是各家的端点、鉴权与特殊行为。
- `src/public_api/`：**跨插件通路**。`api/` 门面、`application/` 服务与作业引擎、`domain/` 端口与模型、`storage/` 持久化、`providers/<vendor>/` 四件套。
- `src/runtime/`：Client 容器、Host Runtime 容器、工厂与唯一接触 SDK Provider 基类的 ingress。
- `src/core/`：运行参数、JSON 类型、参数目录/策略、解析、日志脱敏和 SQLite 状态存储。
- `src/schemas/`：Host 请求快照、Provider 响应合约及协议 Pydantic 模型。
- `docs/`、`README.md`：配置、参数覆写、模型示例、能力矩阵、架构说明和开发文档；`docs/provider_docs/` 是按需生成且被 Git 忽略的本地供应商资料。
- `scripts/`：配置生成和供应商文档维护工具；`tests/` 是纳入版本管理的正式测试套件。

两条入口链路：

```text
Host  : plugin.py -> src/plugin.py -> runtime.ingress -> host_adapters/<vendor> -> clients/<vendor>
Public: plugin.py -> public_api/api -> application/job_engine -> providers/<vendor>/driver -> clients/<vendor>
```

Runtime 首次使用时惰性创建，配置热更新只使 Runtime 实例失效。插件级 `PluginStateStore` 在 `on_load` 获得标准数据目录，固定使用 `ctx.paths.data_dir / "maidock_state.sqlite3"`，首次读写时打开，并仅在 `on_unload` 关闭。Public API 另有独立的 `maidock_public_api.sqlite3`。

## Provider Architecture

架构是 **Client 中心**的四层，而不是单一的 Provider 树：

```text
                    clients/<vendor>          <- 唯一共享点
                   /                 \
   host_adapters/<vendor>        public_api/providers/<vendor>
            |                              |
      runtime/ingress                 public_api/api
```

**两条上层通路彼此独立，只共享供应商 Client。** 这是重写换来的核心教训：第一次尝试让 Host Adapter 兼任公共 API 的基础设施，结果两边的需求互相拉扯，最终不得不推倒。**不得让一个入口的 Adapter 成为另一个入口的基础设施**——`host_adapters/**` 不得导入 `public_api`，`public_api/**` 不得导入 `host_adapters` 或 `schemas`。这两条由 `tests/test_provider_architecture.py` 强制。

Host 通路内部的依赖方向：

```text
具体 Provider -> 对应协议 Family -> host_adapters/common -> core / schemas / clients
```

`*_family/**` 不得依赖任何 `*_provider`；`host_adapters/common/**` 不得反向依赖 Family 或具体 Provider。DashScope 和 Anthropic 是独立协议实现，可以直接使用 `common`。

Client 层必须自包含：`clients/**` 不得导入 `core`、`runtime`、`schemas` 或任一上层；`clients/families/**` 不得导入任何 `clients/<vendor>`。

| Provider | 文本生成 | ASR | Embedding |
| --- | --- | --- | --- |
| OpenAI | `responses_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| 百炼 Responses | `responses_family` | 不支持 | 不支持 |
| ARK | `responses_family` | `responses_family` | `responses_family` |
| SiliconFlow | `chat_completions_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| Mimo | `chat_completions_family` | `chat_completions_family` | 不支持 |
| DashScope | 独立实现 | 独立实现 | 独立实现 |
| Anthropic | 独立实现 | 不支持 | 不支持 |

Provider 目录通常包含 `adapter.py`、能力模块、`streaming.py`、`multimodal.py`、`tools.py` 和 `parameter_translation.py`。这些门面必须进入真实 Mapper 调用链，不能只保留未使用的转发文件。详细边界见 `docs/development/provider_architecture.md`。

### Provider API 与目录模板

| Provider | 核心 API | 流式协议 |
| --- | --- | --- |
| `openai_responses_provider` | `POST /v1/responses`、`/v1/embeddings`、`/v1/audio/transcriptions` | Responses SSE |
| `bailian_responses_provider` | `POST /api/v2/apps/protocols/compatible-mode/v1/responses` | Responses SSE |
| `anthropic_messages_provider` | `POST /v1/messages` | Anthropic Messages SSE |
| `volcengine_ark_provider` | `POST /api/v3/responses`、`/api/v3/embeddings/multimodal` | Responses SSE |
| `dashscope_provider` | `POST /api/v1/services/aigc/text-generation/generation`、`/api/v1/services/aigc/multimodal-generation/generation`（ASR）、按模型分流的 Embedding | DashScope SSE |
| `siliconflow_provider` | `POST /v1/chat/completions`、`/v1/embeddings`、`/v1/audio/transcriptions` | `choices[0].delta` |
| `xiaomi_mimo_provider` | `POST /v1/chat/completions`，包含专用 ASR | `choices[0].delta` |

```text
host_adapters/<name>_provider/
  adapter.py               # Host RPC 三入口、Connection 构造与能力分发
  responses.py / chat.py   # 请求构建与响应解析
  embeddings.py            # Embedding 能力
  audio_transcriptions.py  # ASR 能力
  streaming.py             # SSE 收集
  multimodal.py            # 图片或音频内容映射
  tools.py                 # 工具定义、调用和结果转译
  parameter_translation.py # Host 参数到协议参数
```

共享层的职责必须清晰：

- `responses_family/`：OpenAI Responses 与 ARK 的请求、响应、流式、工具和多模态映射。
- `chat_completions_family/`：SiliconFlow 与 Mimo 的 Chat Completions 映射。
- `openai_auxiliary_family/`：OpenAI 与 SiliconFlow 的 multipart ASR 和 OpenAI 风格 Embedding。
- `host_adapters/common/`：RPC 边界别名、参数管线、httpx 配置、Client 桥接、音频签名、图片处理和协议无关解析。

### Client 层

Client 只负责"把一次调用发出去、把响应解回来"，不认识 Host 快照，也不认识 Public API 的领域模型。

```text
clients/
  common/http.py     # 六个原语：request_json / request_optional_json /
                     # request_multipart / stream_sse_json / upload_file / download
  common/types.py    # 自包含的 JsonValue 别名（不依赖 core）
  families/          # JsonResource / MultipartResource 等资源模板
  <vendor>.py        # Connection（不可变快照）+ Session（资源）+ Client（连接池）
```

Connection 是一次调用期间不可变的快照；Session 从共享 `SharedHttpClient` 借出短生命周期租约；Client 在插件生命周期内单例并被两条上层通路共用。错误由各家的 `JsonErrorFactory` 转成 `ClientHttpError` 子类——注意工厂在 2xx 上也会执行，这样才能捕获"HTTP 200 但响应体里带 error"这类整单失败。

### Public API 通路

```text
public_api/
  api/            # 12 个动态 RPC 门面，仅做 envelope 与参数校验
  application/    # 服务、job_engine 状态机、媒体元数据探测
  domain/         # PublicProviderDriver 端口、五结局模型、严格 JsonValue
  storage/        # 独立 SQLite、上传、产物、配额
  providers/common/     # 供应商无关的约束原语
  providers/<vendor>/   # 四件套：wire / registry / driver / contribution
```

Driver 的五个操作返回统一的五结局模型 `Completed / Accepted / Running / Failed / Canceled`。`providers/<vendor>/` 之间互相不可见，任何共享必须下沉到 `providers/common/`。新增一家供应商时按四件套补齐，并在 `providers/__init__.py` 的两个 tuple 中注册——`scripts/generate_config.py` 与 WebUI schema 都按这份目录动态展开，不要写死供应商名。

### Core 与 Schema 职责

| 模块 | 用途 |
| --- | --- |
| `core/json_types.py` | `JsonValue`、JSON 窄化、取值、规范化与 `json_array` |
| `core/parameter_catalog.py` | Core typed fields 及 MaiDock 覆写字段的目标路径、类型、默认值与文档元数据 |
| `core/parameter_policy.py` | Provider/能力粒度的覆写解析与叶级合并规则 |
| `core/common.py` | `ProviderRuntimeOptions`、只读 `RuntimeOptionsView`、URL、图片、usage、客户端通用配置 |
| `core/parsing.py` | 工具参数、reasoning 和 XML fallback 解析 |
| `core/diagnostics.py` | API Key、请求和响应日志脱敏 |
| `core/state_store.py` | 插件级 SQLite KV 与 TTL 生命周期 |

`schemas/host_snapshots.py` 定义 Host → Plugin 合约；`provider_contracts.py` 定义 Plugin → Host 响应；`responses_compat.py` 与 `anthropic_messages.py` 定义协议边界；`base.py` 提供模型基类。协议 JSON 必须先经过 Schema 或现有窄化函数，不能靠随意链式 `get` 掩盖结构错误。

### 类型硬约束

`core`、`clients`、`host_adapters`、`public_api` 四层禁止四类转义舱口，零豁免，由
`test_layer_has_no_untyped_escape_hatches` 强制：

1. `Any`
2. `cast(...)`
3. `dict[..., Any]` / `dict[..., object]`
4. 注解位置上的裸容器（单独出现的 `dict` / `list` / `Mapping` / `Sequence` 等）

需要窄化时用 `core/json_types.py` 里的运行时校验函数，而不是断言。全仓源码同时禁止
`type: ignore` 与 `pyright: ignore`。

**三套 JsonValue 按层封存，不要合并**：`clients/common/types.py` 是纯别名（保证 Client
自包含）、`core/json_types.py` 宽松窄化（服务 Host 尽力转译）、`public_api/domain/json_types.py`
严格抛异常（服务 Driver 合约）。三者结构等价，pyright 判定互相兼容，跨层直通不需要转换。
合并会强迫某一层接受不属于它的失败语义。

一个反复出现的坑：`list` 在类型系统里是不变的，`list[dict[str, JsonValue]]` **不是**
`list[JsonValue]`。把元素类型更精确的列表放进 JSON 槽位时用 `json_array(...)` 显式收拢。

## Parameter and Endpoint Rules

参数流按以下顺序执行：

1. Host 请求快照只提取 Core 明确定义的 typed fields；请求级和模型级 `extra_params` 任何内容与类型都直接忽略。
2. Provider 翻译器把 typed fields 写入 `TranslationEnvelope` 的 body、headers 和 query；请求级值为 `None` 时由 Core 自行回落模型级或默认值。
3. `[{provider}.{capability}.overrides]` 中的非空字符串按目录类型解析，以最高优先级对 body/headers/query 执行叶级合并。
4. Host Function tools 在前，覆写目录中的原生工具在后，保持顺序追加且不去重。百炼 Responses 本轮不开放服务器托管工具覆写。

覆写目录是唯一的额外参数入口；未知覆写键必须在配置归一化阶段直接报错，不得静默跳过。新增字段必须同时更新参数目录、Provider 翻译器、配置 Schema、模板、参考文档与请求测试。

`normalize_base_url()` 不提供静默回退：空值或非法 URL 必须报错，缺失 scheme 时使用 HTTPS。OpenAI、Anthropic 使用 Host URL；ARK、DashScope、SiliconFlow 默认强制官方端点，关闭 `force_official_endpoint` 后才使用 Host URL。不要用宽泛 fallback、吞异常或伪造成功响应掩盖协议错误。

## Build, Test, and Development Commands

从 MaiDock 根目录运行，优先使用 UV，且不要显式指定外部缓存或临时目录：

```bash
uv sync                                                                  # 安装 dev 与 sdk 默认依赖组
uv run --locked ruff check --select I --fix scripts tests                # 开发资产导入整理
uv run --locked ruff format plugin.py src scripts tests                  # Ruff 格式化
uv run --locked ruff format --check plugin.py src scripts tests          # 格式检查
uv run --locked ruff check plugin.py src scripts tests                   # Lint
uv run --locked pyright                                                  # Pyright standard
uv run --locked pytest -q                                                # 全量测试与分支覆盖
uv run --locked pytest tests/test_config_runtime.py -k test_default_config
uv run scripts/generate_config.py                                        # 从 Schema 重建模板
```

修改 Python 后，上述格式、Lint、类型和测试四道质量门必须全部通过。不要用 `# type: ignore`、无意义的 `Any` 或放宽规则绕过错误。格式化只交给 Ruff；不要手工整理导入顺序。

## Coding Style & Naming Conventions

- Python 3.12+，四空格缩进；函数、变量和模块用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`。
- 导入边界、相对导入规则以及测试和独立脚本的例外统一见 `docs/development/imports.md`。
- 保留既有类型注解。复杂函数必须补齐参数和返回类型；Pydantic 用于 IO 边界，内部状态优先使用 `@dataclass(slots=True)`。
- JSON 使用 `JsonValue`、`Mapping[str, JsonValue]` 和现有窄化函数。`object` 只用于真实边界，窄化后立即转换。
- Docstring、行内注释、日志和错误消息以简体中文为主。单句 docstring 写成 `"""说明。"""`；复杂逻辑才添加简短注释。
- 已确认属性存在时直接访问，避免 `getattr` / `setattr`；不要增加无依据的 `or` fallback。
- 保留文件原有换行风格。Windows 工作树通常使用 CRLF，Git 提交时归一化为 LF；功能提交不得夹带纯换行变更，统一换行应单独提交。

完整递归 JSON 类型为：

```python
type JsonValue = (
    str | int | float | bool | None
    | dict[str, "JsonValue"]
    | list["JsonValue"]
)
```

容器泛型优先使用完整参数化类型。结构过于复杂且具有稳定字段时，先考虑用 Pydantic 模型表达和校验；结构动态或不适合 Pydantic、继续参数化也没有真实信息增益时，才允许使用裸 `dict`、`list` 或 `Mapping`，并在读取边界尽快窄化。不要为了兼容旧 Python 降低注解。多行 docstring 的开闭三引号各自独占一行，并保留概述与必要细节；重构时原有注释和类型若仍准确必须保留。

## i18n 扩展规范

- 固定支持 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`，默认 `zh-CN`；新增语言或修改 manifest 仍需维护者明确许可。
- 所有语言使用稳定消息键和命名占位符；五份 JSON 必须键集合、占位符集合完全一致，且不得有重复键或空文本。目录错误必须阻止加载，禁止静默回退。
- WebUI 可见 Schema、MaiDock 日志及向 Host/RPC 返回的错误必须本地化。Provider/模型名、字段路径、协议枚举、HTTP/SSE/JSON/Base64 和错误码等技术值保持原样。
- `ProviderRuntimeOptions.locale` 是请求语言来源，只在插件 Provider 调用边界绑定 `ContextVar`；执行中请求保持原语言，热更新后的新请求使用新语言，内部 helper 不得覆盖上下文。
- 不翻译发送给模型的默认提示词、图片占位内容和用户参数。上游详情必须先经 `core/diagnostics.py` 脱敏；Pydantic 错误只暴露字段路径、稳定错误码和本地化说明。
- i18n 改动必须覆盖目录失败分支、五语 Schema、日志/异常、异步隔离及热更新测试，并通过完整 Ruff、Pyright 和 Pytest。

## 关键实现模式

- 所有 HTTP 调用使用 `httpx.AsyncClient` 和 Family/common 传输入口。`sdk` 依赖组仅供本地检索供应商实现，生产代码不得导入其中的 SDK。
- 图片管线固定为 Base64 校验、解码、Pillow 几何/帧数检查、格式转换和重新编码。
- 音频必须严格 Base64 解码，通过文件签名或显式格式确定 MIME；格式冲突立即报错。
- 工具调用支持协议原生结构，并在既有实现允许时从 XML 文本恢复；参数统一经过 `normalize_arguments`。
- reasoning 优先读取协议原生字段，既有 `<think>` fallback 只用于明确支持的路径，不得扩大成宽泛兜底。
- Responses 的流式与非流式解析必须保持 usage、工具调用、reasoning 和 `incomplete:length` 截断内容。
- ARK 前缀缓存和 Mimo reasoning 共用插件级 Store，但使用独立 namespace；凭据只参与不可逆摘要。

## Testing Guidelines

使用 Pytest 和 `pytest-asyncio`。文件命名为 `test_<area>.py`，测试命名为 `test_<behavior>`。新增或修改协议行为时，应覆盖请求 body、headers、query、端点、流式终态、工具、多模态、usage、错误语义和参数冲突。

所有上游调用使用 `httpx.MockTransport` 或等价本地替身；禁止执行收费的真实 Provider 请求。架构变更应保留 AST 依赖边界测试及 Mapper 门面调用测试。Bug 修复必须包含能复现旧行为的回归测试。项目当前不设机械覆盖率阈值，但受影响分支必须有针对性验收。

`tests/` 与测试配置属于仓库正式资产，应随行为变更同步提交。Pytest 已固定只收集 `tests/`，使用严格配置、严格 marker 和严格 asyncio；测试的一次性文件使用内置 `tmp_path`，由系统临时目录管理。`downloads/test/` 只用于人工测试下载。

## Configuration, Documentation, and Security

配置改动应同步 `src/config.py`、`src/config_schema.py`、`config.toml`、`README.md` 和相关 `docs/`。优先从 Schema 运行生成脚本，不手工制造模板空行差异。除非明确要求，不新增迁移 Hook，也不顺手修改无关兼容逻辑。

不得提交 API Key、认证头、完整用户提示词、真实 reasoning、SQLite 数据库或供应商响应样本。日志必须经过 `src/core/diagnostics.py` 脱敏。持久化文件只能位于 Core 授予的 `ctx.paths.data_dir`；namespace 之间必须隔离。若功能会在本地明文保存 reasoning，应在 README 和配置参考中明确范围、保留期与风险。

文档职责：

- `README.md`：能力矩阵、端点、主要配置和用户可见行为。
- `docs/plugin_config_reference.md`：完整 `config.toml`、策略字段和 WebUI Schema 语义。
- `docs/extra_params_reference.md`：各 Provider 支持的参数覆写与 body/headers/query 目标。
- `docs/model_config_examples.md`：`model_config.toml` 的供应商示例。
- `docs/development/provider_architecture.md`：分层边界、能力归属和新增 Provider 接入规则。
- `docs/development/imports.md`：生产代码、测试和独立脚本的 Python 导入规范。
- `docs/development/scripts.md`：开发依赖、配置生成器和供应商文档同步器的维护方式。
- `docs/provider_docs/`：通过开发脚本按需下载的本地供应商资料，不进入版本管理。

需要供应商协议资料辅助开发时，按 `docs/development/scripts.md` 执行对应同步器，在本地生成 `docs/provider_docs/`。同步器的范围、网络参数和安全选项使用脚本顶部的模块常量配置。`docs/provider_docs/` 和 `downloads/` 均不提交。

## Commit & Pull Request Guidelines

历史使用 Conventional Commits。格式为 `type(scope): imperative summary`，例如：

```text
feat(provider): add auxiliary protocol family
fix(volcengine): rebuild Responses prefix cache
refactor(xiaomi_mimo_provider): simplify parameter translation
chore: update ignored development artifacts
```

一个提交聚焦一个可审阅主题；版本提升使用独立 `bump:` 提交，并仅在维护者授权后执行。不要把 `.claude/`、`.omc/`、`downloads/`、`docs/provider_docs/`、缓存或无关格式变化混入功能提交；`tests/` 和 `scripts/` 是可提交的开发资产。

PR 描述应说明问题、协议依据、实现边界、兼容性影响和验证命令；关联 Issue 或原 PR。涉及请求协议时给出脱敏 payload 结构或本地文档路径；涉及 WebUI 配置时附截图。提交前确认 Ruff、Pyright、Pytest 全部通过，并明确记录任何无法运行的检查或已知风险。
