# Client 中心 Provider 与 Public API 架构

MaiDock 有两个彼此独立的业务入口：Host 的 LLM Provider，以及面向其他插件的 Public API。两条链路只共享无状态供应商 Client 和连接池，不共享 Host Schema、参数策略或供应商执行策略。

## 总体调用链

```text
MaiBot Core RPC
  -> src/plugin.py 中的 @LLMProvider 入口
  -> LLMProviderIngress
  -> VendorRuntime.host_adapter
  -> VendorClient.session(immutable connection)
  -> 精确的 Vendor Resource / Protocol Family
  -> clients/common
  -> 上游 HTTP、multipart 或 SSE
```

跨插件图像/视频链路如下：

```text
其他插件
  -> SDK Public API
  -> PublicApiFacade
  -> PublicApiService / MediaJobEngine
  -> JobRepository / UploadRepository / ArtifactRepository
  -> PublicProviderDriver
  -> DashScopePublicDriver
  -> 共享 DashScopeClient
```

`DashScopePublicDriver` 是 `src/public_api/` 内部的供应商端口实现，不是 Host Adapter，也不存在平级的媒体 Adapter 层。`PublicApiFacade` 是 12 个公开方法的唯一边界。

## 分层与依赖方向

| 层级 | 目录 | 所有权 |
| --- | --- | --- |
| 插件入口 | `src/plugin.py` | SDK 注册、语言上下文、配置代次切换、统一错误边界 |
| Runtime | `src/runtime/` | Host Runtime、共享 Client 容器、懒加载与统一关闭 |
| Host Adapter | `src/host_adapters/` | Host Schema 校验、参数优先级、消息/工具/多模态映射、Connection 构造、Host 返回值与用户可见错误 |
| Vendor Client | `src/clients/*.py` | 供应商精确资源、端点调用和供应商原生错误 |
| Client Family | `src/clients/families/` | 只接收 wire DTO 的 JSON、SSE、multipart 协议资源复用 |
| Client Common | `src/clients/common/` | 共享连接池、租约、JSON/SSE/multipart、上传下载、超时、资源级重试和传输错误 |
| Public API | `src/public_api/` | 公开 Schema/Envelope、配置 catalog、Facade、Job Engine、Store 和供应商 Driver |
| Host Schema/Core | `src/schemas/`、`src/core/` | Host 快照、参数目录/策略、诊断、状态存储 |

依赖必须保持单向：

- `clients/common` 不得导入 Host Schema、配置、i18n、Host Adapter 或 Public API。
- `clients/families` 不得依赖具体供应商 Client 或 Adapter，只处理 wire DTO。
- Vendor Client 不读取 `ResponseRequestSnapshot` 等 Host 类型，不根据模型或 `protocol_family` 猜测端点。
- Host Adapter 可以依赖 Client 和 Host Schema；Client 不得反向依赖 Host Adapter。
- 只有 `src/runtime/ingress.py` 可以继承 `maibot_sdk.LLMProviderBase`。
- Public API 公共组件不得导入 DashScope；供应商差异只通过 config catalog 和 Driver registry 注入。
- Public Driver 不得导入 Host Schema、Host Adapter 或 Host 参数策略。

这些边界由 `tests/test_provider_architecture.py` 固定。旧 `src/providers/` 与顶层 `src/media/` 已删除，不保留兼容转发或双架构。

## Client、Connection 与 Session

插件生命周期内的 `VendorClientContainer` 按供应商懒加载一个 Client。Client 内部只有一个共享 `httpx.AsyncClient` 连接池；凭据、base URL、默认 header/query 和超时不存放在 Client，而是冻结在每次调用使用的不可变 Connection 中。

```text
共享 VendorClient
  + Connection A(api_key A, base_url A)
      -> 短生命周期 Session A
  + Connection B(api_key B, base_url B)
      -> 短生命周期 Session B
```

Host Adapter 从 Host 请求构造 Connection；Public API catalog 从 Profile 构造 Connection。二者可并发使用同一 Client 而不串用凭据。退出 Session 只释放租约，不关闭共享连接池。

关闭流程如下：

1. Client 进入 closing 状态并拒绝新 Session。
2. 等待所有已发放 Session 租约退出。
3. 只关闭一次共享连接池。

Host 配置更新只替换 Host Runtime 代次，不关闭共享 Client。Public API Profile 更新只影响新作业；在途作业继续持有旧的不可变绑定。插件卸载时 Client 拒绝新 Session，等待全部 Host/API 租约退出后关闭一次连接池。

## 运行时选项

`ProviderRuntimeOptions` 仍是配置模型到运行时的解码结果，用于保持现有配置字段、默认值和优先级。每个 Host Adapter 构造时立即将其拆为：

- `HostCommonOptions`：原始数据返回、日志、reasoning/tool 参数解析、图片限制和参数策略。
- 供应商 Host Options：端点选择、模型探测、前缀缓存、reasoning、ASR 默认值等供应商行为。
- `ConnectionOptions`：User-Agent、重试次数、强制优先级和重试间隔等 Connection 构造默认值。

API key、Host base URL、Host timeout、默认 header/query 仍来自每次请求的 Host 快照，最终冻结到供应商 Connection。配置值与 Host 请求值的原有优先级不变。

拆分之后，请求映射与结果转换链路统一接收 `RuntimeOptionsView`（`core/common.py`），这是一个只读 Protocol，列出这些函数真正会读到的九个字段。完整的 `ProviderRuntimeOptions` 与窄化后的 `HostCommonOptions` 都满足它，因此 Adapter 传窄化视图、测试直接传完整选项可以走同一批函数。

这条约束不是风格问题：`HostCommonOptions` 是 slots 数据类，把它 `cast` 成 `ProviderRuntimeOptions` 会让"读供应商专属字段"在类型检查里合法、在运行时 `AttributeError`。用 Protocol 表达之后，映射链路误读供应商字段会在 pyright 阶段就失败。协议成员一律写成只读 `property`——链路只读不写，frozen 的 `HostCommonOptions` 也才能匹配。

## 三层 JSON 类型策略

仓库里有三套结构等价、职责不同的 `JsonValue`，**按层封存，不要合并**：

| 位置 | 语义 | 服务对象 |
| --- | --- | --- |
| `clients/common/types.py` | 纯别名，不带任何辅助函数 | Client 层自包含，不依赖 core |
| `core/json_types.py` | 宽松窄化，转不动就降级（`normalize_json_value` 把非 JSON 对象转成 `str`） | Host 侧尽力转译，不能让一次映射失败 |
| `public_api/domain/json_types.py` | 严格校验，转不动直接抛异常 | Public Driver 合约，宁可失败也不能静默变形 |

三者结构等价，pyright 判定互相兼容，因此跨层直通不需要 `cast`，也不需要在边界重新构造容器（见 `host_adapters/common/client_bridge.py`）。合并成一套会强迫某一层接受不属于它的失败语义：Client 层会被迫依赖 core，Host 层会因为一个畸形字段整单失败，或者 Driver 层会静默吞掉本该报错的形状。

## 六家资源归属

| Client | 精确资源 |
| --- | --- |
| OpenAI | Responses、Embeddings、Audio Transcriptions |
| Anthropic | Messages |
| Volcengine ARK | Responses、Multimodal Embeddings、ASR、Tokenization |
| DashScope | Text Generation、Multimodal Generation、Embeddings、Audio Transcriptions，以及 Public API 使用的五个生成资源、tasks、uploads、artifacts |
| SiliconFlow | Chat Completions、Embeddings、Audio Transcriptions |
| Xiaomi Mimo | Chat Completions、专用 ASR |

Client 只暴露上述资源，不提供 `generate(raw_json)` 一类万能入口。协议 body 和 SSE 读取属于 Client/Client Family；Host 消息、工具、多模态和结果合约属于 Host Adapter/Host Family。

DashScope 未知模型的文本/多模态端点探测与缓存只位于 `DashScopeHostAdapter`。Public API 未知模型必须由请求或 Profile 显式锁定协议族，不做端点探测或 fallback。

ARK 前缀缓存协调器位于 Host Adapter，实际的 Responses/Tokenization 请求由 `ArkClient` 执行。Mimo reasoning 恢复和持久化位于 Host Adapter，因为它依赖 Host 历史与工具元数据。

## 重试与错误

重试由具体资源显式传入 `RetryPolicy`，Client Common 只执行策略：

- 普通 LLM 资源沿用现有 Host/配置重试优先级。
- SSE 只允许在首个事件发出前重试；已经产生输出后中断不得重发整次请求。
- 图像/视频生成提交不重试。提交超时或中断且没有 remote handle 时标记为 `uncertain`。
- 查询、取消、上传策略和下载等安全操作可以声明有限重试。
- `x-should-retry` 的明确上游指示优先于默认状态码判断。

Client 抛传输错误或供应商原生错误；Host Adapter 负责映射 Host 用户可见错误。Public Driver 将供应商错误归一化为稳定 code，Facade 再按当前语言生成 `{ok, data, error}`；持久层不保存本地化 message。

## Public API 内部边界

`src/public_api/domain/` 定义供应商无关的不可变 `MediaRequest`、`PreparedMediaOperation`、`VersionedOpaqueHandle`、`PublicProviderDriver`，以及 `Completed/Accepted/Running/Failed/Canceled` outcome。`api/` 使用严格 Pydantic Command 和判别式 Envelope，`application/` 只编排领域对象，SQLite 行与 opaque JSON 则先经过 `storage/records.py` 的 Pydantic record 校验。prepared payload 与 remote handle 都带版本且不包含 API Key；Job Engine 只持久化 driver key、payload version、Profile、凭据指纹和 opaque JSON，不解析供应商字段。

`PublicDriverRegistry` 只按 Profile 和 driver key 查找执行器，不包含供应商分支。类型化 `PublicProviderContribution` 负责把供应商配置、WebUI 元数据、Profile 解析器和 Driver 工厂接入运行时。增加供应商时只扩展 Client 精确资源、配置模型、contribution 和 Driver，不修改 Facade、Job Engine 或 Store。

`DashScopePublicDriver` 持有精确模型 registry，负责 capability、mode、input role、参数范围、Profile defaults、请求 parameters、Profile overrides 和协议族锁定。已实现五个精确资源：

1. Multimodal Generation
2. Image Generation
3. Text2Image Synthesis
4. Image2Image Synthesis
5. Video Generation

辅助资源包括 tasks 查询/取消、OSS 上传策略与上传、artifact 下载。下载只接受 HTTPS，按流限制大小并计算 SHA-256；Store 负责 staging 和原子落盘。

Wan2.6 交错输出固定使用 Multimodal Generation 和 `parameters.stream=true`，保留图文顺序并合并 usage。Qwen 图片模型的 `n` 限制由 registry 校验。显式或 registry 选定的协议族失败后不得自动切换到另一协议族。

## 状态所有权

| 状态 | 所有者 | 持久化 |
| --- | --- | --- |
| HTTP 连接池与 Session 租约 | Vendor Client | 否 |
| 共享 HTTP Client 与 Session 租约 | Vendor Client Container | 否 |
| Host 配置代次与六家 Runtime | Runtime Container | 否 |
| DashScope 未知模型端点缓存 | DashScope Host Adapter | 否 |
| ARK 前缀缓存 | ARK Host Adapter / `PluginStateStore` | 是，沿用现有 namespace |
| Mimo reasoning 连续性 | Mimo Host Adapter / `PluginStateStore` | 是，沿用现有 namespace |
| Public API Profile 与凭据快照 | Public API catalog / Driver | 仅内存，数据库只存指纹 |
| 作业、幂等、上传与 artifact 索引 | `maidock_public_api.sqlite3` | WAL SQLite |
| 上传与 artifact 内容 | `public_api/uploads`、`public_api/artifacts` | 文件系统原子落盘 |

恢复时，同名 Profile 的供应商、driver key 和凭据指纹必须一致，否则作业以 `PROFILE_CHANGED` 终止。数据库永不保存 API Key。提交阶段中断且没有 remote handle 时以 `EXECUTION_UNCERTAIN` 终止，不重复提交。

## 动态 API 生命周期

12 个 API 由 SDK 2.7 动态注册，固定 `public=True`、`version="1"`、`timeout_ms=25000`。插件内部注册名为 `media.*`；SDK 以插件 ID 加前缀后，其他插件使用 `chinokou.maidock.media.*` 调用。`public_api.enabled=false` 时不注册；从启用切换到关闭时立即停止接单并注销，但进程内在途作业继续收尾。卸载会停止 worker、持久化当前原子操作结果，再等待共享 Client 关闭。

Public API 不设计调用鉴权。DashScope `api_key` 只用于访问上游，并在 WebUI 以普通字符串字段明文编辑；日志、repr 和异常诊断必须脱敏。

完整接口、限制与调用示例见 [Public API 参考](../media_api_reference.md)。

## 验证

```powershell
uv run --locked pytest -q tests
uv run --locked ruff format --check plugin.py src scripts tests
uv run --locked ruff check plugin.py src scripts tests
uv run --locked pyright
git diff --check
```
