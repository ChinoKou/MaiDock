# Provider 架构

本文说明 MaiDock 当前 Provider 实现的分层、能力归属和扩展边界。

## 总体调用链

MaiDock 由 Core 动态加载根入口 `plugin.py`。根入口只导出 `create_plugin`，实际注册、生命周期和分发逻辑位于 `src/plugin.py`。

```text
MaiBot Core RPC
  -> plugin.py
  -> MaiDockPlugin 中的六个 @LLMProvider 入口
  -> LLMProviderBase.dispatch(operation, request)
  -> 具体 Provider
  -> Host 请求快照 Schema
  -> Provider 适配层 / 协议 Family
  -> 参数目录与参数策略
  -> Family transport / common HTTP
  -> 上游协议响应或 SSE
  -> 协议 Schema / 流式收集器
  -> ProviderResponse
  -> Core 可接收的 JSON object
```

插件入口只接受以下三种 operation：

- `response`：文本、多模态、工具和 reasoning 请求。
- `embedding`：文本或多模态向量请求。
- `audio_transcription`：音频转录请求。

不支持的 operation 在插件边界拒绝；具体 Provider 不支持的能力由 `NotImplementedError` 转换为本地化错误。Pydantic 校验错误也在该边界转换，不把内部模型异常直接暴露给 Core。

## 分层与依赖方向

主要依赖方向如下：

```text
src/plugin.py
  -> 具体 Provider
      -> Provider 自有适配模块
      -> 对应协议 Family
          -> providers/common
              -> core / schemas / i18n
```

| 层级 | 目录 | 职责 |
| --- | --- | --- |
| 插件边界 | `src/plugin.py` | 注册六个 Provider、读取配置、绑定语言、管理实例与 Store、统一分发和错误转换 |
| 具体 Provider | `src/providers/*_provider/` | 端点、鉴权、模型限制、供应商默认值、供应商错误结构和差异化行为 |
| 协议 Family | `src/providers/*_family/` | 复用协议级请求、响应、流式、工具、多模态、ASR、Embedding 和 transport 接口 |
| 通用 Provider 原语 | `src/providers/common/` | HTTP、SSE、参数翻译基础、音视频校验、向量校验和协议无关解析 |
| Core | `src/core/` | 运行选项、JSON 类型、参数目录/策略、解析、诊断脱敏和状态存储 |
| Schema | `src/schemas/` | Host 请求快照、Provider 返回合约和协议边界模型 |
| i18n | `src/i18n.py`、`locales/` | 稳定消息键、语言上下文和面向 Host 的本地化文本 |

依赖边界由 `tests/test_provider_architecture.py` 固定：

- 使用 Family 的具体 Provider 不得绕过 Family 直接依赖 `providers/common`。
- Family 可以依赖 `common`、`core`、`schemas` 和 `i18n`，但不得依赖任何具体 Provider。
- `common` 不得反向依赖 Family 或具体 Provider。
- Anthropic Messages 和 DashScope 是独立协议实现，可以直接使用 `common`。
- Provider 自有的 `tools.py`、`multimodal.py` 和参数模块必须进入真实 Mapper 调用链，不能只是未被调用的转发文件。

Family 的 `transport.py` 是有意设置的传输门面。使用 Family 的 Provider 应从该门面取得 HTTP 能力，以保持依赖方向稳定；协议无关的底层实现仍集中在 `providers/common/httpx.py`。

## 协议 Family

### `responses_family`

承载 OpenAI Responses 风格协议的共用实现，包括请求与响应映射、流式状态、工具、多模态、格式控制、ASR、Embedding 和参数翻译。

- `openai_responses_provider` 使用它实现 Responses 主链路。
- `volcengine_ark_provider` 使用它实现 Responses、ASR 和 Embedding，并在 Provider 层增加 ARK 端点、参数差异和显式前缀缓存。

### `chat_completions_family`

承载 Chat Completions 风格协议，包括消息转换、工具、多模态、流式增量、格式控制和音频处理。

- `siliconflow_provider` 使用它实现 Chat Completions 主链路。
- `xiaomi_mimo_provider` 使用它实现 Chat Completions 与专用 ASR，并在 Provider 层处理 reasoning 连续性。

### `openai_auxiliary_family`

承载 OpenAI 兼容但不属于文本生成主协议的辅助能力：multipart 音频转录和 Embedding。

- `openai_responses_provider` 使用它实现 ASR 与 Embedding。
- `siliconflow_provider` 使用它实现 ASR 与 Embedding。

### 独立协议 Provider

`anthropic_messages_provider` 和 `dashscope_provider` 没有强行套入现有 Family：

- Anthropic 使用 Messages 请求结构和事件类型，独立处理 system、thinking、tool use、tool result 与图片内容。
- DashScope 按模型和能力选择原生端点，独立处理文本生成、多模态 ASR、Embedding、错误结构和 SSE。

共享应以协议语义一致为前提。只有字段长得相似但状态机、错误语义或能力约束不同的代码，不应为了减少文件数量被合并进 Family。

## 能力归属

| Provider | `response` | `audio_transcription` | `embedding` |
| --- | --- | --- | --- |
| OpenAI Responses | `responses_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| Anthropic Messages | 独立实现 | 不支持 | 不支持 |
| Volcengine ARK | `responses_family` | `responses_family` | `responses_family` |
| DashScope | 独立实现 | 独立实现 | 独立实现 |
| SiliconFlow | `chat_completions_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| Xiaomi Mimo | `chat_completions_family` | `chat_completions_family` 加 Provider 特化 | 不支持 |

同一 Provider 可以按能力进入不同 Family。能力归属取决于实际端点协议，而不是供应商名称。

## 具体 Provider 的适配职责

Family 定义协议骨架，具体 Provider 通过显式适配模块注入供应商差异。常见目录如下，实际文件按能力增减：

```text
<name>_provider/
  provider.py              # LLMProviderBase、客户端与能力分发
  responses.py / chat.py   # 请求映射和非流式响应解析
  embeddings.py            # Embedding 入口
  audio_transcriptions.py  # ASR 入口
  streaming.py             # 供应商流式收集门面
  multimodal.py            # 图片、音频等内容映射
  tools.py                 # 工具定义、调用和结果转译
  parameter_translation.py # Host 参数到上游 body/header/query
```

新增差异时应放在最窄的正确层级：

1. 只有端点、鉴权、模型或供应商策略不同，放在具体 Provider。
2. 多个供应商共享同一上游协议语义，放在对应 Family。
3. 与供应商和协议均无关的 HTTP、校验或解析原语，放在 `common`。
4. Host 合约、全局参数规则、脱敏或持久化等跨 Provider 规则，放在 `core` 或 `schemas`。

## 请求快照与响应合约

Provider 入口先把 Core 传入的字典校验为 `src/schemas/host_snapshots.py` 中的请求快照：

- `ResponseRequestSnapshot`
- `EmbeddingRequestSnapshot`
- `AudioTranscriptionRequestSnapshot`

请求快照隔离 SDK/Core 的原始结构，Provider 和 Family 不应在深层反复猜测裸字典形状。协议响应经过对应 Schema 或 JSON 窄化函数后，统一构造 `ProviderResponse`，最后调用 Host 序列化接口返回 JSON object。

`src/schemas/` 的主要边界为：

| 模块 | 边界 |
| --- | --- |
| `host_snapshots.py` | Core 到插件的请求快照 |
| `provider_contracts.py` | 插件到 Core 的文本、工具、reasoning 与 usage 合约 |
| `responses_compat.py` | Responses API 的响应与流式兼容模型 |
| `anthropic_messages.py` | Anthropic Messages 的内容块与事件模型 |
| `sdk_dump.py` | 将 SDK/Pydantic 对象稳定转换为 JSON 数据 |
| `usage.py` | 不同协议 usage 字段的归一化 |

## 参数管线

参数合并和传输策略分为两个阶段。

第一阶段由 `build_normalized_host_parameters()` 生成标准化 Host 参数，优先级从低到高为：

1. `policy.default_params`
2. `model_info.extra_params`
3. model typed fields
4. `request.extra_params`
5. request typed fields

第二阶段由具体 Provider 翻译器和 `apply_transport_parameter_policy()` 完成：

```text
标准化 Host 参数
  -> ParameterCatalog 定义字段、别名、类型和目标路径
  -> Provider 翻译器生成 TranslationEnvelope(body, headers, query)
  -> rejected_paths 拒绝不允许的显式值
  -> disabled_paths 删除禁用值
  -> override_params 写入最终强制值
  -> transport 发送请求
```

`unknown_extra_params` 决定未知参数是 `forward`、`drop` 还是 `reject`。新增参数不能只在某个请求构造函数里临时读取；应同时明确目录定义、能力范围、目标位置、冲突优先级和策略行为。

## HTTP、流式与诊断

`providers/common/httpx.py` 提供客户端配置、URL 解析、鉴权、JSON/multipart POST、重试和 SSE 读取。Provider 或 Family 负责把供应商语义传入该层，`common` 不判断具体模型能力。

流式收集器必须维护协议状态，而不只是拼接文本。需要保留的内容包括工具参数分片、reasoning、usage、完成原因和协议终态；首个有效事件已经发出后发生的中断不得作为整次请求重新发送。

向日志或 Host 暴露上游错误前必须经过 `src/core/diagnostics.py` 脱敏。API Key、认证头、完整请求、响应正文和 reasoning 不得直接写入日志。

## 生命周期与状态

六个 Provider 均由 `MaiDockPlugin` 注册并惰性构造。同一配置代次内复用实例：

- `on_load` 校验语言目录、清空运行缓存，并使用 Core 授予的数据目录初始化 `PluginStateStore`。
- `on_config_update` 使运行选项和所有 Provider 实例失效；下一次调用按新配置重建。
- `on_unload` 清空实例并关闭 Store；重复卸载保持安全。
- Store 文件固定为 `ctx.paths.data_dir / "maidock_state.sqlite3"`，首次读写时才打开。

ARK 的显式前缀缓存仅在功能启用时要求 Store。Mimo 构造时要求 Store，用独立 namespace 保存 reasoning 连续性数据。两者不得共享键空间，凭据只能参与不可逆摘要。

## 新增或扩展 Provider

1. 确认上游端点实际属于 Responses、Chat Completions、OpenAI 辅助协议还是独立协议。
2. 在最窄层级实现差异，避免具体 Provider 复制已有 Family 状态机。
3. 为每项能力定义请求快照入口、参数目录、策略、body/header/query 和响应合约。
4. 明确非流式与流式终态、usage、工具、多模态、reasoning 和错误结构。
5. 在 `src/plugin.py` 注册入口并验证惰性构造、配置更新和卸载行为。
6. 同步配置 Schema、README 能力矩阵、参数文档和五份 `locales/*.json` 语言目录文件；版本与 manifest 仍需维护者明确授权。
7. 使用 `httpx.MockTransport` 或等价本地替身补齐请求、响应、错误和并发测试，不调用真实收费接口。
8. 运行架构测试，确认依赖方向和 Provider 门面均进入真实调用链。

架构相关的最低验证命令：

```powershell
uv run --locked pytest -q tests/test_provider_architecture.py tests/test_plugin_dispatch.py tests/test_plugin_lifecycle.py
uv run --locked ruff check plugin.py src tests
uv run --locked pyright
```
