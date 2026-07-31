# 插件配置参考

插件配置文件为 MaiDock 目录下的 `config.toml`。模型、API Provider 与单次请求参数仍在 MaiBot 的 `model_config.toml` 中配置。

> 💡 大部分配置项可通过 MaiBot WebUI 的**"插件配置"标签页**直接修改。此文档面向需要手动编辑 TOML 的场景。

---

## `[plugin]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `locale` | str | `"zh-CN"` | MaiDock 配置页、日志及向 Host/RPC 返回的错误文本语言；可选 `zh-CN` / `zh-TW` / `en-US` / `ja-JP` / `ko-KR`。保存后新请求立即生效，配置页需重新打开或刷新 |
| `enabled` | bool | `true` | 是否启用 MaiDock 插件 |
| `config_version` | str | `"1.2.0"` | 配置版本标记，一般不需要手动修改 |

---

## `[diagnostics]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `include_raw_data` | bool | `false` | 是否把脱敏后的上游响应摘要放入 Host `raw_data` |
| `log_payload_summary` | bool | `true` | 是否记录脱敏后的请求/响应摘要日志 |
| `log_payload_debug` | bool | `false` | 是否记录脱敏后的详细请求载荷 |

脱敏会移除凭据、认证头和 Base64 数据，但不是“完全不保留内容”：prompt、instruction、content、text 等文本字段最多保留前 300 个字符，超出部分才截断。生产环境开启 `include_raw_data` 或 `log_payload_debug` 前，应确认 Host 返回值和日志的访问权限满足隐私要求。

---

## `[compatibility]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tool_argument_parse_mode` | select | `"auto"` | 工具调用参数解析模式：`auto` / `strict` / `repair` / `double_decode` |
| `reasoning_parse_mode` | select | `"auto"` | 推理内容解析模式：`auto` / `native` / `think_tag` / `none` |
| `invalid_image_policy` | select | `"placeholder"` | 无效图片处理策略：`placeholder`（替换为占位文本）/ `skip`（跳过）/ `error`（报错） |
| `max_image_bytes_mb` | int | `30` | 单张图片 base64 解码后最大字节数（MB），非正数回退默认值 |
| `max_image_pixels` | int | `25000000` | 单张图片最大像素数，同时用于 Pillow decompression bomb 防护 |
| `max_image_dimension` | int | `8192` | 单张图片单边最大像素 |
| `max_image_frames` | int | `64` | 动图最大帧数 |


---

## 跨插件 Public API

Public API 配置与 Host LLM Provider 配置完全独立。公共配置只管理开关、默认 Profile 和资源限制；供应商连接与参数策略放在对应 Profile 中。WebUI 位于“跨插件 API”标签页。

### `[public_api]`

```toml
[public_api]
enabled = false
default_image_profile = ""
default_video_profile = ""
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否注册 12 个跨插件图像/视频 API；关闭后立即停止接单并注销 API |
| `default_image_profile` | str | `""` | 图像请求未指定 `profile` 时使用的全局 Profile 名称 |
| `default_video_profile` | str | `""` | 视频请求未指定 `profile` 时使用的全局 Profile 名称 |

Profile 名称在全部 Public API 供应商之间必须全局唯一。默认 Profile 留空时，调用方必须显式传入 `profile`。

### `[public_api.resources]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `max_concurrent_jobs` | int | `2` | 同时运行的作业数，范围 `1..32` |
| `max_queued_jobs` | int | `32` | 排队作业上限，范围 `1..1024` |
| `max_upload_mb` | int | `512` | 单个上传上限 MiB |
| `max_artifact_mb` | int | `512` | 单个产物上限 MiB |
| `storage_quota_gb` | int | `10` | 上传、产物和 staging 总配额 GiB |
| `incomplete_upload_ttl_hours` | int | `24` | 未完成上传保留小时数 |
| `completed_upload_ttl_days` | int | `7` | 完成上传保留天数 |
| `artifact_ttl_days` | int | `7` | 产物保留天数 |
| `job_metadata_ttl_days` | int | `30` | 作业元数据、幂等记录和删除墓碑保留天数 |
| `max_tracking_hours` | int | `23` | 远端任务最长跟踪小时数 |

### `[[public_api.dashscope.profiles]]`

```toml
[[public_api.dashscope.profiles]]
name = "dashscope-main"
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/api/v1"
workspace_id = ""
default_image_model = "qwen-image-2.0"
default_video_model = "wan2.6-t2v"
connect_timeout_seconds = 10.0
request_timeout_seconds = 1800.0
safe_max_retries = 3
retry_interval_seconds = 1.0

[[public_api.dashscope.profiles.image_default_parameters]]
name = "size"
value_type = "string"
value = "1024*1024"

[[public_api.dashscope.profiles.image_override_parameters]]
name = "watermark"
value_type = "boolean"
value = "false"

[[public_api.dashscope.profiles.protocol_routes]]
capability = "image_generation"
model = "custom-image-model"
mode = "text_to_image"
protocol_family = "dashscope_text2image_synthesis"
```

| 配置项 | 说明 |
| --- | --- |
| `name` | 全局唯一 Profile 名称 |
| `api_key` | DashScope 上游密钥；WebUI 使用普通字符串控件并明文展示，日志和异常诊断会脱敏 |
| `base_url` | 必须是 HTTPS、无 URL 用户凭据/query/fragment，且路径以 `/api/v1` 结尾 |
| `workspace_id` | 可选的 DashScope workspace header |
| `default_image_model` / `default_video_model` | 对应能力未传 `model` 时使用的模型 |
| `connect_timeout_seconds` / `request_timeout_seconds` | 建连和单次上游请求超时 |
| `safe_max_retries` / `retry_interval_seconds` | tasks 查询/取消、上传策略和下载等安全操作的有限重试；生成提交固定不重试 |
| 四个 `*_parameters` | 参数项列表，每项填写 `name`、`value_type`、`value`；参数优先级为 defaults、请求 parameters、overrides |
| `protocol_routes` | 可增删精确路由；`mode=""` 表示该模型与 capability 的全部模式 |

`protocol_family` 可选值为 `dashscope_multimodal_generation`、`dashscope_image_generation`、`dashscope_text2image_synthesis`、`dashscope_image2image_synthesis`、`dashscope_video_generation`。未知模型必须通过请求或 Profile route 显式锁定协议族，不进行端点探测或 fallback。

`value_type` 可选 `string`、`integer`、`number`、`boolean`、`json`、`null`。`value` 始终在 WebUI 中按文本填写：`boolean` 只接受 `true`/`false`，`json` 只接受 object/array，`null` 必须留空；同一参数列表内的 `name` 不得重复。

### `[[public_api.volcengine_ark.profiles]]`

```toml
[[public_api.volcengine_ark.profiles]]
name = "ark-main"
api_key = "..."
base_url = "https://ark.cn-beijing.volces.com/api/v3"
default_image_model = "doubao-seedream-4-0"
default_video_model = "doubao-seedance-2-0"
connect_timeout_seconds = 10.0
request_timeout_seconds = 1800.0
safe_max_retries = 3
retry_interval_seconds = 1.0

[[public_api.volcengine_ark.profiles.video_default_parameters]]
name = "duration"
value_type = "integer"
value = "5"

[[public_api.volcengine_ark.profiles.video_override_parameters]]
name = "generate_audio"
value_type = "boolean"
value = "true"

[[public_api.volcengine_ark.profiles.protocol_routes]]
capability = "video_generation"
model = "ep-20260101-abcde"
mode = ""
protocol_family = "ark_content_generation_tasks"
```

字段含义与 DashScope Profile 一致，差异如下：

| 配置项 | 说明 |
| --- | --- |
| `base_url` | 必须是 HTTPS、无 URL 用户凭据/query/fragment，且路径以 `/api/v3` 结尾 |
| `workspace_id` | **不存在**。ARK 的多租户靠 API Key 本身区分 |
| `protocol_family` | 可选值为 `ark_images_generations`、`ark_content_generation_tasks` |

`name` 在 `[public_api.dashscope]` 与 `[public_api.volcengine_ark]` 之间**必须全局唯一**。调用方只按 Profile 名寻址、不带供应商前缀，同名会让请求落到哪一家变得不确定，因此配置加载阶段就会报错。

以 `ep-` 开头的接入点 ID 不在内置模型目录中，必须通过请求或上面这样的 Profile route 显式锁定协议族。

接口参数、上传流程和作业状态见 [跨插件图像与视频 Public API](media_api_reference.md)。

---

## Provider 配置

每个 Provider 的配置分为两类：

1. **Provider 级**字段：`user_agent` + 重试配置（所有 Provider），部分含 `force_official_endpoint`；ARK 额外提供前缀缓存配置，Mimo 提供 reasoning 持久化配置
2. **能力子段**：`[{provider}.{capability}]` — 包含该 Provider 某项能力的参数覆写目录 `[{provider}.{capability}.overrides]`，每个参数一个覆写键（空白表示不覆写，覆写值拥有最终优先级）

能力子段与覆写目录的完整字段说明见 [能力参数覆写](#能力参数覆写-capabilityparameteroverridesconfig)。

### `[openai_responses]`

```toml
[openai_responses]
user_agent = ""
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent，留空时使用 MaiDock 内置默认值 |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值（Host 未配置时使用），开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式（Host 提供值时优先），开启=始终使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式（Host 提供值时优先），开启=始终使用上方的值 |

该 Provider 拥有以下能力子段：

| 子段 | 说明 |
| --- | --- |
| `[openai_responses.response]` | 文本生成（Responses API） |
| `[openai_responses.embeddings]` | Embeddings |
| `[openai_responses.audio_transcription]` | 语音转录 |

### `[anthropic_messages]`

```toml
[anthropic_messages]
user_agent = ""
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[anthropic_messages.chat_completion]` | 文本生成（Messages API） |

### `[dashscope]`

```toml
[dashscope]
user_agent = ""
force_official_endpoint = true
auto_detect_endpoint = true
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `force_official_endpoint` | bool | `true` | 是否忽略 Host 提供的 `base_url`，强制使用阿里云百炼 DashScope 官方 endpoint；使用百炼工作空间域名时需关闭 |
| `auto_detect_endpoint` | bool | `true` | 无图片请求遇到结构化 `InvalidParameter + url error` 时，是否向文本/多模态的相反端点重试一次，并在内存中记录成功的端点类型 |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[dashscope.chat_completion]` | 文本生成（Generation API） |
| `[dashscope.embeddings]` | Embeddings（文本 / 多模态） |
| `[dashscope.audio_transcription]` | 语音转录（多模态生成端点） |

DashScope 的端点类型缓存不保存绝对 URL，每次请求都会基于当前 Host `base_url` 重新拼接路径，因此切换工作空间域名后不会复用旧域名。实际含图片的请求固定使用多模态端点，已知纯文本模型会在本地拒绝图片，任何图片请求都不会回退文本端点。

### `[siliconflow]`

```toml
[siliconflow]
user_agent = ""
force_official_endpoint = true
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `force_official_endpoint` | bool | `true` | 是否忽略 Host 提供的 `base_url`，强制使用 SiliconFlow 官方 endpoint |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[siliconflow.chat_completion]` | 文本生成（Chat Completions API） |
| `[siliconflow.embeddings]` | Embeddings |
| `[siliconflow.audio_transcription]` | 语音转录 |

### `[volcengine_ark]`

```toml
[volcengine_ark]
user_agent = ""
force_official_endpoint = true
builtin_endpoint_mode = "standard"
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
prefix_cache_enabled = false
prefix_cache_ttl_seconds = 259200
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `force_official_endpoint` | bool | `true` | 是否忽略 Host 提供的 `base_url`，强制使用火山方舟官方 endpoint |
| `builtin_endpoint_mode` | str | `"standard"` | 内置端点类型，仅在开启上方开关时生效。`standard`=按量付费 `/api/v3`；`agent_plan`=Agent Plan 订阅 `/api/plan/v3`（需其专属 API Key）；`coding_plan`=Coding Plan 订阅 `/api/coding/v3`。订阅端点仅覆盖 Responses 文本链路，选择后前缀缓存自动停用；embeddings/tokenization 在订阅端点上无官方文档，若上游不支持会返回明确错误 |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `prefix_cache_enabled` | bool | `false` | 是否自动管理 ARK Responses 显式前缀缓存。需要 Core 1.0.9，并需先开启方舟“推理（缓存）”计价 |
| `prefix_cache_ttl_seconds` | int | `259200` | 缓存有效期秒数，范围 `3600..604800`；使用缓存不会延长有效期 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[volcengine_ark.response]` | 文本生成（Responses API） |
| `[volcengine_ark.embeddings]` | Embeddings |
| `[volcengine_ark.audio_transcription]` | 语音转录（Responses `input_audio`） |

### `[bailian_responses]`

```toml
[bailian_responses]
user_agent = ""
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[bailian_responses.response]` | 文本生成（OpenAI Responses 规范） |

百炼 Responses 只使用 Host `api_provider.base_url`（必须以 `/v1` 结尾，MaiDock 自动追加 `/responses`），不读取原生 DashScope 的 `force_official_endpoint` 或自动端点探测。官方 base URL：北京 `https://dashscope.aliyuncs.com/compatible-mode/v1`、新加坡 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`、美国 `https://dashscope-us.aliyuncs.com/compatible-mode/v1`、日本 `https://dashscope-jp.aliyuncs.com/compatible-mode/v1`、德国 `https://dashscope-eu.aliyuncs.com/compatible-mode/v1`。

### `[xiaomi_mimo]`

```toml
[xiaomi_mimo]
user_agent = ""
reasoning_retention_days = 30
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `reasoning_retention_days` | int | `30` | 带工具调用轮次的完整 reasoning 本地保留天数，范围 `1..365`；成功使用时续期 |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[xiaomi_mimo.chat_completion]` | 文本生成（Chat Completions API） |
| `[xiaomi_mimo.audio_transcription]` | 语音转录；使用 Chat Completions 端点的专用单音频 ASR 协议 |

Mimo 思考默认由 `[xiaomi_mimo.chat_completion.overrides]` 中的 `thinking`（默认 `{"type":"disabled"}`）控制；改为 `enabled` 或清空后启用思考，此时缺少状态存储会在请求阶段明确报错。ASR 语言由 `[xiaomi_mimo.audio_transcription.overrides]` 中的 `language`（默认 `auto`）控制。

Mimo reasoning 只保存带工具调用的 assistant 轮次，因为 Core 目前仅为工具调用提供可稳定往返的 `extra_content` 和 call ID。完整内容以明文保存在 `maidock_state.sqlite3` 的独立 namespace 中；API Key、提示词和工具定义不会写入数据库。

### 重试配置的 force/fallback 逻辑

每个 Provider 的 4 个重试字段（`max_retries` / `force_max_retries` / `retry_interval` / `force_retry_interval`）遵循一致的解析逻辑：

```
if force == true:
    effective = config_value          → 始终使用插件配置值，忽略 Host
else:
    if Host 提供了有效值:
        effective = Host_value        → 回退模式，Host 优先
    else:
        effective = config_value      → Host 未配置时，使用插件配置值
```

**`retry_interval` 特殊行为：** 非 force 模式下，`0` 视为无效值（设计意图：避免意外启用零间隔重试）。若确实需要零间隔立刻重试，请将 `force_retry_interval = true` 并将 `retry_interval = 0.0`。

这条链路在代码中由 `resolve_max_retries()` 和 `resolve_retry_interval()`（`src/core/common.py`）实现，所有 Provider 的重试值均通过此机制计算后传入底层 HTTP 函数。

**Host 侧配置：** Host（MaiBot 核心）在 `model_config.toml` 的 `[[api_providers]]` 中提供了 `max_retry`（默认 3）和 `retry_interval`（默认 5）字段。当插件侧 `force_* = false` 时，这些 Host 值会被优先使用。

### User-Agent 生效优先级

1. `model_config.toml` 中 `[[api_providers]]` 的 `default_headers.User-Agent`（最高）
2. 插件配置中对应 Provider 的 `user_agent`
3. MaiDock 内置默认值 `MaiDock/<版本号>`（最低）

```toml
# 方式 1：按模型供应商单独设置（推荐）
[[api_providers]]
name = "maidock-openai"
client_type = "maidock-openai-responses"
[api_providers.default_headers]
"User-Agent" = "Mozilla/5.0"

# 方式 2：按 Provider 类型统一设置
[openai_responses]
user_agent = "Mozilla/5.0"
```

---

## 能力参数覆写 (CapabilityParameterOverridesConfig)

每个 `[{provider}.{capability}]` 子段包含一个 **`overrides` 子表**：

```toml
[dashscope.chat_completion.overrides]
temperature = ""
max_tokens = ""
enable_search = ""
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `overrides` | table | `{}` | 参数覆写目录；键为目录参数名，值为字符串（空白表示不覆写） |

- 所有覆写值均为字符串：布尔使用 `true/false`，整数和浮点数使用 JSON 数字，数组和对象使用合法 JSON；字符串类型字段直接写文本，不需要 JSON 引号。
- 覆写值拥有最终优先级：覆盖 Core 请求级类型字段；同一对象路径采用叶级合并。
- 稳定请求默认值已写入覆写框默认文本（如 DashScope `result_format=message`、SiliconFlow/Ark Embedding `encoding_format=float`、Ark ASR `prompt`、Mimo ASR `language=auto`、Mimo `thinking={"type":"disabled"}`、百炼 `store=false`），可随时修改或清空。
- 参数名、目标路径、值类型与说明见 [参数覆写参考](extra_params_reference.md)；WebUI 每个参数渲染一个跨双列全宽覆写框。
- `fields`、`default_params`、`override_params`、`accept_*`、`unknown_extra_params`、`disabled_paths`、`rejected_paths` 等旧结构已被删除；1.1.3 配置会自动迁移，迁移后不再残留。

**示例：强制所有 OpenAI Responses 请求使用 `temperature = 0.7`**

```toml
[openai_responses.response.overrides]
temperature = "0.7"
```

Host 请求级与模型级 `extra_params` 不参与组包，且任何值类型都会被忽略。MaiDock 只转译 Core 明确定义的 typed fields，再以本节的非空覆写值执行叶级覆盖。`tools` 覆写不会删除 Host Function tools：Host 工具在前，覆写工具按配置顺序追加，不去重。
