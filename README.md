# MaiDock

MaiDock 是一个 MaiBot LLM Provider 插件，用于补充主程序未覆盖的端点。

目前已实现：

- `maidock-openai-responses`: OpenAI Responses API。
- `maidock-anthropic-messages`: Anthropic Messages API。

## Provider 能力概览

### `maidock-openai-responses`

基础能力：

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 文本响应 | ✅ | 基础对话与文本生成。 |
| 多模态图片输入 | ✅ | 转换为 Responses API 的 input image。 |
| 工具调用 | ✅ | 原生工具调用，兼容部分 XML 工具调用输出。 |
| 推理/思考 | ✅ | 透传 `extra_params.reasoning`，并提取 `reasoning_content`。 |

额外能力：

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| JSON / `response_format` | ✅ | 转换为 Responses API 的 `text.format`。 |
| Embedding | 🧪 | 复用 OpenAI SDK embeddings 端点。 |
| 音频转写 | 🧪 | 复用 OpenAI SDK audio transcriptions 端点。 |

### `maidock-anthropic-messages`

基础能力：

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 文本响应 | ✅ | 基础对话与文本生成。 |
| 多模态图片输入 | ✅ | 支持 Anthropic Messages 图片块。 |
| 工具调用 | ✅ | 原生工具调用，兼容部分 XML 工具调用输出。 |
| 推理/思考 | ✅ | 透传 `extra_params.thinking`，并提取 `reasoning_content`。 |

## 注意事项

### 图片，多模态相关 / 帧大小超过限制

报错信息为：**"插件 LLM Provider RPC 调用失败: [E_UNKNOWN] 帧大小 xxx 超过最大限制 16777216"**

当前由于传输层有 16 MB 单帧限制。如果发送大图，图片 base64 可能在到达本插件前就让 RPC 帧超过 16 MB。

WebUI 可视化界面暂无**最大图片数量**的设置界面，请切换到上方源代码后修改配置文件。

配置文件 `bot_config.toml` 关键字段：

```toml
[visual]
max_image_num = 1 # 建议为 1, 具体视上下文长度与单图片大小而定
max_image_size_mb = 5 # 视情况而定
```

### 超时

报错信息为：**插件 LLM Provider RPC 调用失败: [E_TIMEOUT] 请求 plugin.invoke_llm_provider 超时 (30000ms)**

建议操作：

- 变更默认超时(30s)设置
- 更换响应更快的模型、提供商

模型配置文件 `model_config.toml` 关键字段：

```toml
[[api_providers]]
timeout = 30 # 默认值为 30
```

## 模型配置示例

### OpenAI Responses

```toml
[[api_providers]]
name = "maidock-openai"
client_type = "maidock-openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"
auth_type = "bearer"

[[models]]
name = "Responses/gpt-5.5"
api_provider = "maidock-openai"
model_identifier = "gpt-5.5"
visual = true
extra_params = { reasoning = { effort = "medium" } }
```

### Anthropic Messages

```toml
[[api_providers]]
name = "maidock-anthropic-messages"
client_type = "maidock-anthropic-messages"
base_url = "https://api.anthropic.com"
# 自动兼容 /v1
api_key = "your-api-key"
auth_type = "header"
auth_header_name = "x-api-key"
# 也可配置为 auth_type = "bearer"；MaiDock 会在 Anthropic Provider 内部转换为 SDK 的 api_key/X-Api-Key 鉴权。

[[models]]
name = "messages/claude-opus-4-8"
api_provider = "maidock-anthropic-messages"
model_identifier = "claude-opus-4-8"
visual = true
extra_params = { thinking = { type = "enabled", budget_tokens = 1024 } }
```

## 插件配置

插件配置文件通常是本插件目录下的 `config.toml`，用于控制 MaiDock 自身的诊断日志、兼容模式、默认 User-Agent 和图片处理限制。模型、API Provider 与单次请求参数仍在 MaiBot 的 `model_config.toml` 中配置。

### 完整示例

```toml
[plugin]
enabled = true
config_version = "1.0.3"

[diagnostics]
include_raw_data = false
log_payload_summary = true
log_payload_debug = false
anthropic_sdk_log_level = "INFO"

[openai_responses]
user_agent = ""

[anthropic_messages]
user_agent = ""

[compatibility]
tool_argument_parse_mode = "auto"
reasoning_parse_mode = "auto"
strict_extra_params = false
invalid_image_policy = "placeholder"
max_image_bytes_mb = 30
max_image_pixels = 25000000
max_image_dimension = 8192
max_image_frames = 64
```

### `[plugin]`

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否启用 MaiDock 插件。 |
| `config_version` | 当前插件版本 | 配置文件版本标记，一般不需要手动修改。 |

### `[diagnostics]`

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `include_raw_data` | `false` | 是否把脱敏后的上游响应摘要放入 Host `raw_data`。调试时可打开，日常建议关闭。 |
| `log_payload_summary` | `true` | 是否记录脱敏后的请求/响应摘要日志。 |
| `log_payload_debug` | `false` | 是否记录脱敏后的详细请求载荷。会输出更多内容，仅建议排查问题时临时开启。 |
| `anthropic_sdk_log_level` | `INFO` | Anthropic SDK logger 级别。可选值: `inherit`、`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`；`inherit` 表示不修改 SDK logger。 |

### Provider 独立配置

| 配置段 | 配置项 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `[openai_responses]` | `user_agent` | `""` | OpenAI Responses Provider 使用的默认 User-Agent。留空时使用 MaiDock 内置默认值。 |
| `[anthropic_messages]` | `user_agent` | `""` | Anthropic Messages Provider 使用的默认 User-Agent。留空时使用 MaiDock 内置默认值。 |

#### User-Agent 生效优先级

优先级从高到低如下，命中高优先级后就不会再使用低优先级：

1. `model_config.toml` 中当前 `[[api_providers]]` 的 `default_headers.User-Agent`。
2. 插件配置中对应 Provider 的 `user_agent`。
3. MaiDock 内置默认值 `MaiDock/<版本号>`。

如果只想给某一个模型供应商单独设置 User-Agent，推荐写在 `model_config.toml` 对应的 `[[api_providers]]` 下：

```toml
[[api_providers]]
name = "maidock-openai"
client_type = "maidock-openai-responses"
# 其他字段略...

[api_providers.default_headers]
"User-Agent" = "Mozilla/5.0"
```

如果希望同类 Provider 共用一个 User-Agent，则写在插件配置里：

```toml
[openai_responses]
user_agent = "Mozilla/5.0"

[anthropic_messages]
user_agent = "Mozilla/5.0"
```

### `[compatibility]`

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `tool_argument_parse_mode` | `auto` | 工具调用参数解析模式。通常保持 `auto` 即可；需要严格排查模型输出格式时再改为其他模式。 |
| `reasoning_parse_mode` | `auto` | 推理/思考内容解析模式。通常保持 `auto` 即可。 |
| `strict_extra_params` | `false` | 是否拒绝未知 `extra_params` 字段。`false` 时未知字段会进入 `extra_body`；`true` 时直接报错，适合排查配置拼写问题。 |
| `invalid_image_policy` | `placeholder` | 无效图片处理策略，见下方说明。 |
| `max_image_bytes_mb` | `30` | 单张图片 base64 解码后的最大字节数，非正数会回退默认值。 |
| `max_image_pixels` | `25000000` | 单张图片最大像素数量，同时用于 Pillow decompression bomb 防护。 |
| `max_image_dimension` | `8192` | 单张图片单边最大像素。 |
| `max_image_frames` | `64` | 动图最大帧数。 |

`invalid_image_policy` 可选值：

- `placeholder`: 把无效图片替换为 `[图片内容不可用]`。
- `skip`: 跳过无效图片。
- `error`: 直接报错。

## extra_params 约定

`extra_params` 用来把模型级和请求级的额外参数传给上游 SDK。MaiDock 会先合并两处配置：

1. `model_config.toml` 中 `[[models]]` 的 `extra_params`。
2. Host 在单次调用里传入的 `extra_params`。

如果两边有同名字段，单次调用里的值优先生效；值为 `null` 的字段会被忽略。

### 通用拆分规则

```toml
extra_params = {
  headers = { X-Trace-Id = "trace-1" },
  query = { debug = "true" },
  body = { custom_field = "value" },
  metadata = { source = "maibot" }
}
```

| 字段 | 去向 | 说明 |
| --- | --- | --- |
| `headers` | SDK `extra_headers` | 仅接受字符串键值对，适合传 `X-Trace-Id`、临时鉴权头等请求头。 |
| `query` | SDK `extra_query` | 追加到请求 URL query。 |
| `body` | SDK `extra_body` | 直接合入请求 body，适合上游兼容接口的非标准 body 字段。 |
| Provider 支持的顶层字段 | SDK 原生命名参数 | 例如 `reasoning`、`thinking`、`metadata` 等，见下方列表。 |
| 未识别顶层字段 | 默认进入 SDK `extra_body` | 如果 `[compatibility].strict_extra_params = true`，则不会自动放入 `extra_body`，而是直接报错。 |

注意：`headers` 必须是字符串到字符串的映射；`query`、`body` 需要是 object/table。

### OpenAI Responses

响应请求中，以下 `extra_params` 顶层字段会作为 `client.responses.create(...)` 的原生命名参数传入：

- `include`
- `instructions`
- `max_output_tokens`
- `metadata`
- `parallel_tool_calls`
- `previous_response_id`
- `reasoning`
- `service_tier`
- `store`
- `text`
- `tool_choice`
- `top_p`
- `truncation`
- `user`

示例：

```toml
extra_params = {
  reasoning = { effort = "medium" },
  text = { verbosity = "medium" },
  metadata = { source = "maibot" }
}
```

特殊规则：

- `input`、`model`、`stream`、`temperature`、`tools` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- `max_output_tokens` 如果写在 `extra_params` 顶层，会覆盖 MaiBot 的 `max_tokens` / 模型 `max_tokens` 换算结果。
- `text` 会和 MaiBot 的 `response_format` 合并；如果两边同时设置了冲突的格式字段，会直接报错。

OpenAI Embeddings 和 Audio Transcriptions 也会使用同一套 `headers` / `query` / `body` 拆分规则；除这三个分组外，其他顶层字段默认进入 `extra_body`，开启 `strict_extra_params` 后会报错。

### Anthropic Messages

响应请求中，以下 `extra_params` 顶层字段会作为 `client.messages.create(...)` 的原生命名参数传入：

- `metadata`
- `service_tier`
- `stop_sequences`
- `thinking`
- `tool_choice`
- `top_k`
- `top_p`

示例：

```toml
extra_params = {
  thinking = { type = "enabled", budget_tokens = 1024 },
  stop_sequences = ["\n\nHuman:"],
  metadata = { source = "maibot" }
}
```

特殊规则：

- `max_tokens`、`messages`、`model`、`stream`、`system`、`temperature`、`tools` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- `tool_choice` 可以覆盖 MaiDock 自动生成的默认工具选择策略。
- 其他未识别顶层字段默认进入 `extra_body`；开启 `[compatibility].strict_extra_params = true` 后会报错。
