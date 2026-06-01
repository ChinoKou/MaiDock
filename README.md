# MaiDock

MaiDock 是一个 MaiBot LLM Provider 插件，用于补齐主程序原生客户端当前未覆盖的端点: 

目前已实现：
- `maidock-openai-responses`: OpenAI Responses API。
- `maidock-anthropic-messages`: Anthropic Messages API。

## Provider 能力矩阵

| client_type | response | vision | tool calling | reasoning/thinking | JSON response_format | embedding | audio transcription |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `maidock-openai-responses` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ 未测试 | ✅ |
| `maidock-anthropic-messages` | ✅ | ✅ | ✅ | ✅ | N/A | ❌ | ❌ |

说明: MaiBot 当前插件 LLM Provider 链路会等待插件返回完整 dict，暂不支持 Host 侧自定义 streaming callback。MaiDock 在 `force_stream_mode = true` 时只做插件内部流式累积，最后一次性返回完整响应。`maidock-openai-responses` 的 embedding 链路已有实现但尚未做真实上游验证。

## 图片与插件 RPC 注意事项

MaiBot Host 通过插件 RPC 发送给 Runner，然后才进入 MaiDock。当前由于 maibot_sdk 传输层有 16 MB 单帧限制。
如果发送大图，`message_list` 中的图片 base64 可能在到达 MaiDock 前就让 RPC 帧超过 16 MB。可尝试修改 MaiBot 配置 `[visual]` 或 WebUI 可视化配置降低入站图片体积与多模态上下文图片数量。

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
# 兼容 https://api.anthropic.com/v1
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

```toml
[plugin]
enabled = true
config_version = "1.0.0"

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

`diagnostics`: 

- `include_raw_data`: 是否把脱敏后的上游响应摘要放入 Host `raw_data`，默认关闭。
- `log_payload_summary`: 是否记录请求/响应摘要日志。
- `log_payload_debug`: 是否记录脱敏后的详细请求载荷，默认关闭。
- `anthropic_sdk_log_level`: Anthropic SDK logger 级别，支持 `inherit`、`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`；`inherit` 表示不修改 SDK logger。

不同 Provider 独立配置: 

`openai_responses`: 
- `user_agent`: OpenAI Responses Provider 使用的自定义 User-Agent。

`anthropic_messages`
- `user_agent`: Anthropic Messages Provider 使用的自定义 User-Agent。留空时，MaiDock 会自动使用内置默认值 `MaiDock/<版本号>`。

若 user_agent 留空时，MaiDock 会自动使用内置默认值 `MaiDock/<版本号>`。。
如果配置文件 **model_config.toml** 中 `api_provider.default_headers` 已显式配置 `User-Agent`/`user-agent`，该请求级配置优先，MaiDock 不会覆盖。

`invalid_image_policy`: 

- `placeholder`: 把无效图片替换为 `[图片内容不可用]`。
- `skip`: 跳过无效图片。
- `error`: 直接报错。

图片处理资源上限: 

- `max_image_bytes_mb`: 单张图片 base64 解码后的最大字节数，非正数会回退默认值。
- `max_image_pixels`: 单张图片最大像素数量，同时用于 Pillow decompression bomb 防护。
- `max_image_dimension`: 单张图片单边最大像素。
- `max_image_frames`: 动图最大帧数。

## extra_params 约定

MaiDock 兼容 MaiBot OpenAI 风格的请求附加参数拆分: 

```toml
extra_params = {
  headers = { X-Trace-Id = "trace-1" },
  query = { debug = "true" },
  body = { custom_field = "value" }
}
```

- `headers` 进入 SDK `extra_headers`。
- `query` 进入 SDK `extra_query`。
- `body` 进入 SDK `extra_body`。
- Provider 明确支持的顶层字段会作为 SDK 原生参数传入。
- 未识别字段默认进入 `extra_body`；如果 `[compatibility].strict_extra_params = true`，则直接报错。

OpenAI Responses 常见顶层字段: `text`、`reasoning`、`metadata`、`parallel_tool_calls`、`previous_response_id`、`service_tier`、`store`、`top_p`、`truncation`、`tool_choice` 等。

Anthropic Messages 常见顶层字段: `thinking`、`metadata`、`stop_sequences`、`top_k`、`top_p`、`service_tier`、`tool_choice` 等。
