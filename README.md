# MaiDock

MaiDock 是一个 MaiBot LLM Provider 插件，用于补齐主程序原生客户端当前未覆盖的端点：

- `maidock-openai-responses`：OpenAI Responses API。
- `maidock-anthropic`：Anthropic Messages API。

## Provider 能力矩阵

| client_type | response | vision | tool calling | reasoning/thinking | JSON response_format | embedding | audio transcription |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `maidock-openai-responses` | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ 未测试 | ✅ |
| `maidock-anthropic` | ✅ | ✅ | ✅ | ✅ | N/A | ❌ | ❌ |

说明：MaiBot 当前插件 LLM Provider 链路会等待插件返回完整 dict，暂不支持 Host 侧自定义 streaming callback。MaiDock 在 `force_stream_mode = true` 时只做插件内部流式累积，最后一次性返回完整响应。`maidock-openai-responses` 的 embedding 链路已有实现但尚未做真实上游验证。

## 图片与插件 RPC 注意事项

插件 LLM Provider 请求会先由 MaiBot Host 通过插件 RPC 发送给 Runner，然后才进入 MaiDock。当前传输层有 16 MB 单帧限制。
如果发送大图，`message_list` 中的图片 base64 可能在到达 MaiDock 前就让 RPC 帧超过 16 MB。可尝试修改 MaiBot 配置 `[visual]` 或 WebUI 可视化配置降低入站图片体积与多模态上下文图片数量。

```toml
[visual]
handle_oversized_images = true
max_image_size_mb = 5.0 # 注意单张图片若过大仍可能触发单次识图 RPC 爆帧
oversized_image_handle_method = "compress" # 或 "discard"
max_image_num = 0 # 最稳妥，不向插件 LLM Provider 请求携带图片 base64，只保留文本/图片描述上下文
```

`max_image_num = 0` 模型仍能读取已有的文本、图片描述和识图结果，但不能重新直接看原图细节。需要让模型直接看图时可用 `max_image_num = 1`，但历史和重复图片累加后仍可能超过 RPC 单帧限制。

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
name = "maidock-anthropic"
client_type = "maidock-anthropic"
base_url = "https://api.anthropic.com"
# 兼容 https://api.anthropic.com/v1
api_key = "your-api-key"
auth_type = "header"
auth_header_name = "x-api-key"
# 也可配置为 auth_type = "bearer"；MaiDock 会在 Anthropic Provider 内部转换为 SDK 的 api_key/X-Api-Key 鉴权。

[[models]]
name = "messages/claude-opus-4-8"
api_provider = "maidock-anthropic"
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

`diagnostics`：

- `include_raw_data`：是否把脱敏后的上游响应摘要放入 Host `raw_data`，默认关闭。
- `log_payload_summary`：是否记录请求/响应摘要日志。
- `log_payload_debug`：是否记录脱敏后的详细请求载荷，默认关闭。
- `anthropic_sdk_log_level`：Anthropic SDK logger 级别，支持 `inherit`、`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`；`inherit` 表示不修改 SDK logger。

`invalid_image_policy`：

- `placeholder`：把无效图片替换为 `[图片内容不可用]`。
- `skip`：跳过无效图片。
- `error`：直接报错。

图片处理资源上限：

- `max_image_bytes_mb`：单张图片 base64 解码后的最大字节数，非正数会回退默认值。
- `max_image_pixels`：单张图片最大像素数量，同时用于 Pillow decompression bomb 防护。
- `max_image_dimension`：单张图片单边最大像素。
- `max_image_frames`：动图最大帧数。

## extra_params 约定

MaiDock 兼容 MaiBot OpenAI 风格的请求附加参数拆分：

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

OpenAI Responses 常见顶层字段：`text`、`reasoning`、`metadata`、`parallel_tool_calls`、`previous_response_id`、`service_tier`、`store`、`top_p`、`truncation`、`tool_choice` 等。

Anthropic Messages 常见顶层字段：`thinking`、`metadata`、`stop_sequences`、`top_k`、`top_p`、`service_tier`、`tool_choice` 等。

## response_format

`maidock-openai-responses` 会把 MaiBot 的 response_format 转换为 Responses API 的 `text.format`：

- text/default：不传 `text.format`。
- `json_object`：`{"type": "json_object"}`。
- `json_schema`：`{"type": "json_schema", "name": ..., "schema": ...}`。

## Tool calling

MaiDock 会把 Host 的 tool definitions 转换为上游 function/tool 声明，并把上游返回的工具调用转换回 Host 可解析的：

```python
{
    "id": "call-id",
    "function": {
        "name": "tool_name",
        "arguments": {}
    },
    "extra_content": {}
}
```

工具参数解析模式由 `[compatibility].tool_argument_parse_mode` 控制：`auto`、`strict`、`repair`、`double_decode`。空参数字符串会被视为 `{}`。

## OpenAI Responses 历史与工具调用

`maidock-openai-responses` 默认使用无状态完整历史回放，不伪造 OpenAI Responses 的服务端 item ID：

- 普通 assistant 文本历史使用 Responses `EasyInputMessage` 形状：`{"role": "assistant", "content": "..."}`。
- 只有 OpenAI 服务端真实返回并被 Host 保存的 output item 才能使用 `msg...` / `fc...` 这类 item ID；MaiDock 不会本地生成 `msg...` 或 `oai_hist...`。
- 如果需要使用服务端状态链路，可在 `extra_params` 中显式传入 `previous_response_id`；MaiDock 不会从 `raw_data` 自动推断，避免与完整历史回放产生重复上下文。
- Responses `function_call.call_id` 是 Host 的 canonical tool call ID，后续 `function_call_output.call_id` 只使用这个值。
- Responses function call item 的 `id` 会保存在 `tool_calls[].extra_content.openai_responses.item_id`，不会替代 `call_id`。
- 如果历史被截断导致 tool result 找不到前置 assistant `function_call`，MaiDock 会把该工具结果降级为普通 user 文本摘要，而不是发送非法的孤儿 `function_call_output`。

## Reasoning / Thinking

- OpenAI Responses：读取 `reasoning`/`reasoning_summary` 类输出摘要，并支持从文本 `<think>...</think>` 中分离 reasoning。
- Anthropic Messages：读取 `thinking` / `redacted_thinking` block。
- `[compatibility].reasoning_parse_mode` 支持：`auto`、`native`、`think_tag`、`none`。
