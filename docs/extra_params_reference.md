# extra_params 参考

`extra_params` 用来把模型级和请求级的额外参数传给上游 HTTP API。MaiDock 会先合并两处配置：

1. `model_config.toml` 中 `[[models]]` 的 `extra_params`。
2. Host 在单次调用里传入的 `extra_params`。

如果两边有同名字段，单次调用里的值优先生效；值为 `null` 的字段会被忽略。

## 通用规则

在 `model_config.toml` 的 `[[models]].extra_params` 中，MaiBot 对以下三个特殊键做 transport 层拆分：`headers` 作为请求头传入，`query` 追加到 URL 查询参数，`body` 合并到请求体。其余顶层键均作为请求体额外字段传入。

```toml
extra_params = {
  headers = { X-Trace-Id = "trace-1" },
  query = { debug = "true" },
  body = { custom_field = "value" }
}
```

以下列出 MaiDock 内置支持的 `extra_params` 顶层字段，按 Provider 分组。

## OpenAI Responses / Volcengine Ark Responses

响应请求中，以下 `extra_params` 顶层字段会作为 Responses body 字段传入：

- `caching`
- `expire_at`
- `include`
- `instructions`
- `max_output_tokens`
- `max_tool_calls`
- `metadata`
- `parallel_tool_calls`
- `previous_response_id`
- `reasoning`
- `session`
- `service_tier`
- `store`
- `text`
- `thinking`
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

- `input`、`model`、`stream`、`temperature` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- `tools` 可作为 Ark Responses 原生工具列表传入；MaiDock 会与 Host function tools 合并，并按工具类型自动补 `ark-beta-*` header（`web_search`、`mcp`、`knowledge_search`、`doubao_app`、`image_process`）。
- `max_output_tokens` 如果写在 `extra_params` 顶层，会覆盖 MaiBot 的 `max_tokens` / 模型 `max_tokens` 换算结果。
- `text` 会和 MaiBot 的 `response_format` 合并；如果两边同时设置了冲突的格式字段，会直接报错。
- Ark Embedding 使用同一套 `headers` / `query` / `body` 拆分规则，并额外支持顶层 `encoding_format`、`dimensions`、`sparse_embedding`；`encoding_format = "base64"` 会直接报错，因为 MaiBot Host 当前只接受 float 向量。
- 启用 ARK 自动前缀缓存后，显式 `caching` 或 `previous_response_id` 仍优先，MaiDock 不会覆盖手动缓存链。
- 自动缓存只处理至少 256 tokens 的开头 system 前缀；`instructions`、`json_schema`、`store=false` 或非 function tools 会让当前请求直接按普通 Responses 请求发送。

OpenAI Embeddings 和 Audio Transcriptions 也会使用同一套 `headers` / `query` / `body` 拆分规则；除这些分组外，其他顶层字段默认进入 `extra_body`。如需严格模式，在对应能力的 `[{provider}.{capability}]` 子段设置 `unknown_extra_params = "reject"`。

## Anthropic Messages

响应请求中，以下 `extra_params` 顶层字段会作为 Anthropic Messages HTTP body 字段传入：

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
- 其他未识别顶层字段默认进入 `extra_body`。如需严格模式，在 `[anthropic_messages.chat_completion]` 中设置 `unknown_extra_params = "reject"`。

## 阿里云百炼 DashScope

文本生成请求中，以下 `extra_params` 顶层字段会进入阿里云百炼 DashScope `parameters`：

- `customized_model_id`（特殊放入 `input.customized_model_id`）
- `enable_search`
- `enable_thinking`
- `incremental_output`
- `max_length`
- `max_tokens`
- `n`
- `parallel_tool_calls`
- `plugins`（特殊转为 `X-DashScope-Plugin` header）
- `presence_penalty`
- `repetition_penalty`
- `response_format`
- `result_format`
- `seed`
- `stop`
- `temperature`
- `tool_choice`
- `tools`
- `top_k`
- `top_p`

特殊规则：

- `input`、`model`、`parameters`、`stream` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- `result_format` 默认设置为 `message`，用于控制阿里云百炼 DashScope API 返回 JSON 结构；Host `response_format` 会单独映射到 `parameters.response_format` 作为模型输出内容格式约束，目前仅确认并支持 `json_object`。
- 流式请求默认设置 `incremental_output = true` 和 `stream = true`，并发送 `Accept: text/event-stream`、`X-Accel-Buffering: no`、`X-DashScope-SSE: enable`。
- 阿里云百炼 DashScope Embedding 使用同一套 `headers` / `query` / `body` 拆分规则，并额外支持顶层 `dimension`、`encoding_format`、`enable_fusion`、`text_type`、`output_type`、`instruct`、`fps`、`res_level`、`max_video_frames`、`auto_truncation` 写入 `parameters`。阿里云百炼 DashScope native embedding 使用 singular `dimension`，不会接受 OpenAI-compatible 的 `dimensions`。

## SiliconFlow / Xiaomi Mimo Chat Completions

文本生成请求中，以下 `extra_params` 顶层字段会作为 Chat Completions body 字段传入：

- `temperature`
- `max_tokens`
- `response_format`
- `top_p`
- `tool_choice`
- `tools`
- `frequency_penalty`
- `presence_penalty`
- `seed`
- `stop`
- `n`

特殊规则：

- `messages`、`model`、`stream` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- Xiaomi Mimo 的 `thinking` 不再作为文本生成能力字段暴露；默认由插件配置 `[xiaomi_mimo].force_disable_thinking = true` 在最终请求体中强制写入 `{ "type": "disabled" }`。如需自行实验原生 thinking，请先关闭该 Provider 级开关。
- SiliconFlow 和 Mimo 的工具调用、多模态图片均通过各自 Provider 入口委托 Chat Completions family 标准实现。

## Xiaomi Mimo Audio Transcription

Mimo 没有独立的语音转录 API。MaiDock 会把 Host 的音频转录请求转换为 Chat Completions + `input_audio` 请求。

支持的 `extra_params` 顶层字段：

- `prompt`：作为 text content part 与 `input_audio` 一同发送，默认 `"请转写这段音频"`。

特殊规则：

- `format` / `audio_format` 只用于推断 `data:audio/...;base64,...` 的 MIME 格式，不会作为顶层 body 字段透传。
- `model`、`messages`、`stream` 是 MaiDock 自己构造的保留字段。
