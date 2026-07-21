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

### Volcengine Ark Audio Transcription

ARK 语音转录使用 Responses `input_audio.audio_url` + `input_text`，固定非流式且不进入自动前缀缓存。

- `max_tokens` / `max_output_tokens`：发送为 `max_output_tokens`。
- `prompt`：覆盖 Provider 基础设置中的转录提示词。
- `format` / `audio_format`：仅用于校验音频并构造 data URL，不会作为顶层字段发送。

支持 MP3、WAV、AAC、M4A，Base64 解码后的音频文件上限为 25 MiB。

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
- `enable_code_interpreter`
- `enable_search`
- `enable_thinking`
- `incremental_output`
- `max_completion_tokens`
- `max_tokens`
- `n`
- `parallel_tool_calls`
- `plugins`（特殊转为 `X-DashScope-Plugin` header）
- `presence_penalty`
- `repetition_penalty`
- `response_format`
- `reasoning_effort`
- `result_format`
- `search_options`
- `seed`
- `stop`
- `temperature`
- `thinking_budget`
- `tool_choice`
- `tool_stream`
- `tools`
- `top_k`
- `top_p`
- `vl_high_resolution_images`

特殊规则：

- `input`、`model`、`parameters`、`stream` 是 MaiDock 自己构造的保留字段，不会从 `extra_params` 透传。
- Host `max_tokens` 是通用的最大输出预算：对官方明确支持的 Qwen3.7-Max+、Qwen3.5-Plus+、Qwen3.5-Flash+、Kimi K2.5+、GLM 5+、MiniMax M2.5+ 与受支持 DeepSeek 系列，自动发送为 `parameters.max_completion_tokens`；未知模型、旧模型和第三方直供模型继续发送为 `parameters.max_tokens`。
- 显式 `max_completion_tokens` 优先于 Host 通用预算，此时不会同时发送 `max_tokens`；若用户同时显式指定两个原生字段则直接报冲突。`max_tokens`、`max_completion_tokens` 与 `thinking_budget` 均要求正整数。
- `reasoning_effort` 支持 `low`、`medium`、`high`、`xhigh`、`max`；`search_options` 必须是对象，新布尔字段严格要求布尔值。
- `tool_choice` 在参数策略和 `tools` 覆写完成后统一处理。思考模式只允许 `auto`/`none`；非思考模式的 `required` 仅在最终恰好一个工具时转换为指定函数对象，否则直接报错。
- `result_format` 默认设置为 `message`，用于控制阿里云百炼 DashScope API 返回 JSON 结构；Host `response_format` 会单独映射到 `parameters.response_format` 作为模型输出内容格式约束，目前仅确认并支持 `json_object`。
- 流式请求默认设置 `incremental_output = true` 和 `stream = true`，并发送 `Accept: text/event-stream`、`X-Accel-Buffering: no`、`X-DashScope-SSE: enable`。
- 原生工具流同时兼容增量块与累计块，工具调用没有 `index` 仍是合法响应；只有标识确实冲突或多个未决调用无法判定归属时才报错。
- 阿里云百炼 DashScope Embedding 使用同一套 `headers` / `query` / `body` 拆分规则，并支持顶层 `dimension`、`enable_fusion`、`text_type`、`output_type`、`instruct`、`fps`、`res_level`、`max_video_frames`、`auto_truncation` 写入 `parameters`。原生 API 不发送 OpenAI-compatible 的 `encoding_format`；`dimensions` 会映射为原生 singular `dimension`。文本模型支持 `text-embedding-v*` 与 `qwen3.7-text-embedding*`。

### DashScope Audio Transcription

- 支持 WAV、MP3、AAC、FLAC、OGG；不接受 M4A 或未知格式。
- `format` / `audio_format` 只作为本地 Data URL 格式提示，经过配置优先级和文件签名冲突检查后会从请求体移除，不发送到 DashScope `parameters`。
- Base64 必须严格合法，编码后长度上限为 10 MiB。未知签名、显式格式与签名冲突、格式不支持或超限都会直接报错，不会默认按 WAV 发送。
- 原生 ASR usage 不能准确构造 Host 总 Token，因此 Host usage 保持零；开启原始响应后仍会保留上游 usage。

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
- Xiaomi Mimo 会把 Host `max_tokens`、顶层 `max_completion_tokens` 和旧的 `body.max_tokens` 统一发送为官方 `max_completion_tokens`；多个来源值不一致时直接报错。
- Xiaomi Mimo 默认由 `[xiaomi_mimo].force_disable_thinking = true` 强制关闭思考。关闭后，带工具调用轮次的 `reasoning_content` 会通过 `extra_content` 和 SQLite 完整回传。
- SiliconFlow 和 Mimo 的工具调用、多模态图片均通过各自 Provider 入口委托 Chat Completions family 标准实现。

## Xiaomi Mimo Audio Transcription

Mimo 语音转录按模型使用两种 Chat Completions 请求结构：

- `mimo-v2.5-asr`：专用单音频 ASR，仅发送 `input_audio`，支持 MP3/WAV 和 `asr_options.language`。
- 其他模型：保持通用音频理解结构，发送 `input_audio` + 文本提示词；官方仅确认 `mimo-v2.5` 支持该能力。

支持的 `extra_params` 顶层字段：

- `prompt`：覆盖通用音频理解路径的文本提示词；专用 ASR 不发送。
- `language`：专用 ASR 识别语言，可选 `auto`、`zh`、`en`。
- `max_tokens` / `max_completion_tokens`：仅通用路径使用，发送为 `max_completion_tokens`。

特殊规则：

- `format` / `audio_format` 只用于校验文件签名并构造 MIME data URL，不会作为顶层 body 字段透传；冲突或未知格式会直接报错。
- `model`、`messages`、`stream` 是 MaiDock 自己构造的保留字段。
- 专用 ASR 的 Base64 字符串上限为 10 MiB；通用路径上限为 50 MiB，并额外支持 FLAC、M4A、OGG。
