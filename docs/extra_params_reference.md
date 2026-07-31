# 参数覆写参考

MaiDock 1.2.0 起不再读取模型配置或单次请求的 `extra_params`（`model_config.toml` 中的 `[[models]].extra_params` 与 Host 请求注入的 `extra_params` 都会被完全忽略）。所有额外参数只能通过插件 WebUI / `config.toml` 中每个 Provider 能力段的 **参数覆写目录** 添加。

本页对字段用途和值域的说明会参考本地供应商资料与对应 SDK；MaiDock 实际接受的字段、目标路径和转译行为只以当前 `src/core/parameter_catalog.py`、参数管线及各 Provider `parameter_translation.py` 的代码链路为准。

## 优先级与合并规则

```text
Core 已解析的请求级类型字段 -> MaiDock 供应商参数转译 -> MaiDock 非空覆写值 -> Provider 请求体
```

- 只有 Core 请求级类型字段（文本请求的 `temperature`、`max_tokens`、`response_format`，Embedding 的 `dimensions`，语音请求的 `max_tokens`）会进入参数管线；请求字段为 `None` 时不注入，也不再回退读取 `model_info` 的对应字段。
- **MaiDock 非空覆写值拥有最终优先级**：覆写 `temperature = "0.9"` 会覆盖 Host 请求的 `temperature`。
- 同一对象路径采用叶级合并：例如 Host `response_format` 转译为 `text.format`，而覆写 `text` 对象时只替换同名叶子（如 `text.format`），保留其他已转译字段。
- 两级 `extra_params`（模型级、请求级）即使包含冲突或非法值也完全无效，不会报错也不会进入请求体。
- `input`、`model`、`stream` 和 Host Function 工具由 Adapter 生成，不可覆写。

## 覆写值语法

所有覆写值都是字符串，写在 `config.toml` 对应能力段的 `[provider.capability.overrides]` 下：

```toml
[dashscope.chat_completion.overrides]
temperature = "0.4"              # 数字使用 JSON 数字
enable_search = "true"           # 布尔使用 true/false
stop = '["end1","end2"]'         # 数组使用合法 JSON
search_options = '{"forced_search":true}'  # 对象使用合法 JSON
```

- **空白表示不覆写**（`""` 或键缺失）。
- 字符串类型字段直接写文本，不需要 JSON 引号。
- 覆写值会在启动时按目录声明的类型解析；类型错误会直接报错，不会静默忽略。
- WebUI 中每个参数只有一个跨双列全宽的 textarea，参数名后标注值类型；稳定默认值已作为可编辑默认文本写入对应覆写框（如 DashScope `result_format=message`、SiliconFlow/Ark Embedding `encoding_format=float`、Mimo ASR `language=auto`、Mimo `thinking={"type":"disabled"}`、百炼 `store=false`）。

## OpenAI Responses / Volcengine Ark Responses

覆写目录字段（发送为 Responses body 字段）：

| 覆写键 | 目标路径 | 类型 | 说明 |
| --- | --- | --- | --- |
| `temperature` | `body.temperature` | number | 采样温度 |
| `max_tokens` | `body.max_output_tokens` | integer | 输出预算（覆写 Host `max_tokens`） |
| `response_format` | `body.text.format` | json | 结构化输出（Host response_format 也转译到此） |
| `top_p` | `body.top_p` | number | 核采样 |
| `reasoning` | `body.reasoning` | json | 推理配置 |
| `thinking` | `body.thinking` | json | 思考配置 |
| `text` | `body.text` | json | 文本对象（与 response_format 叶级合并） |
| `tool_choice` | `body.tool_choice` | json | 工具选择 |
| `parallel_tool_calls` | `body.parallel_tool_calls` | boolean | 并行工具调用 |
| `max_tool_calls` | `body.max_tool_calls` | integer | 单轮最大工具调用 |
| `include` | `body.include` | string_list | 附加输出类型 |
| `instructions` | `body.instructions` | string | 指令 |
| `metadata` | `body.metadata` | json | 元数据 |
| `store` | `body.store` | boolean | 是否存储响应 |
| `truncation` | `body.truncation` | string | 截断策略 |
| `service_tier` | `body.service_tier` | string | 服务等级 |
| `previous_response_id` | `body.previous_response_id` | string | 续接响应 |
| `user` | `body.user` | string | 用户标识 |
| `session` | `body.session` | json | 会话状态 |
| `caching` | `body.caching` | json | 显式缓存 |
| `expire_at` | `body.expire_at` | integer | 缓存过期时间 |
| `tools` | `body.tools` | json | 附加协议原生工具（如 `web_search`、`mcp`）；与 Host 工具合并发送，并按工具类型自动补 `ark-beta-*` header |

启用 ARK 自动前缀缓存后，显式 `caching` 或 `previous_response_id` 覆写仍优先，MaiDock 不会覆盖手动缓存链。

### OpenAI Embedding / Audio Transcription

Embedding 覆写字段：`dimensions`、`encoding_format`（默认 `float`）、`user`。

Audio Transcription 覆写字段：`language`、`prompt`、`response_format`、`temperature`、`timestamp_granularities`、`chunking_strategy`、`include`、`stream`（仅作为 multipart 字段发送，Provider 不消费转录 SSE，通常保持 `false`）。

### Volcengine Ark Embedding / Audio Transcription

Embedding 覆写字段：`dimensions`、`sparse_embedding`（布尔会转成官方 `{"type":"enabled|disabled"}` 对象）、`encoding_format`（默认 `float`）。

Audio Transcription 覆写字段：`max_tokens`（发送为 `max_output_tokens`）、`prompt`（默认提示词已作为可编辑默认文本）、`format` / `audio_format`（仅用于校验音频并构造 data URL）。

## 阿里云百炼 Responses（maidock-bailian-responses）

百炼 Responses 只开放官方 Responses 规范中的稳定覆写参数：`temperature`、`max_tokens`、`response_format`、`top_p`、`reasoning`、`text`、`tool_choice`、`parallel_tool_calls`、`max_tool_calls`、`include`、`instructions`、`metadata`、`store`、`truncation`、`service_tier`。

- `max_tokens` 按 OpenAI Responses 规范转译为 `max_output_tokens`；`response_format` 转译为 `text.format`。
- `store` 默认 `false`（可编辑默认文本）：无状态 Host 链路不产生远端会话存储，用户可显式改为 `true`。
- Host base URL 必须以 `/v1` 结尾（例如 `https://dashscope.aliyuncs.com/compatible-mode/v1`），MaiDock 自动追加 `/responses`；填写完整 `/responses` 端点、DashScope 原生 `/api/v1` 地址或其他形式时明确报错。
- 官方 base URL：北京 `https://dashscope.aliyuncs.com/compatible-mode/v1`、新加坡 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`、美国 `https://dashscope-us.aliyuncs.com/compatible-mode/v1`、日本 `https://dashscope-jp.aliyuncs.com/compatible-mode/v1`、德国 `https://dashscope-eu.aliyuncs.com/compatible-mode/v1`。
- 服务端原生工具、OCR/input_file、Responses CRUD 和应用 Responses API 本轮不开放；Function 工具按百炼文档省略未声明的 `strict` 字段。

## Anthropic Messages

覆写目录字段（发送为 Anthropic Messages body 字段）：`temperature`、`max_tokens`、`top_p`、`top_k`、`thinking`、`tool_choice`、`stop_sequences`、`metadata`、`service_tier`。

- `messages`、`model`、`stream`、`system`、`tools` 是 MaiDock 自己构造的保留字段，不可覆写。
- `tool_choice` 可以覆盖 MaiDock 自动生成的默认工具选择策略。
- Anthropic 继续拒绝 Core 类型化的 `response_format`；`extra_params` 中的同名值已被完全忽略，不再需要检查。

## 阿里云百炼 DashScope

文本生成覆写目录字段（发送为 DashScope `parameters`，`customized_model_id` 特殊放入 `input.customized_model_id`，`plugins` 特殊转为 `X-DashScope-Plugin` header）：

| 覆写键 | 目标路径 | 类型 | 默认 |
| --- | --- | --- | --- |
| `temperature` | `body.parameters.temperature` | number | |
| `max_tokens` | `body.parameters.max_tokens` | integer | |
| `max_completion_tokens` | `body.parameters.max_completion_tokens` | integer | |
| `thinking_budget` | `body.parameters.thinking_budget` | integer | |
| `reasoning_effort` | `body.parameters.reasoning_effort` | string | |
| `response_format` | `body.parameters.response_format` | json | |
| `result_format` | `body.parameters.result_format` | string | `message` |
| `top_p` / `top_k` | `body.parameters.top_p` / `top_k` | number / integer | |
| `enable_thinking` / `enable_search` | `body.parameters.*` | boolean | |
| `search_options` | `body.parameters.search_options` | json | |
| `incremental_output` / `stream` / `tool_stream` / `parallel_tool_calls` / `enable_code_interpreter` / `vl_high_resolution_images` | `body.parameters.*` | boolean | |
| `seed` / `n` | `body.parameters.*` | integer | |
| `stop` | `body.parameters.stop` | json | |
| `presence_penalty` / `repetition_penalty` | `body.parameters.*` | number | |
| `tool_choice` / `tools` | `body.parameters.*` | json | |
| `plugins` | `headers.X-DashScope-Plugin` | json | |
| `customized_model_id` | `body.input.customized_model_id` | string | |

- `input`、`model`、`parameters` 是保留结构；`stream` 是否走 SSE 仍以 Host `force_stream_mode` 为准，两者不得配置成相反值。
- Host `max_tokens` 是通用最大输出预算：对官方明确支持的 Qwen3.7-Max+、Qwen3.5-Plus+、Qwen3.5-Flash+、Kimi K2.5+、GLM 5+、MiniMax M2.5+ 与受支持 DeepSeek 系列，自动发送为 `parameters.max_completion_tokens`；未知模型、旧模型和第三方直供模型继续发送为 `parameters.max_tokens`。
- 显式覆写 `max_tokens` 视为用户指定原生旧字段：此时保留 `max_tokens`，不再自动转译为 `max_completion_tokens`；若同时显式覆写 `max_completion_tokens` 则直接报冲突。
- `reasoning_effort` 支持 `low`、`medium`、`high`、`xhigh`、`max`；`search_options` 必须是对象，布尔字段严格要求布尔值。
- `tool_choice` 在覆写与 `tools` 合并后统一处理。思考模式只允许 `auto`/`none`；非思考模式的 `required` 仅在最终恰好一个工具时转换为指定函数对象，否则直接报错。
- 流式请求默认设置 `incremental_output = true` 和 `stream = true`（Host 未覆写时）。

### DashScope Embedding / Audio Transcription

Embedding 覆写字段：`dimensions`（映射为原生 `parameters.dimension`）、`output_type`、`instruct`、`text_type`、`auto_truncation`、`enable_fusion`（仅 `qwen3-vl-embedding` 默认启用，覆写优先）、`fps`、`max_video_frames`、`res_level`。

Audio Transcription 覆写字段：`language`（发送为 `parameters.asr_options.language`）、`enable_itn`（发送为 `parameters.asr_options.enable_itn`）、`format` / `audio_format`（仅作为本地音频格式提示，不发送给上游）。

## SiliconFlow / Xiaomi Mimo Chat Completions

覆写目录字段（发送为 Chat Completions body 字段）：`temperature`、`max_tokens`、`response_format`、`top_p`、`tool_choice`、`tools`、`frequency_penalty`、`presence_penalty`、`seed`、`stop`、`n`。

Mimo Chat 额外支持：

- `thinking`：发送为官方 `thinking` 对象，默认 `{"type":"disabled"}`（可编辑默认文本）。清空或改为 `{"type":"enabled"}` 后启用思考；启用时缺少状态存储会在请求阶段明确报错。
- Mimo 的 `max_tokens` 统一转译为官方 `max_completion_tokens`，并受 131072 上限约束。

### SiliconFlow Embedding / Audio Transcription

Embedding 覆写字段：`dimensions`、`encoding_format`（默认 `float`）。

Audio Transcription 覆写字段：`language`、`prompt`、`response_format`、`temperature`、`timestamp_granularities`、`chunking_strategy`、`include`、`stream`。

## Xiaomi Mimo Audio Transcription

Mimo 语音转录统一使用专用单音频 ASR 结构，并复用 Chat Completions 文本生成端点；官方当前仅支持 `mimo-v2.5-asr`。

覆写目录字段：

- `language`：发送为 `asr_options.language`，可选 `auto`（默认）、`zh`、`en`。
- `format` / `audio_format`：用于校验文件签名、构造 MIME data URL，并作为 `input_audio.format` 发送；不会作为顶层 body 字段发送。
