# 插件配置参考

插件配置文件为 MaiDock 目录下的 `config.toml`。模型、API Provider 与单次请求参数仍在 MaiBot 的 `model_config.toml` 中配置。

> 💡 大部分配置项可通过 MaiBot WebUI 的**"插件配置"标签页**直接修改。此文档面向需要手动编辑 TOML 的场景。

---

## `[plugin]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | 是否启用 MaiDock 插件 |
| `config_version` | str | `"1.0.6"` | 配置版本标记，一般不需要手动修改 |

---

## `[diagnostics]`

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `include_raw_data` | bool | `false` | 是否把脱敏后的上游响应摘要放入 Host `raw_data` |
| `log_payload_summary` | bool | `true` | 是否记录脱敏后的请求/响应摘要日志 |
| `log_payload_debug` | bool | `false` | 是否记录脱敏后的详细请求载荷 |

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

## Provider 配置

每个 Provider 有两层配置：

1. **Provider 级**字段：`user_agent` + 重试配置（所有 Provider），部分含 `force_official_endpoint`；Mimo 额外含 `force_disable_thinking`、`audio_transcription_prompt`
2. **能力子段**：`[{provider}.{capability}]` — 控制该 Provider 某项能力的 `extra_params` 参数策略
3. **字段开关子段**：`[{provider}.{capability}.fields]` — 由 WebUI 自动生成，控制单个 `extra_params` 字段的启用/禁用/覆写
4. **默认参数 / 覆写参数**：`[{provider}.{capability}.default_params]` 和 `[{provider}.{capability}.override_params]` — 空的 inline table，可手动填入 JSON

能力子段和能力参数策略的完整字段说明见 [能力参数策略](#能力参数策略-capabilityparameterpolicyconfig)。

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
| `[openai_responses.image_generation]` | 图像生成（**占位**，目前无 Provider 实际实现此能力） |

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
| `[anthropic_messages.image_generation]` | 图像生成（**占位**） |

### `[dashscope]`

```toml
[dashscope]
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
| `force_official_endpoint` | bool | `true` | 是否忽略 Host 提供的 `base_url`，强制使用阿里云百炼 DashScope 官方 endpoint |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[dashscope.chat_completion]` | 文本生成（Generation API） |
| `[dashscope.embeddings]` | Embeddings（文本 / 多模态） |
| `[dashscope.audio_transcription]` | 语音转录（多模态生成端点） |
| `[dashscope.image_generation]` | 图像生成（**占位**） |

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
| `[siliconflow.image_generation]` | 图像生成（**占位**） |

### `[volcengine_ark]`

```toml
[volcengine_ark]
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
| `force_official_endpoint` | bool | `true` | 是否忽略 Host 提供的 `base_url`，强制使用火山方舟官方 endpoint |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[volcengine_ark.response]` | 文本生成（Responses API） |
| `[volcengine_ark.embeddings]` | Embeddings |
| `[volcengine_ark.image_generation]` | 图像生成（**占位**） |

### `[xiaomi_mimo]`

```toml
[xiaomi_mimo]
user_agent = ""
force_disable_thinking = true
audio_transcription_prompt = "请转写这段音频"
max_retries = 3
force_max_retries = false
retry_interval = 5.0
force_retry_interval = false
```

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `user_agent` | str | `""` | 自定义 User-Agent |
| `force_disable_thinking` | bool | `true` | 是否在最终请求体中强制写入 `thinking = { type = "disabled" }`。Mimo 要求 thinking + 工具调用历史必须回传思考内容，但 Host 不会向 MaiDock 提供历史 reasoning_content，因此默认关闭 |
| `audio_transcription_prompt` | str | `"请转写这段音频"` | Mimo 伪语音转录请求中与 input_audio 一同发送的文本提示词。未配置时转录请求会报错 |
| `max_retries` | int | `3` | 最大重试次数。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_max_retries` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |
| `retry_interval` | float | `5.0` | 重试间隔（秒）。关闭下方开关时为回退值，开启时强制覆写 Host 值 |
| `force_retry_interval` | bool | `false` | 关闭=回退模式，开启=强制使用上方的值 |

| 子段 | 说明 |
| --- | --- |
| `[xiaomi_mimo.chat_completion]` | 文本生成（Chat Completions API） |
| `[xiaomi_mimo.audio_transcription]` | 语音转录伪造层（策略控制），实际使用 Chat Completions + `input_audio` |

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

## 能力参数策略 (CapabilityParameterPolicyConfig)

每个 `[{provider}.{capability}]` 子段包含以下 **5 个显式字段 + 3 个子段**：

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `accept_model_extra_params` | bool | `true` | 是否接受模型级 `extra_params` |
| `accept_request_extra_params` | bool | `true` | 是否接受请求级 `extra_params` |
| `unknown_extra_params` | select | `"forward"` | 未识别顶层字段的处理策略：`"forward"`（放入 extra_body 传给上游）、`"drop"`（静默丢弃）、`"reject"`（直接报错） |
| `disabled_paths` | list[str] | `[]` | 需要静默移除的参数路径列表，如 `["body.temperature", "headers.X-Test"]` |
| `rejected_paths` | list[str] | `[]` | 出现时直接拒绝请求的参数路径列表，如 `["headers.Authorization"]` |

以及三个子段：

| 子段 | 说明 |
| --- | --- |
| `[{provider}.{capability}.fields]` | 逐字段启用/禁用与强制覆写开关（WebUI 自动生成） |
| `[{provider}.{capability}.default_params]` | 低优先级默认参数（inline table，可填入任意 JSON） |
| `[{provider}.{capability}.override_params]` | 最高优先级强制覆写参数（inline table，支持 `body`/`headers`/`query` 子对象） |

### fields 子段

`fields` 中的每个键由 `{config_key}_{作用}` 组成。`config_key` 来自字段在 Provider HTTP API 中的目标路径，由 `safe_parameter_key()` 转为小写蛇形命名：

| 目标路径 | config_key |
|----------|------------|
| `body.temperature` | `body_temperature` |
| `body.parameters.enable_thinking` | `body_parameters_enable_thinking` |
| `headers.X-DashScope-Plugin` | `headers_x_dashscope_plugin` |

每个字段产生三个配置键：

```toml
body_temperature_enabled = true          # 是否允许该字段通过
body_temperature_override_enabled = false  # 是否启用强制覆写
body_temperature_override_value = ""       # 覆写目标值
```

- `_enabled`：bool，默认 `true`。设为 `false` 后该字段被静默丢弃
- `_override_enabled`：bool，默认 `false`。设为 `true` 后启用强制覆写
- `_override_value`：布尔字段默认 `false`，其他字段默认 `""`。当 `_override_enabled = true` 时，此值强制覆盖所有上游传入的该字段

**示例：强制所有 OpenAI Responses 请求使用 `temperature = 0.7`**

```toml
[openai_responses.response.fields]
body_temperature_override_enabled = true
body_temperature_override_value = "0.7"
```

> 💡 `fields` / `default_params` / `override_params` 由 WebUI 自动管理，手动编辑 TOML 时通常无需自行填写。

### default_params 与 override_params

```toml
[openai_responses.response.default_params]

[openai_responses.response.override_params]
```

两者均为空的 inline table，可填入任意 JSON。差异在于优先级：

| 层级 | 优先级 | 说明 |
|------|--------|------|
| `default_params` | 最低 | 会被模型级和请求级 `extra_params` 覆盖 |
| 模型级 `extra_params` | 中 | `model_config.toml` 中 `[[models]]` 的 `extra_params` |
| 请求级 `extra_params` | 高 | Host 单次调用传入的 `extra_params`，覆盖模型级同名字段 |
| `override_params` | 最高 | 不受其他任何配置影响，强制写入最终请求 |

`override_params` 支持 `body` / `headers` / `query` 子对象：

```toml
[volcengine_ark.response.override_params]
body = { temperature = 0.3 }
headers = { "X-Custom" = "value" }
```

### 完整示例：OpenAI Responses 能力策略

```toml
[openai_responses.response]
accept_model_extra_params = true
accept_request_extra_params = true
disabled_paths = []
rejected_paths = []
unknown_extra_params = "forward"

[openai_responses.response.fields]
body_temperature_enabled = true
body_temperature_override_enabled = false
body_temperature_override_value = ""
body_top_p_enabled = true
body_top_p_override_enabled = false
body_top_p_override_value = ""
# ... 更多字段由 WebUI 自动生成

[openai_responses.response.default_params]

[openai_responses.response.override_params]
```
