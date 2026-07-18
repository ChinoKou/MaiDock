# MaiDock

MaiDock 是一个 MaiBot LLM Provider 插件，用于补充主程序未覆盖的端点，并且提供了高级参数控制功能。

**最低支持的 MaiBot 版本: 1.0.9**

目前已实现：
- `maidock-openai-responses` — OpenAI Responses API
- `maidock-anthropic-messages` — Anthropic Messages API
- `maidock-dashscope` — 阿里云百炼 DashScope API
- `maidock-siliconflow` — 硅基流动 SiliconFlow API
- `maidock-volcengine-ark-responses` — 火山方舟 Volcengine Ark API
- `maidock-xiaomi-mimo` — 小米 Mimo API

---

## 能力矩阵

| Provider | 文本生成 | Embedding | 音频转录 | 流式输出 | 工具调用 | 多模态 | 推理/思考 | 响应格式 |
|---|---|---|---|---|---|---|---|---|
| `maidock-openai-responses` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maidock-anthropic-messages` | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `maidock-dashscope` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| `maidock-siliconflow` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maidock-volcengine-ark-responses` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `maidock-xiaomi-mimo` | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> - 阿里云百炼 DashScope 不支持 `json_schema`。
> - 小米 Mimo 仍默认强制关闭思考；关闭该开关后，MaiDock 会通过工具调用 `extra_content` 与 SQLite 恢复并回传历史 `reasoning_content`。
> - `mimo-v2.5-asr` 使用专用单音频 ASR 协议；其他模型保持“音频 + 自定义提示词”的通用音频理解路径，官方仅确认 `mimo-v2.5` 支持该能力。

---

## 配置

### 插件配置

**插件管理 → MaiDock** 中直接修改。如需编辑源文件，参考 **[插件配置参考](docs/plugin_config_reference.md)**。

### 模型供应商配置

**模型管理 → 添加厂商** 中选择对应的客户端类型。如需编辑源文件，参考 **[model_config.toml 编辑示例](docs/model_config_examples.md)**。

### 模型额外参数

**编辑模型 → 额外参数** 中填入。如需编辑源文件，参考 **[extra_params 参考文档](docs/extra_params_reference.md)**。

---

### 火山方舟 Responses 前缀缓存

在插件配置的 **Volcengine Ark 基础设置** 中可开启显式前缀缓存。该功能默认关闭；开启后，MaiDock 会：
该功能需要 MaiBot Core 1.0.9 或更高版本，并需先在火山方舟开通管理中开启“推理（缓存）”计价。


- 使用 ARK 分词 API 确认开头的 system 前缀不少于 256 tokens。
- 将固定 system 前缀与 function tools 创建为非流式缓存，真实请求只发送剩余历史。
- 按模型、账号、前缀、thinking 和 tools 分别保存缓存 ID，默认有效期为 3 天。

ARK 会收取缓存存储费用和缓存命中输入费用。`instructions`、`json_schema`、`store=false`、非 function 工具不参与自动缓存；模型额外参数中显式设置 `caching` 或 `previous_response_id` 时，以手动参数为准。

缓存索引保存在 MaiBot Core 分配的 `data/plugins/chinokou.maidock/maidock_state.sqlite3` 中，不会写入插件源码目录。

### 语音转录与 Mimo 思考回传

- ARK 通过 Responses `input_audio.audio_url` + `input_text` 转录音频，提示词可在 Provider 基础设置中修改。
- Mimo 专用 ASR 支持 MP3/WAV 和 `auto/zh/en` 语言选项；通用音频理解支持 MP3/WAV/FLAC/M4A/OGG。
- MaiDock 会校验 Base64、文件签名、显式格式和提供商大小限制，未知格式不会再默认按 WAV 发送。
- Mimo 仅保存带工具调用轮次的完整思考内容。内容以明文保存在插件数据目录的 SQLite 中，默认保留 30 天并在使用时续期。

---

## 注意事项

### Embedding / 嵌入维度

示例报错信息为：**"embedding 真实输出维度与当前向量存储不一致: expected=存储的维度值, encoded=xxx"**

当前由于长期记忆模块未对插件注册的客户端适配，即不会向第三方客户端注入维度参数，会让模型以默认维度输出向量。

建议操作:

- 强制覆写维度参数

插件配置 → 选择对应的提供商 → **Embeddings 字段开关与覆写** → **开启`覆写「dimension(s)」`** → 填写覆写值

插件配置文件 `config.toml` / 源代码编辑模式 请查阅上方[插件配置参考](#插件配置)


### 图片，多模态相关 / 帧大小超过限制

报错信息为：**"插件 LLM Provider RPC 调用失败: [E_UNKNOWN] 帧大小 xxx 超过最大限制 16777216"**

当前由于传输层有 16 MB 单帧限制。如果发送大图，图片 base64 可能在到达本插件前就让 RPC 帧超过 16 MB。

建议操作:

- 修改 `多模态最大图片数` (高级设置)
- 修改 `最大图片大小`

麦麦设置 → 视觉 → 打开高级设置 → 修改

配置文件 `bot_config.toml` / 源代码编辑模式 关键字段：

```toml
[visual]
max_image_num = 1 # 建议为 1, 具体视上下文长度与单图片大小而定
max_image_size_mb = 5 # 视情况而定
```

### 超时

报错信息为：**"插件 LLM Provider RPC 调用失败: [E_TIMEOUT] 请求 plugin.invoke_llm_provider 超时 (30000ms)"**

建议操作：

- 变更默认超时(30s)设置
- 更换响应更快的模型、提供商

模型管理 → 模型厂商设置 → 编辑提供商 → 修改超时

模型配置文件 `model_config.toml` 关键字段：

```toml
[[api_providers]]
timeout = 30 # 默认值为 30
```

### 端点与默认行为

在 WebUI 的**模型管理 → 添加厂商**界面中选择对应的客户端类型即可使用。

| Provider | 默认 Base URL | 强制官方端点 |
|---|---|---|
| `maidock-openai-responses` | 无 | ❌ |
| `maidock-anthropic-messages` | 无 | ❌ |
| `maidock-dashscope` | `https://dashscope.aliyuncs.com/api/v1` | ✅ |
| `maidock-siliconflow` | `https://api.siliconflow.cn/v1` | ✅ |
| `maidock-volcengine-ark-responses` | `https://ark.cn-beijing.volces.com/api/v3` | ✅ |
| `maidock-xiaomi-mimo` | 无 | ❌ |

> - 阿里云百炼 DashScope、SiliconFlow、Volcengine Ark 默认使用官方端点，如需自定义地址可在 MaiDock 配置页面中关闭对应开关。
> - Xiaomi Mimo 无默认端点，始终使用 Host 提供的 base_url。Mimo 官方有按量付费与 Token Plan 两套地址，由 Host 侧按需配置。
