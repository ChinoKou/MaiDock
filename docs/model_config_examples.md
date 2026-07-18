# model_config.toml 源文件编辑示例

### OpenAI Responses

```toml
[[api_providers]]
name = "openai"
client_type = "maidock-openai-responses"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
auth_type = "bearer"

[[models]]
name = "GPT-5.5"
model_identifier = "gpt-5.5"
api_provider = "openai"
extra_params = { reasoning = { effort = "medium" } }
```

### Anthropic Messages

```toml
[[api_providers]]
name = "anthropic"
client_type = "maidock-anthropic-messages"
base_url = "https://api.anthropic.com"
api_key = "sk-ant-..."
auth_type = "header"
auth_header_name = "x-api-key"
# 如果配置为以下鉴权方式, MaiDock 内部自动转为 x-api-key 鉴权
# auth_type = "bearer"

[[models]]
name = "Claude Opus 4.8"
model_identifier = "claude-opus-4-8"
api_provider = "anthropic"
extra_params = { thinking = { type = "enabled", budget_tokens = 1024 } }
```

### 阿里云百炼 DashScope

```toml
[[api_providers]]
name = "dashscope"
client_type = "maidock-dashscope"
api_key = "sk-..."

[[models]]
name = "Qwen3.7-Plus"
model_identifier = "qwen3.7-plus"
api_provider = "dashscope"
extra_params = { enable_thinking = true, result_format = "message" }

[[models]]
name = "Tongyi-Embedding-Vision"
model_identifier = "tongyi-embedding-vision-flash-2026-03-06"
api_provider = "dashscope"
```

### SiliconFlow

```toml
[[api_providers]]
name = "siliconflow"
client_type = "maidock-siliconflow"
api_key = "sk-..."

[[models]]
name = "GLM-5.1"
model_identifier = "Pro/zai-org/GLM-5.1"
api_provider = "siliconflow"

[[models]]
name = "BGE-M3"
model_identifier = "BAAI/bge-m3"
api_provider = "siliconflow"
```

### Volcengine Ark

```toml
[[api_providers]]
name = "volcengine"
client_type = "maidock-volcengine-ark-responses"
api_key = "..."

[[models]]
name = "Doubao-Seed-2.0-Lite"
model_identifier = "doubao-seed-2-0-lite-260428"
api_provider = "volcengine"
extra_params = { reasoning = { effort = "medium" } }
```

### Xiaomi Mimo

```toml
[[api_providers]]
name = "mimo"
client_type = "maidock-xiaomi-mimo"
api_key = "sk-..."

[[models]]
name = "MIMO-2.5-PRO"
model_identifier = "mimo-v2.5-pro"
api_provider = "mimo"
# thinking 默认由插件配置"强制关闭 Mimo 深度思考"控制
# 关闭该开关后，MaiDock 会自动回传工具调用历史中的 reasoning_content。

[[models]]
name = "MIMO-2.5-ASR"
model_identifier = "mimo-v2.5-asr"
api_provider = "mimo"
extra_params = { language = "auto" }
# 专用 ASR 不发送文本 prompt；通用 mimo-v2.5 音频理解才使用 prompt。
```
