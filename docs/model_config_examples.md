# model_config.toml 源文件编辑示例

> MaiDock 1.2.0 起完全忽略 `[[models]].extra_params`（模型配置与单次请求的 `extra_params` 均无效）。模型在 MaiDock 配置页中各 Provider 能力段的**参数覆写目录**中按模型需要统一配置；同一个模型供应商下的所有模型共享该覆写目录。以下示例只保留供应商端点与模型标识的写法。

## OpenAI Responses

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

[[models]]
name = "Text-Embedding-3-Small"
model_identifier = "text-embedding-3-small"
api_provider = "openai"

[[models]]
name = "GPT-4o-Mini-Transcribe"
model_identifier = "gpt-4o-mini-transcribe"
api_provider = "openai"
```

## Anthropic Messages

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
```

## 阿里云百炼 DashScope

```toml
[[api_providers]]
name = "dashscope"
client_type = "maidock-dashscope"
api_key = "sk-..."

[[models]]
name = "Qwen3.7-Plus"
model_identifier = "qwen3.7-plus"
api_provider = "dashscope"

[[models]]
name = "Tongyi-Embedding-Vision"
model_identifier = "tongyi-embedding-vision-flash-2026-03-06"
api_provider = "dashscope"

[[models]]
name = "Qwen3.7-Text-Embedding"
model_identifier = "qwen3.7-text-embedding"
api_provider = "dashscope"

[[models]]
name = "Qwen3-ASR-Flash"
model_identifier = "qwen3-asr-flash"
api_provider = "dashscope"
```

## 阿里云百炼 Responses（maidock-bailian-responses）

百炼 Responses 使用 OpenAI Responses 规范，要求 base URL 以 `/v1` 结尾（MaiDock 自动追加 `/responses`）；北京、新加坡、美国、日本、德国均有官方 base URL。

```toml
[[api_providers]]
name = "bailian-responses"
client_type = "maidock-bailian-responses"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-..."
auth_type = "bearer"

[[models]]
name = "Qwen3.7-Plus (Responses)"
model_identifier = "qwen3.7-plus"
api_provider = "bailian-responses"
```

## SiliconFlow

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

[[models]]
name = "SenseVoice-Small"
model_identifier = "FunAudioLLM/SenseVoiceSmall"
api_provider = "siliconflow"
```

## Volcengine Ark

```toml
[[api_providers]]
name = "volcengine"
client_type = "maidock-volcengine-ark-responses"
api_key = "..."

[[models]]
name = "Doubao-Seed-2.0-Lite"
model_identifier = "doubao-seed-2-0-lite-260428"
api_provider = "volcengine"

[[models]]
name = "Doubao-Embedding-Vision"
model_identifier = "doubao-embedding-vision-251215"
api_provider = "volcengine"

[[models]]
name = "Doubao-Seed-2.0-Lite-ASR"
model_identifier = "doubao-seed-2-0-lite-260428"
api_provider = "volcengine"
```

## Xiaomi Mimo

```toml
[[api_providers]]
name = "mimo"
client_type = "maidock-xiaomi-mimo"
base_url = "https://api.xiaomimimo.com/v1"
api_key = "sk-..."
# Token Plan 请把 base_url 和 api_key 替换为控制台提供的集群地址与 tp-... 凭据。

[[models]]
name = "MIMO-2.5-PRO"
model_identifier = "mimo-v2.5-pro"
api_provider = "mimo"
# 思考默认由插件参数覆写目录中的 thinking（默认 {"type":"disabled"}）控制；
# 改为 enabled 或清空后，MaiDock 会自动回传工具调用历史中的 reasoning_content。

[[models]]
name = "MIMO-2.5-ASR"
model_identifier = "mimo-v2.5-asr"
api_provider = "mimo"
# Mimo ASR 复用 Chat Completions 文本生成端点；language 覆写会映射到 asr_options.language。
```
