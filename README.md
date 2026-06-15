# MaiDock

MaiDock 是一个 MaiBot LLM Provider 插件，用于补充主程序未覆盖的端点，并且提供了高级参数控制功能。

**最低支持的 MaiBot 版本: 1.0.0-rc.3**

目前已实现：
- `maidock-openai-responses` — OpenAI Responses API
- `maidock-anthropic-messages` — Anthropic Messages API
- `maidock-dashscope` — 阿里云百炼 DashScope API
- `maidock-siliconflow` — 硅基流动 SiliconFlow API
- `maidock-volcengine-ark-responses` — 火山方舟 Volcengine Ark API
- `maidock-xiaomi-mimo` — 小米 Mimo API

---

## 能力矩阵

<table style="border-collapse: collapse; width: 100%;">
<thead>
<tr>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: left; font-weight: 600;">Provider</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">文本生成</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">Embedding</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">音频转录</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">流式输出</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">工具调用</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">多模态</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">推理/思考</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">响应格式</th>
</tr>
</thead>
<tbody>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-openai-responses</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-anthropic-messages</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-dashscope</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">⚠️</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-siliconflow</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-volcengine-ark-responses</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: left;"><code>maidock-xiaomi-mimo</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">⚠️</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">⚠️</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
</tbody>
</table>

> - 阿里云百炼 DashScope 不支持 `json_schema`。
> - 小米 Mimo 在开启深度思考且历史会话存在工具调用时要求回传思考内容，本插件无法满足该协议要求，因此默认强制关闭思考。
> - 小米 Mimo 无独立音频转录 API，实际通过文本生成端点 + `input_audio` 实现转录。可在插件管理 → MaiDock 中修改默认转录提示词。

---

## 配置

### 插件配置

**插件管理 → MaiDock** 中直接修改。如需编辑源文件，参考 **[插件配置参考](docs/plugin_config_reference.md)**。

### 模型供应商配置

**模型管理 → 添加厂商** 中选择对应的客户端类型。如需编辑源文件，参考 **[model_config.toml 编辑示例](docs/model_config_examples.md)**。

### 模型额外参数

**编辑模型 → 额外参数** 中填入。如需编辑源文件，参考 **[extra_params 参考文档](docs/extra_params_reference.md)**。

---

## 注意事项

### 图片，多模态相关 / 帧大小超过限制

报错信息为：**"插件 LLM Provider RPC 调用失败: [E_UNKNOWN] 帧大小 xxx 超过最大限制 16777216"**

当前由于传输层有 16 MB 单帧限制。如果发送大图，图片 base64 可能在到达本插件前就让 RPC 帧超过 16 MB。

WebUI 可视化界面需要打开高级设置才能更改 `多模态最大图片数`

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

### 端点与默认行为

在 WebUI 的**模型管理 → 添加厂商**界面中选择对应的客户端类型即可使用。

<table style="border-collapse: collapse; width: 100%;">
<thead>
<tr>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: left; font-weight: 600;">Provider</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: left; font-weight: 600;">默认 Base URL</th>
  <th style="border: 1px solid #30363d; padding: 8px 12px; text-align: center; font-weight: 600;">强制官方端点</th>
</tr>
</thead>
<tbody>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-openai-responses</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">无</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-anthropic-messages</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">无</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-dashscope</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>https://dashscope.aliyuncs.com/api/v1</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-siliconflow</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>https://api.siliconflow.cn/v1</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-volcengine-ark-responses</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>https://ark.cn-beijing.volces.com/api/v3</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">✅</td>
</tr>
<tr>
  <td style="border: 1px solid #30363d; padding: 8px 12px;"><code>maidock-xiaomi-mimo</code></td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">无</td>
  <td style="border: 1px solid #30363d; padding: 8px 12px; text-align: center;">❌</td>
</tr>
</tbody>
</table>

> - 阿里云百炼 DashScope、SiliconFlow、Volcengine Ark 默认使用官方端点，如需自定义地址可在 MaiDock 配置页面中关闭对应开关。
> - Xiaomi Mimo 无默认端点，始终使用 Host 提供的 base_url。Mimo 官方有按量付费与 Token Plan 两套地址，由 Host 侧按需配置。
