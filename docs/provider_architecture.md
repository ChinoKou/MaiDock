# Provider 分层架构

MaiDock 的 Provider 代码固定遵循以下依赖方向：

```text
具体 Provider -> 协议 Family -> common -> Core / schemas
```

## 各层职责

- 具体 Provider：端点、鉴权、模型限制、默认值和供应商特有行为。
- 协议 Family：请求与响应结构、流式事件、工具、多模态及协议参数映射。
- `common`：HTTP、参数管线、音视频校验、通用解析等无供应商和无协议语义的原语。

Family Provider 不得直接导入 `providers.common`。Family 可以使用 `common`，但不能依赖具体 Provider；`common` 也不能反向依赖 Family 或 Provider。

## 能力归属

| Provider | 文本生成 | 音频转录 | Embedding |
| --- | --- | --- | --- |
| OpenAI Responses | `responses_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| Volcengine ARK | `responses_family` | `responses_family` | `responses_family` |
| SiliconFlow | `chat_completions_family` | `openai_auxiliary_family` | `openai_auxiliary_family` |
| Xiaomi Mimo | `chat_completions_family` | `chat_completions_family` | 不支持 |
| DashScope / Anthropic | 独立实现 | 独立实现 | 独立实现 |

同一 Provider 可以按端点协议进入不同 Family。新增能力时，先确定上游协议归属；只有无法归入现有 Family 的供应商差异才留在具体 Provider。

具体 Provider 的 `multimodal.py`、`tools.py` 和参数模块是实际扩展点，由 Provider Mapper 调用后再委托 Family，不能只保留未接入主链路的转发文件。

## 导入边界

Core 在运行时使用动态包名加载插件，同时已经占用顶层 `src` 包。因此，MaiDock 的生产代码必须使用包相对导入，根入口固定使用 `from .src.plugin import create_plugin`；`src/` 内同目录使用 `.module`，跨目录使用 `..` / `...`。顶层 `src` 导入仅供从插件根目录运行的本地测试使用。

导入排序和格式统一交给 Ruff，禁止为了视觉顺序手工调整。
