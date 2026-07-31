# 跨插件图像与视频 Public API

MaiDock 通过 SDK 2.7 动态公开图像和视频作业接口。调用方使用 `ctx.api.call()`，完整 API 名位于 `chinokou.maidock.media.*` 命名空间，版本固定为 `1`。所有接口只接受一个 `request` object，注册超时为 25 秒；生成过程在持久化作业中异步完成，调用方应通过查询接口跟踪状态。

Public API 默认关闭。管理员需要在 WebUI 的“跨插件 API”页配置至少一个上游 Profile（DashScope 或 Volcengine ARK）、选择默认 Profile，并打开总开关。Profile 名在两家供应商之间必须全局唯一——调用方只按名字寻址，不带供应商前缀。

## 统一响应

所有方法都返回相同 envelope：

```python
{
    "ok": True,
    "data": {...},
    "error": None,
}
```

失败时：

```python
{
    "ok": False,
    "data": None,
    "error": {
        "code": "PROFILE_NOT_FOUND",
        "message": "当前语言的错误信息",
        "retryable": False,
        "uncertain": False,
        "provider_request_id": None,
    },
}
```

`code`、`retryable` 和 `uncertain` 是稳定的程序判断字段；`message` 会随 MaiDock 当前语言变化。上游已接收提交但本地没有取得 remote handle 时，`uncertain=true`，调用方不得自动重复提交。

Public API 不包含调用鉴权字段。供应商 `api_key` 只存在于管理员配置的上游 Profile 中，调用方不得也无需传入供应商凭据。

## API 列表

| 完整 API 名 | `request` | 说明 |
| --- | --- | --- |
| `chinokou.maidock.media.capabilities` | `{}` | 返回脱敏 Profile、默认 Profile 和模型能力 |
| `chinokou.maidock.media.jobs.create` | `CreateJobRequest` | 提交图像或视频作业 |
| `chinokou.maidock.media.jobs.get` | `{"job_id": ...}` | 查询公开作业状态与输出 |
| `chinokou.maidock.media.jobs.cancel` | `{"job_id": ...}` | 请求取消排队或远端作业 |
| `chinokou.maidock.media.jobs.delete` | `{"job_id": ...}` | 删除终态作业并写入删除墓碑 |
| `chinokou.maidock.media.uploads.create` | `CreateUploadRequest` | 创建可分块上传记录 |
| `chinokou.maidock.media.uploads.upload` | `OneShotUploadRequest` | 不超过 8 MiB 的单次 bytes 上传 |
| `chinokou.maidock.media.uploads.get` | `{"upload_id": ...}` | 查询上传进度 |
| `chinokou.maidock.media.uploads.write_chunk` | `WriteUploadChunkRequest` | 按 recorded offset 写入最多 1 MiB |
| `chinokou.maidock.media.uploads.complete` | `{"upload_id": ...}` | 校验大小与 SHA-256 后完成上传 |
| `chinokou.maidock.media.uploads.delete` | `{"upload_id": ...}` | 删除未被活动作业引用的上传 |
| `chinokou.maidock.media.artifacts.read` | `ReadArtifactRequest` | 分块读取产物，单块最多 1 MiB |

## 提交作业

`media.jobs.create` 的 request 是严格 object，不接受未知字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `capability` | string | `image_generation` 或 `video_generation` |
| `profile` | string/null | 指定 Profile；省略时使用对应能力的默认 Profile |
| `model` | string/null | 指定模型；省略时使用 Profile 默认模型 |
| `protocol_family` | string/null | 显式锁定供应商协议族；公共层只验证非空 |
| `mode` | string | 见下方模式表 |
| `prompt` | string/null | 正向提示词 |
| `negative_prompt` | string/null | 负向提示词 |
| `inputs` | array | 最多 16 项带 role 的媒体输入 |
| `parameters` | object | 只允许可序列化的有限 JSON 值 |
| `idempotency_key` | string/null | 最长 200 字符 |

参数合并顺序固定为 Profile defaults、请求 `parameters`、Profile overrides。相同幂等键和相同请求摘要返回原作业；摘要不同返回 `IDEMPOTENCY_CONFLICT`。

支持的模式：

| capability | mode |
| --- | --- |
| 图像 | `text_to_image`、`image_edit` |
| 视频 | `text_to_video`、`first_frame_to_video`、`first_last_frame_to_video`、`video_continuation`、`reference_to_video`、`video_edit` |

输入项格式如下。`source` 必须且只能包含不带用户凭据的 HTTPS `url`，或已完成的 `upload_id`：

```python
{
    "role": "source_image",
    "source": {"upload_id": "upl_..."},
}
```

可用 role 为 `source_image`、`reference_image`、`first_frame`、`last_frame`、`first_clip`、`reference_video`、`video`、`driving_audio`。模型、模式、role、参数范围和输出数量的精确约束以 `media.capabilities` 返回值为准。

示例：

```python
result = await self.ctx.api.call(
    "chinokou.maidock.media.jobs.create",
    version="1",
    request={
        "capability": "image_generation",
        "mode": "text_to_image",
        "prompt": "雨后的城市街道",
        "parameters": {"size": "1024*1024", "n": 1},
        "idempotency_key": "my-plugin:scene:42",
    },
)
```

## 作业状态与输出

公开状态固定为 `queued`、`running`、`succeeded`、`failed`、`canceled`、`expired`。`outputs` 保持供应商输出顺序，并包含两种结构：

```python
{"type": "text", "text": "..."}

{
    "type": "artifact",
    "artifact_id": "art_...",
    "media_type": "image/png",
    "size": 123456,
    "sha256": "...",
    "width": 1024,
    "height": 1024,
    "duration_seconds": None,
    "expires_at": "2026-07-31T12:00:00Z",
}
```

部分产物失败时，成功产物仍会保留；`warnings` 和 `failed_output_count` 描述损失。只要至少一个媒体产物成功落盘，作业可进入 `succeeded`。

轮询示例：

```python
job = await self.ctx.api.call(
    "chinokou.maidock.media.jobs.get",
    version="1",
    request={"job_id": result["data"]["job_id"]},
)
```

## 上传与 artifact

单次上传适合不超过 8 MiB 的输入：

```python
upload = await self.ctx.api.call(
    "chinokou.maidock.media.uploads.upload",
    version="1",
    request={
        "media_type": "image/png",
        "data": image_bytes,
        "sha256": image_sha256,
        "file_name": "source.png",
    },
)
```

更大的文件使用 `media.uploads.create` 创建记录，然后重复调用 `media.uploads.write_chunk`。每一块必须从 `media.uploads.get` 返回的 `received_size` 开始；重试同一块前，服务端会按已记录 offset 截断未提交尾部。所有块完成后调用 `media.uploads.complete`。每次调用仍将对应 Command 放在唯一的 `request` 参数中。

`media.artifacts.read` 返回：

```python
{
    "artifact_id": "art_...",
    "offset": 0,
    "next_offset": 1048576,
    "eof": False,
    "chunk": b"...",
    "media_type": "video/mp4",
    "size": 7340032,
    "sha256": "...",
}
```

调用方应持续使用 `next_offset`，直到 `eof=true`，并在本地校验完整内容的 `size` 与 SHA-256。

## DashScope 路由

内置协议族为：

- `dashscope_multimodal_generation`
- `dashscope_image_generation`
- `dashscope_text2image_synthesis`
- `dashscope_image2image_synthesis`
- `dashscope_video_generation`

路由优先级是请求 `protocol_family`、Profile 精确 route、已知模型 registry 默认族。未知模型只有在请求或 Profile route 显式指定协议族时才能执行；失败后不探测、不 fallback。生成提交不重试，tasks 查询/取消、OSS 上传策略和 artifact 下载才使用 Profile 的安全重试设置。

## Volcengine ARK 路由

内置协议族为：

- `ark_images_generations`：图片，对应 `POST /api/v3/images/generations`
- `ark_content_generation_tasks`：视频，对应 `POST/GET/DELETE /api/v3/contents/generations/tasks`

路由优先级与 DashScope 一致。以 `ep-` 开头的接入点 ID 不在模型目录里，必须在请求或 Profile route 中显式指定协议族。

ARK 与 DashScope 有几处行为差异，调用方需要知道：

- **图片是同步接口**。提交后一次返回全部结果，不产生远端任务，作业直接进入终态。视频才是异步任务。
- **组图允许部分失败**。`data[]` 中被审核拦下的项不会让整单失败，而是计入 `failed_output_count` 并出现在 `warnings` 里。
- **`response_format` 锁定 `url`**，Profile override 也改不动。`b64_json` 会把整张图塞进 JSON 响应，与"先拿 URL、再落盘为 artifact"的流程冲突。
- **视频生成参数走请求体顶层字段**（ARK 文档称"新方式"，强校验），而不是拼在提示词后的 `--key value` 后缀（弱校验，填错会被静默忽略）。
- **取消只对排队中的任务有效**。ARK 用同一个 `DELETE` 表达两件事：`queued` 时是取消，`succeeded`/`failed`/`expired` 时是**不可逆地删除任务记录**。MaiDock 因此先查状态再决定：只有 `queued` 才真正发出 `DELETE`，`running` 与各终态一律原样返回，绝不触碰远端记录。对已完成的作业调用 `media.jobs.cancel` 不会丢失产物。
- **视频任务超时**映射为独立错误码 `UPSTREAM_TASK_EXPIRED`，与普通失败区分开。

### ARK 的上传语义

ARK 没有 DashScope 那样的 OSS 直传流程。`media.uploads.*` 上传的图片与音频会被就地转成 `data:<mime>;base64,...` 内联进请求体，不产生任何额外网络请求；单文件上限图片 30 MB、音频 15 MB。

**视频不支持上传**：ARK 请求体整体上限 64 MB，视频经 base64 编码后必然超限。注意 `media.uploads.*` 本身与供应商无关，视频文件仍会上传成功并占用配额；错误在作业执行阶段才出现——引用该上传的 ARK 作业会以 `UPLOAD_UNSUPPORTED` 失败。视频输入请直接传 HTTPS URL。

## 持久化与保留期

MaiDock 使用插件数据目录下独立的 `maidock_public_api.sqlite3`，上传与产物分别存放在 `public_api/uploads` 和 `public_api/artifacts`。数据库使用 WAL 和事务，文件通过 staging 原子落盘。

默认并发 2、队列 32、单上传/单产物 512 MiB、总配额 10 GiB。未完成上传保留 24 小时；完成上传与 artifact 保留 7 天；作业元数据、幂等记录和删除墓碑保留 30 天；远端任务最长跟踪 23 小时。管理员可以在“跨插件 API”配置页调整这些限制。
