import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from typing import Literal, cast

from anthropic import DEFAULT_MAX_RETRIES, APIConnectionError, APIStatusError, AsyncAnthropic, not_given
from maibot_sdk import LLMProviderBase

from ..core.common import (
    ProviderRuntimeOptions,
    build_anthropic_client_config,
    build_usage_from_snapshot,
    image_media_type,
    log_request_summary,
    log_response_summary,
    merge_extra_params,
    message_text,
    normalize_image_for_openai,
    read_max_retries,
    read_model_identifier,
    read_timeout,
    split_request_overrides,
)
from ..core.json_types import mapping_to_json_object
from ..core.diagnostics import (
    build_connection_error_message,
    build_parse_error_message,
    build_status_error_message,
    sanitize_for_log,
)
from ..core.parsing import extract_xml_tool_calls, merge_native_or_text_reasoning, normalize_arguments
from ..core.schemas import (
    AnthropicContentBlock,
    AnthropicImageBlock,
    AnthropicImageMediaType,
    AnthropicImageSource,
    AnthropicMessage,
    AnthropicMessagesRequest,
    JsonObject,
    AnthropicRawData,
    AnthropicResponseSnapshot,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    ApiProviderSnapshot,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
    ObjectFields,
    ProviderFunctionCall,
    ProviderResponse,
    ProviderToolCall,
    ResponseRequestSnapshot,
    SdkDumpAdapter,
    ToolOptionSnapshot,
)

logger = logging.getLogger("maibot_plugin.maidock.anthropic_messages")
ANTHROPIC_SDK_LOGGER_NAME = "anthropic._base_client"
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

ANTHROPIC_DIRECT_BODY_KEYS = {
    "metadata",
    "service_tier",
    "stop_sequences",
    "thinking",
    "tool_choice",
    "top_k",
    "top_p",
}
ANTHROPIC_RESERVED_BODY_KEYS = {
    "max_tokens",
    "messages",
    "model",
    "stream",
    "system",
    "temperature",
    "tools",
}


class AnthropicMessagesProvider(LLMProviderBase):
    """基于 Anthropic Messages API 的 Provider。"""

    def __init__(self, *, options: ProviderRuntimeOptions) -> None:
        self.options = options
        self._configure_sdk_logging()

    async def get_response(self, request: JsonObject) -> JsonObject:
        request_model = ResponseRequestSnapshot.model_validate(request)
        client = self._build_client(request_model.api_provider)
        upstream_request = self._build_request(request_model)
        log_request_summary(
            logger,
            provider_label="anthropic",
            model=upstream_request.model,
            messages=len(upstream_request.messages),
            tools=len(upstream_request.tools),
            extra=upstream_request.model_dump(mode="json", exclude_none=True),
            options=self.options,
        )

        try:
            response = await self._execute_response_request(client, request_model, upstream_request)
        except APIConnectionError as exc:
            raise RuntimeError(build_connection_error_message("Anthropic Messages", exc)) from exc
        except APIStatusError as exc:
            raise RuntimeError(build_status_error_message("Anthropic Messages", exc)) from exc
        except Exception as exc:
            if isinstance(exc, (RuntimeError, ValueError, TypeError, NotImplementedError)):
                raise
            raise RuntimeError(f"Anthropic Messages 调用失败: {exc}") from exc

        result = self._convert_response(response)
        log_response_summary(
            logger,
            provider_label="anthropic",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    def _configure_sdk_logging(self) -> None:
        log_level = (self.options.anthropic_sdk_log_level or "inherit").strip().upper()
        if log_level in {"", "INHERIT"}:
            return
        logging.getLogger(ANTHROPIC_SDK_LOGGER_NAME).setLevel(LOG_LEVELS.get(log_level, logging.INFO))

    async def get_embedding(self, request: JsonObject) -> JsonObject:
        del request
        raise NotImplementedError("Anthropic Messages API 不提供 embedding 端点，请改用支持 embedding 的 Provider")

    async def get_audio_transcriptions(self, request: JsonObject) -> JsonObject:
        del request
        raise NotImplementedError(
            "Anthropic Messages API 不提供 audio_transcription 端点，请改用支持音频转写的 Provider"
        )

    def _build_client(self, api_provider: ApiProviderSnapshot) -> AsyncAnthropic:
        client_config = build_anthropic_client_config(api_provider, user_agent=self.options.anthropic_user_agent)
        timeout = read_timeout(api_provider)
        return AsyncAnthropic(
            api_key=client_config.api_key,
            base_url=client_config.base_url,
            default_headers=client_config.default_headers or None,
            default_query=client_config.default_query or None,
            timeout=timeout if timeout is not None else not_given,
            max_retries=read_max_retries(api_provider, DEFAULT_MAX_RETRIES),
        )

    def _build_request(self, request: ResponseRequestSnapshot) -> AnthropicMessagesRequest:
        extra_params = merge_extra_params(request)
        overrides = split_request_overrides(
            extra_params,
            direct_body_keys=ANTHROPIC_DIRECT_BODY_KEYS,
            reserved_body_keys=ANTHROPIC_RESERVED_BODY_KEYS,
            strict_extra_params=self.options.strict_extra_params,
        )
        max_tokens = request.max_tokens or request.model_info.max_tokens or 4096
        temperature = request.temperature if request.temperature is not None else request.model_info.temperature
        tools = self._convert_tools(request.tool_options)
        tool_choice = self._build_default_tool_choice(tools, overrides.direct_params)
        is_tool_required = False
        if tool_choice and tool_choice.fields.get("type") == "any":
            is_tool_required = True
        elif "tool_choice" in overrides.direct_params:
            tc = overrides.direct_params["tool_choice"]
            if isinstance(tc, Mapping):
                tool_choice_fields = cast(Mapping[object, object], tc)
                if tool_choice_fields.get("type") == "any":
                    is_tool_required = True
            elif isinstance(tc, str) and tc.lower() in ("any", "required"):
                is_tool_required = True

        return AnthropicMessagesRequest(
            model=read_model_identifier(request.model_info),
            messages=self._convert_messages(request.message_list),
            max_tokens=int(max_tokens),
            system=self._extract_system(request.message_list, is_tool_required),
            temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
            tools=tools,
            tool_choice=tool_choice,
            extra_headers=overrides.extra_headers,
            extra_query=overrides.extra_query,
            extra_body=overrides.extra_body,
            direct_params=overrides.direct_params,
        )

    async def _execute_response_request(
        self,
        client: AsyncAnthropic,
        request: ResponseRequestSnapshot,
        upstream_request: AnthropicMessagesRequest,
    ) -> object:
        kwargs = self._build_message_create_kwargs(upstream_request)
        create_message = cast(Callable[..., Awaitable[object]], client.messages.create)
        if request.model_info.force_stream_mode:
            kwargs["stream"] = True
            stream = await create_message(**kwargs)
            return await self._collect_stream_response(cast(AsyncIterable[object], stream))
        kwargs["stream"] = False
        return await create_message(**kwargs)

    def _build_message_create_kwargs(self, upstream_request: AnthropicMessagesRequest) -> JsonObject:
        kwargs: JsonObject = {
            "model": upstream_request.model,
            "messages": upstream_request.message_params(),
            "max_tokens": upstream_request.max_tokens,
            "extra_headers": upstream_request.extra_headers or None,
            "extra_query": upstream_request.extra_query or None,
            "extra_body": upstream_request.extra_body or None,
        }
        if upstream_request.system is not None and "system" not in upstream_request.direct_params:
            kwargs["system"] = upstream_request.system
        if upstream_request.temperature is not None and "temperature" not in upstream_request.direct_params:
            kwargs["temperature"] = upstream_request.temperature
        tool_params = upstream_request.tool_params()
        if tool_params:
            kwargs["tools"] = tool_params
        if upstream_request.tool_choice is not None and "tool_choice" not in upstream_request.direct_params:
            kwargs["tool_choice"] = upstream_request.tool_choice.to_plain_dict()
        kwargs.update(upstream_request.direct_params)
        return kwargs

    async def _collect_stream_response(self, stream: AsyncIterable[object]) -> object:
        content_blocks: list[JsonObject] = []
        input_json_parts: dict[int, list[str]] = {}
        message_payload: JsonObject = {"content": content_blocks, "usage": {}}
        async for event in stream:
            plain_event = SdkDumpAdapter.to_plain(event)
            if not isinstance(plain_event, Mapping):
                continue
            event_mapping = cast(Mapping[object, object], plain_event)
            event_payload = mapping_to_json_object(event_mapping)
            event_type = str(event_payload.get("type") or "")
            message = event_payload.get("message")
            content_block = event_payload.get("content_block")
            delta = event_payload.get("delta")
            if event_type == "message_start" and isinstance(message, Mapping):
                self._merge_message_start(
                    message_payload, mapping_to_json_object(cast(Mapping[object, object], message))
                )
            elif event_type == "content_block_start" and isinstance(content_block, Mapping):
                index = self._stream_block_index(event_payload, content_blocks)
                self._put_stream_block(
                    content_blocks,
                    index,
                    mapping_to_json_object(cast(Mapping[object, object], content_block)),
                )
            elif event_type == "content_block_delta" and isinstance(delta, Mapping):
                index = self._stream_block_index(event_payload, content_blocks)
                block = self._ensure_stream_block(content_blocks, index)
                self._apply_stream_delta(
                    block, mapping_to_json_object(cast(Mapping[object, object], delta)), input_json_parts, index
                )
            elif event_type == "content_block_stop":
                index = self._stream_block_index(event_payload, content_blocks)
                if 0 <= index < len(content_blocks):
                    self._finalize_stream_block(content_blocks[index], input_json_parts.get(index, []))
            elif event_type == "message_delta":
                self._merge_message_delta(message_payload, event_payload)
        message_payload["content"] = content_blocks
        return message_payload

    @staticmethod
    def _merge_message_start(message_payload: JsonObject, message: JsonObject) -> None:
        for key, value in message.items():
            if key == "content":
                continue
            if key == "usage" and isinstance(value, Mapping):
                AnthropicMessagesProvider._merge_stream_usage(
                    message_payload,
                    mapping_to_json_object(cast(Mapping[object, object], value)),
                )
                continue
            message_payload[key] = value

    @staticmethod
    def _merge_message_delta(message_payload: JsonObject, event: JsonObject) -> None:
        raw_delta = event.get("delta")
        if isinstance(raw_delta, Mapping):
            delta = mapping_to_json_object(cast(Mapping[object, object], raw_delta))
            for key in ("stop_reason", "stop_sequence"):
                value = delta.get(key)
                if value is not None:
                    message_payload[key] = value
            usage = delta.get("usage")
            if isinstance(usage, Mapping):
                AnthropicMessagesProvider._merge_stream_usage(
                    message_payload,
                    mapping_to_json_object(cast(Mapping[object, object], usage)),
                )
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            AnthropicMessagesProvider._merge_stream_usage(
                message_payload, mapping_to_json_object(cast(Mapping[object, object], usage))
            )

    @staticmethod
    def _merge_stream_usage(message_payload: JsonObject, usage: JsonObject) -> None:
        current_usage = message_payload.get("usage")
        merged_usage: JsonObject = (
            mapping_to_json_object(cast(Mapping[object, object], current_usage))
            if isinstance(current_usage, Mapping)
            else {}
        )
        merged_usage.update(usage)
        message_payload["usage"] = merged_usage

    @staticmethod
    def _stream_block_index(event: JsonObject, content_blocks: list[JsonObject]) -> int:
        index = event.get("index")
        if isinstance(index, int) and index >= 0:
            return index
        return max(len(content_blocks) - 1, 0)

    @staticmethod
    def _ensure_stream_block(content_blocks: list[JsonObject], index: int) -> JsonObject:
        while len(content_blocks) <= index:
            content_blocks.append({"type": "text", "text": ""})
        return content_blocks[index]

    def _put_stream_block(self, content_blocks: list[JsonObject], index: int, block: JsonObject) -> None:
        self._ensure_stream_block(content_blocks, index)
        content_blocks[index] = dict(block)

    @staticmethod
    def _apply_stream_delta(
        block: JsonObject,
        delta: JsonObject,
        input_json_parts: dict[int, list[str]],
        index: int,
    ) -> None:
        delta_type = str(delta.get("type") or "")
        text = delta.get("text")
        thinking = delta.get("thinking")
        partial_json = delta.get("partial_json")
        if isinstance(text, str) or delta_type == "text_delta":
            block["type"] = block.get("type") or "text"
            block["text"] = str(block.get("text") or "") + (text if isinstance(text, str) else "")
        if isinstance(thinking, str) or delta_type == "thinking_delta":
            block["type"] = block.get("type") or "thinking"
            block["thinking"] = str(block.get("thinking") or "") + (thinking if isinstance(thinking, str) else "")
        if isinstance(partial_json, str):
            input_json_parts.setdefault(index, []).append(partial_json)

    def _finalize_stream_block(self, block: JsonObject, input_parts: list[str]) -> None:
        if block.get("type") not in ("tool_use", "tool_calls") or not input_parts:
            return
        raw_input = "".join(input_parts)
        block["input"] = normalize_arguments(raw_input, self.options.tool_argument_parse_mode)

    def _extract_system(self, messages: list[MessageSnapshot], is_tool_required: bool = False) -> str | None:
        system = "\n\n".join(
            message_text(message) for message in messages if message.role == "system" and message_text(message)
        )
        return system or None

    def _convert_messages(self, messages: list[MessageSnapshot]) -> list[AnthropicMessage]:
        converted: list[AnthropicMessage] = []
        emitted_tool_use_ids: set[str] = set()
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                tool_use_id = (message.tool_call_id or "").strip()
                if tool_use_id and tool_use_id in emitted_tool_use_ids:
                    converted.append(
                        AnthropicMessage(
                            role="user",
                            content=[AnthropicToolResultBlock(tool_use_id=tool_use_id, content=message_text(message))],
                        )
                    )
                else:
                    converted.append(self._orphan_tool_result_message(message))
                continue
            if message.role not in {"user", "assistant"}:
                continue
            content = self._convert_content_blocks(message)
            if message.role == "assistant":
                tool_use_blocks = self._convert_assistant_tool_calls(message)
                content.extend(tool_use_blocks)
                emitted_tool_use_ids.update(block.id for block in tool_use_blocks)
            if content:
                role: Literal["user", "assistant"] = "assistant" if message.role == "assistant" else "user"
                converted.append(AnthropicMessage(role=role, content=content))
        if not converted:
            converted.append(AnthropicMessage(role="user", content=[AnthropicTextBlock(text="")]))
        return converted

    def _convert_content_blocks(self, message: MessageSnapshot) -> list[AnthropicContentBlock]:
        blocks: list[AnthropicContentBlock] = []
        for part in message.parts:
            if isinstance(part, MessagePartText) and part.text:
                blocks.append(AnthropicTextBlock(text=part.text))
            elif message.role == "user" and isinstance(part, MessagePartImage):
                image_block = self._convert_image_block(part)
                if image_block is not None:
                    blocks.append(image_block)
                elif self.options.invalid_image_policy == "placeholder":
                    blocks.append(AnthropicTextBlock(text="[图片内容不可用]"))
        return blocks

    def _convert_image_block(self, part: MessagePartImage) -> AnthropicImageBlock | None:
        normalized_image = normalize_image_for_openai(part, logger, self.options.image_limits)
        if normalized_image is None:
            if self.options.invalid_image_policy == "error":
                raise ValueError("图片数据无效，无法构建 Anthropic 图片消息片段")
            return None
        image_format, image_base64 = normalized_image
        media_type = self._image_media_type(image_format)
        return AnthropicImageBlock(source=AnthropicImageSource(media_type=media_type, data=image_base64))

    def _image_media_type(self, image_format: str | None) -> AnthropicImageMediaType:
        media_type = image_media_type(image_format)
        if media_type == "image/jpeg":
            return "image/jpeg"
        if media_type == "image/png":
            return "image/png"
        if media_type == "image/gif":
            return "image/gif"
        if media_type == "image/webp":
            return "image/webp"
        raise ValueError(f"Anthropic 不支持图片 media_type: {media_type}")

    def _convert_assistant_tool_calls(self, message: MessageSnapshot) -> list[AnthropicToolUseBlock]:
        blocks: list[AnthropicToolUseBlock] = []
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            if not name:
                continue
            tool_use_id = tool_call.resolved_call_id()
            if not tool_use_id:
                raise ValueError(f"Anthropic Messages 历史工具调用 {name} 缺少 tool_use id，无法构建 tool_use")
            blocks.append(
                AnthropicToolUseBlock(
                    id=tool_use_id,
                    name=name,
                    input=normalize_arguments(tool_call.function.arguments, self.options.tool_argument_parse_mode),
                )
            )
        return blocks

    @staticmethod
    def _orphan_tool_result_message(message: MessageSnapshot) -> AnthropicMessage:
        call_id = (message.tool_call_id or "").strip()
        tool_name = (message.tool_name or "tool").strip() or "tool"
        label = f"{tool_name} ({call_id or 'unknown'})"
        return AnthropicMessage(
            role="user",
            content=[
                AnthropicTextBlock(
                    text=f"工具调用结果（缺少可回放的 assistant tool_use）：{label}: {message_text(message)}"
                )
            ],
        )

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[AnthropicTool]:
        tools: list[AnthropicTool] = []
        for tool in tool_options:
            function = tool.function_definition()
            if function.name is None or not function.name:
                continue
            tools.append(
                AnthropicTool(name=function.name, description=function.description, input_schema=function.parameters)
            )
        return tools

    @staticmethod
    def _build_default_tool_choice(tools: list[AnthropicTool], direct_params: JsonObject) -> ObjectFields | None:
        if not tools or "tool_choice" in direct_params:
            return None
        return ObjectFields(fields={"type": "any", "disable_parallel_tool_use": True})

    def _convert_response(self, response: object) -> ProviderResponse:
        response_model = AnthropicResponseSnapshot.model_validate(SdkDumpAdapter.to_plain_dict(response))
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ProviderToolCall] = []
        for block in response_model.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type in {"thinking", "redacted_thinking"}:
                thinking = block.thinking or block.text
                if thinking:
                    reasoning_parts.append(thinking)
            elif block.type in ("tool_use", "tool_calls"):
                tool_calls.append(
                    ProviderToolCall(
                        id=block.id or "",
                        function=ProviderFunctionCall(name=block.name or "", arguments=block.input),
                        extra_content={"provider": "anthropic_messages"},
                    )
                )

        content = "".join(text_parts) or None
        native_reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        reasoning_content, final_content = merge_native_or_text_reasoning(
            content=content,
            native_reasoning=native_reasoning,
            parse_mode=self.options.reasoning_parse_mode,
        )
        if not tool_calls:
            reasoning_content, reasoning_tool_calls = extract_xml_tool_calls(
                reasoning_content, self.options.tool_argument_parse_mode
            )
            if reasoning_tool_calls:
                tool_calls.extend(reasoning_tool_calls)
            final_content, content_tool_calls = extract_xml_tool_calls(
                final_content, self.options.tool_argument_parse_mode
            )
            if content_tool_calls:
                tool_calls.extend(content_tool_calls)

        usage = build_usage_from_snapshot(response_model.usage)
        raw_data = (
            AnthropicRawData(
                id=response_model.id,
                model=response_model.model,
                stop_reason=response_model.stop_reason,
                usage=response_model.usage,
            ).to_host_dict()
            if self.options.include_raw_data
            else None
        )
        if not final_content and not tool_calls:
            raise ValueError(build_parse_error_message("Anthropic Messages", "响应中既没有文本内容，也没有工具调用"))
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
            raw_data=cast(JsonObject | None, sanitize_for_log(raw_data) if raw_data is not None else None),
        )
