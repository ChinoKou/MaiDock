import logging
import math
from collections.abc import Awaitable, Callable
from typing import Literal, cast

from maibot_sdk import LLMProviderBase
from openai import DEFAULT_MAX_RETRIES, APIConnectionError, APIStatusError, AsyncOpenAI, not_given

from .common import (
    ProviderRuntimeOptions,
    build_audio_file,
    build_openai_compatible_client_config,
    build_usage_from_snapshot,
    extract_response_format,
    image_data_url,
    log_request_summary,
    log_response_summary,
    merge_extra_params,
    message_text,
    read_max_retries,
    read_model_identifier,
    read_timeout,
    split_request_overrides,
    tool_arguments_to_json,
)
from .diagnostics import (
    build_connection_error_message,
    build_parse_error_message,
    build_status_error_message,
    sanitize_for_log,
)
from .parsing import (
    extract_xml_tool_calls,
    fallback_tool_call_id,
    merge_native_or_text_reasoning,
    normalize_arguments,
)
from .schemas import (
    ApiProviderSnapshot,
    AudioTranscriptionRequestSnapshot,
    EmbeddingRequestSnapshot,
    GenericUsageSnapshot,
    JsonObject,
    MessagePartImage,
    MessagePartText,
    MessageSnapshot,
    ObjectFields,
    OpenAIEasyInputMessage,
    OpenAIFunctionCallInputItem,
    OpenAIFunctionCallOutputItem,
    OpenAIInputImageBlock,
    OpenAIInputMessage,
    OpenAIInputTextBlock,
    OpenAIRawData,
    OpenAIResponseInputItem,
    OpenAIResponseOutputItem,
    OpenAIResponseSnapshot,
    OpenAIResponsesRequest,
    OpenAIResponsesTool,
    OpenAITextConfig,
    ProviderFunctionCall,
    ProviderResponse,
    ProviderToolCall,
    ResponseRequestSnapshot,
    SdkDumpAdapter,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)

logger = logging.getLogger("maibot_plugin.maidock.openai_responses")

OPENAI_RESPONSES_DIRECT_BODY_KEYS = {
    "include",
    "instructions",
    "max_output_tokens",
    "metadata",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "service_tier",
    "store",
    "text",
    "tool_choice",
    "top_p",
    "truncation",
    "user",
}
OPENAI_RESPONSES_RESERVED_BODY_KEYS = {
    "input",
    "model",
    "stream",
    "temperature",
    "tools",
}


class OpenAIResponsesProvider(LLMProviderBase):
    """基于 OpenAI Responses API 的 Provider。"""

    def __init__(self, *, options: ProviderRuntimeOptions) -> None:
        self.options = options

    async def get_response(self, request: JsonObject) -> JsonObject:
        request_model = ResponseRequestSnapshot.model_validate(request)
        client = self._build_client(request_model.api_provider)
        upstream_request = self._build_request(request_model)
        log_request_summary(
            logger,
            provider_label="openai-responses",
            model=upstream_request.model,
            messages=len(upstream_request.input),
            tools=len(upstream_request.tools),
            extra=upstream_request.model_dump(mode="json", exclude_none=True),
            options=self.options,
        )

        try:
            response = await self._execute_response_request(client, request_model, upstream_request)
        except APIConnectionError as exc:
            raise RuntimeError(build_connection_error_message("OpenAI Responses", exc)) from exc
        except APIStatusError as exc:
            raise RuntimeError(build_status_error_message("OpenAI Responses", exc)) from exc
        except Exception as exc:
            if isinstance(exc, (RuntimeError, ValueError, TypeError)):
                raise
            raise RuntimeError(f"OpenAI Responses 调用失败: {exc}") from exc

        result = self._convert_response(response)
        log_response_summary(
            logger,
            provider_label="openai-responses",
            content=result.content,
            tool_calls=result.tool_calls,
            usage=result.usage,
            options=self.options,
        )
        return result.to_host_dict()

    async def get_embedding(self, request: JsonObject) -> JsonObject:
        request_model = EmbeddingRequestSnapshot.model_validate(request)
        client = self._build_client(request_model.api_provider)
        model = read_model_identifier(request_model.model_info)
        extra_params = merge_extra_params(request_model)
        overrides = split_request_overrides(extra_params, strict_extra_params=self.options.strict_extra_params)

        try:
            raw_response = await client.embeddings.create(
                model=model,
                input=request_model.embedding_input,
                extra_headers=overrides.extra_headers or None,
                extra_query=overrides.extra_query or None,
                extra_body=overrides.extra_body or None,
            )
        except APIConnectionError as exc:
            raise RuntimeError(build_connection_error_message("OpenAI Embeddings", exc)) from exc
        except APIStatusError as exc:
            raise RuntimeError(build_status_error_message("OpenAI Embeddings", exc)) from exc

        payload = SdkDumpAdapter.to_plain_dict(raw_response)
        embedding = self._extract_embedding(payload)
        usage = build_usage_from_snapshot(GenericUsageSnapshot.model_validate(payload.get("usage") or {}))
        raw_data = (
            sanitize_for_log({"model": payload.get("model"), "usage": payload.get("usage")})
            if self.options.include_raw_data
            else None
        )
        return ProviderResponse(
            embedding=embedding, usage=usage, raw_data=cast(JsonObject | None, raw_data)
        ).to_host_dict()

    async def get_audio_transcriptions(self, request: JsonObject) -> JsonObject:
        request_model = AudioTranscriptionRequestSnapshot.model_validate(request)
        client = self._build_client(request_model.api_provider)
        model = read_model_identifier(request_model.model_info)
        extra_params = merge_extra_params(request_model)
        overrides = split_request_overrides(extra_params, strict_extra_params=self.options.strict_extra_params)
        audio_file = build_audio_file(request_model)

        try:
            raw_response = await client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                extra_headers=overrides.extra_headers or None,
                extra_query=overrides.extra_query or None,
                extra_body=overrides.extra_body or None,
            )
        except APIConnectionError as exc:
            raise RuntimeError(build_connection_error_message("OpenAI Audio Transcriptions", exc)) from exc
        except APIStatusError as exc:
            raise RuntimeError(build_status_error_message("OpenAI Audio Transcriptions", exc)) from exc

        content = raw_response if isinstance(raw_response, str) else getattr(raw_response, "text", None)
        if not isinstance(content, str):
            payload = SdkDumpAdapter.to_plain(raw_response)
            if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                content = payload["text"]
        if not isinstance(content, str):
            raise ValueError(build_parse_error_message("OpenAI Audio Transcriptions", "缺少文本内容"))
        raw_data = sanitize_for_log(SdkDumpAdapter.to_plain(raw_response)) if self.options.include_raw_data else None
        return ProviderResponse(content=content, raw_data=cast(JsonObject | None, raw_data)).to_host_dict()

    def _build_client(self, api_provider: ApiProviderSnapshot) -> AsyncOpenAI:
        client_config = build_openai_compatible_client_config(api_provider)
        timeout = read_timeout(api_provider)
        return AsyncOpenAI(
            api_key=client_config.api_key,
            base_url=client_config.base_url,
            organization=api_provider.organization.strip()
            if api_provider.organization is not None and api_provider.organization.strip()
            else None,
            project=api_provider.project.strip()
            if api_provider.project is not None and api_provider.project.strip()
            else None,
            default_headers=client_config.default_headers or None,
            default_query=client_config.default_query or None,
            timeout=timeout if timeout is not None else not_given,
            max_retries=read_max_retries(api_provider, DEFAULT_MAX_RETRIES),
        )

    def _build_request(self, request: ResponseRequestSnapshot) -> OpenAIResponsesRequest:
        extra_params = merge_extra_params(request)
        text_config = self._build_text_config(request, extra_params)
        overrides = split_request_overrides(
            extra_params,
            direct_body_keys=OPENAI_RESPONSES_DIRECT_BODY_KEYS,
            reserved_body_keys=OPENAI_RESPONSES_RESERVED_BODY_KEYS,
            strict_extra_params=self.options.strict_extra_params,
        )
        if text_config is not None:
            if "text" in overrides.direct_params:
                raise ValueError("extra_params.text 与 response_format 不能同时设置")
            overrides.direct_params["text"] = text_config.to_sdk_param()

        max_tokens = request.max_tokens or request.model_info.max_tokens
        temperature = request.temperature if request.temperature is not None else request.model_info.temperature
        return OpenAIResponsesRequest(
            model=read_model_identifier(request.model_info),
            input=self._convert_messages(request.message_list),
            max_output_tokens=max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None,
            temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
            tools=self._convert_tools(request.tool_options),
            text=text_config,
            extra_headers=overrides.extra_headers,
            extra_query=overrides.extra_query,
            extra_body=overrides.extra_body,
            direct_params=overrides.direct_params,
        )

    def _build_text_config(self, request: ResponseRequestSnapshot, extra_params: JsonObject) -> OpenAITextConfig | None:
        text_payload = extra_params.pop("text", None)
        response_text_config = extract_response_format(request)
        if text_payload is None:
            return response_text_config
        text_fields = ObjectFields.from_unknown(text_payload).to_plain_dict()
        if response_text_config is not None:
            if "format" in text_fields:
                raise ValueError("extra_params.text.format 与 response_format 不能同时设置")
            text_fields["format"] = response_text_config.format
        return OpenAITextConfig.model_validate(text_fields)

    async def _execute_response_request(
        self,
        client: AsyncOpenAI,
        request: ResponseRequestSnapshot,
        upstream_request: OpenAIResponsesRequest,
    ) -> object:
        kwargs = self._build_response_create_kwargs(request, upstream_request)
        create_response = cast(Callable[..., Awaitable[object]], client.responses.create)
        if request.model_info.force_stream_mode:
            kwargs["stream"] = True
            stream = await create_response(**kwargs)
            return await self._collect_stream_response(stream)
        kwargs["stream"] = False
        return await create_response(**kwargs)

    def _build_response_create_kwargs(
        self,
        request: ResponseRequestSnapshot,
        upstream_request: OpenAIResponsesRequest,
    ) -> JsonObject:
        del request
        kwargs: JsonObject = {
            "model": upstream_request.model,
            "input": upstream_request.input_params(),
            "extra_headers": upstream_request.extra_headers or None,
            "extra_query": upstream_request.extra_query or None,
            "extra_body": upstream_request.extra_body or None,
        }
        if upstream_request.max_output_tokens is not None and "max_output_tokens" not in upstream_request.direct_params:
            kwargs["max_output_tokens"] = upstream_request.max_output_tokens
        if upstream_request.temperature is not None and "temperature" not in upstream_request.direct_params:
            kwargs["temperature"] = upstream_request.temperature
        tools = upstream_request.tool_params()
        if tools:
            kwargs["tools"] = tools
        kwargs.update(upstream_request.direct_params)
        return kwargs

    async def _collect_stream_response(self, stream: object) -> object:
        final_response: object | None = None
        output_text_chunks: list[str] = []
        async for event in stream:  # type: ignore[attr-defined]
            plain_event = SdkDumpAdapter.to_plain(event)
            if isinstance(plain_event, dict):
                response = plain_event.get("response")
                if isinstance(response, dict):
                    final_response = response
                delta = plain_event.get("delta")
                if isinstance(delta, str):
                    output_text_chunks.append(delta)
            response_attr = getattr(event, "response", None)
            if response_attr is not None:
                final_response = response_attr
        if final_response is not None:
            return final_response
        return {"output_text": "".join(output_text_chunks), "output": [], "usage": {}}

    def _convert_messages(self, messages: list[MessageSnapshot]) -> list[OpenAIResponseInputItem]:
        converted: list[OpenAIResponseInputItem] = []
        tool_call_names: dict[str, str] = {}
        emitted_function_call_ids: set[str] = set()

        for message in messages:
            if message.role == "tool":
                call_id = (message.tool_call_id or "").strip()
                if call_id and call_id in emitted_function_call_ids:
                    converted.append(OpenAIFunctionCallOutputItem(call_id=call_id, output=message_text(message)))
                else:
                    converted.append(self._orphan_tool_result_message(message, tool_call_names))
                continue
            if message.role not in {"system", "user", "assistant"}:
                continue
            if message.role == "assistant":
                assistant_text = message_text(message)
                if assistant_text:
                    converted.append(OpenAIEasyInputMessage(content=assistant_text))
            else:
                input_content = self._convert_user_content_parts(message)
                if input_content:
                    role: Literal["system", "user"] = "system" if message.role == "system" else "user"
                    converted.append(OpenAIInputMessage(role=role, content=input_content))
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                if not name:
                    continue
                call_id = tool_call.resolved_call_id()
                if not call_id:
                    raise ValueError(f"OpenAI Responses 历史工具调用 {name} 缺少 call_id，无法构建 function_call")
                item_id, status = self._extract_openai_tool_call_item_metadata(tool_call)
                converted.append(
                    OpenAIFunctionCallInputItem(
                        call_id=call_id,
                        name=name,
                        arguments=tool_arguments_to_json(
                            tool_call.function.arguments, self.options.tool_argument_parse_mode
                        ),
                        id=item_id,
                        status=status,
                    )
                )
                tool_call_names[call_id] = name
                emitted_function_call_ids.add(call_id)

        return converted

    def _convert_user_content_parts(
        self, message: MessageSnapshot
    ) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
        parts: list[OpenAIInputTextBlock | OpenAIInputImageBlock] = []
        for part in message.parts:
            if isinstance(part, MessagePartText) and part.text:
                parts.append(OpenAIInputTextBlock(text=part.text))
            elif message.role == "user" and isinstance(part, MessagePartImage):
                data_url = image_data_url(part, logger, self.options.invalid_image_policy, self.options.image_limits)
                if data_url:
                    parts.append(OpenAIInputImageBlock(image_url=data_url, detail="auto"))
                elif self.options.invalid_image_policy == "placeholder":
                    parts.append(OpenAIInputTextBlock(text="[图片内容不可用]"))
        return parts

    @staticmethod
    def _extract_openai_tool_call_item_metadata(
        tool_call: ToolCallSnapshot,
    ) -> tuple[str | None, Literal["in_progress", "completed", "incomplete"] | None]:
        metadata = tool_call.extra_content.to_plain_dict().get("openai_responses")
        if not isinstance(metadata, dict):
            return None, None
        item_id = metadata.get("item_id")
        status = metadata.get("status")
        normalized_item_id = item_id if isinstance(item_id, str) and item_id.strip() else None
        if status in {"in_progress", "completed", "incomplete"}:
            return normalized_item_id, cast(Literal["in_progress", "completed", "incomplete"], status)
        return normalized_item_id, None

    @staticmethod
    def _orphan_tool_result_message(
        message: MessageSnapshot,
        tool_call_names: dict[str, str],
    ) -> OpenAIInputMessage:
        call_id = (message.tool_call_id or "").strip()
        tool_name = (message.tool_name or tool_call_names.get(call_id) or "tool").strip()
        label = f"{tool_name} ({call_id or 'unknown'})"
        return OpenAIInputMessage(
            role="user",
            content=[
                OpenAIInputTextBlock(
                    text=f"工具调用结果（缺少可回放的 assistant function_call）：{label}: {message_text(message)}"
                )
            ],
        )

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
        tools: list[OpenAIResponsesTool] = []
        for tool in tool_options:
            function = tool.function_definition()
            if function.name is None or not function.name:
                continue
            tools.append(
                OpenAIResponsesTool(
                    name=function.name,
                    description=function.description,
                    parameters=function.parameters,
                    strict=False,
                )
            )
        return tools

    def _convert_response(self, response: object) -> ProviderResponse:
        response_model = OpenAIResponseSnapshot.model_validate(SdkDumpAdapter.to_plain_dict(response))
        tool_calls = self._extract_tool_calls(response_model.output)
        text_content = self._extract_text_content(response_model)
        native_reasoning = self._extract_reasoning_content(response_model.output)
        reasoning_content, final_content = merge_native_or_text_reasoning(
            content=text_content,
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
            OpenAIRawData(
                id=response_model.id,
                model=response_model.model,
                status=response_model.status,
                usage=response_model.usage,
            ).to_host_dict()
            if self.options.include_raw_data
            else None
        )
        if not final_content and not tool_calls:
            raise ValueError(build_parse_error_message("OpenAI Responses", "响应中既没有文本内容，也没有工具调用"))
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
            raw_data=raw_data,
        )

    def _extract_text_content(self, response_model: OpenAIResponseSnapshot) -> str | None:
        if response_model.output_text:
            return response_model.output_text
        text_parts: list[str] = []
        for item in response_model.output:
            if item.type != "message":
                continue
            for block in item.content:
                if block.type in {"output_text", "text"} and block.text:
                    text_parts.append(block.text)
        return "".join(text_parts) or None

    def _extract_reasoning_content(self, output_items: list[OpenAIResponseOutputItem]) -> str | None:
        reasoning_parts: list[str] = []
        for item in output_items:
            if item.type not in {"reasoning", "reasoning_summary"}:
                continue
            for block in item.summary or item.content:
                if block.text:
                    reasoning_parts.append(block.text)
        return "\n".join(reasoning_parts) if reasoning_parts else None

    @staticmethod
    def _build_openai_tool_call_extra_content(
        item: OpenAIResponseOutputItem,
        *,
        generated_call_id: bool,
    ) -> JsonObject:
        return {
            "provider": "openai_responses",
            "openai_responses": {
                "item_id": item.id,
                "status": item.status,
                "raw_arguments": item.arguments,
                "generated_call_id": generated_call_id,
            },
        }

    def _extract_tool_calls(self, output_items: list[OpenAIResponseOutputItem]) -> list[ProviderToolCall]:
        tool_calls: list[ProviderToolCall] = []
        for index, item in enumerate(output_items, start=1):
            if item.type != "function_call":
                continue
            generated_call_id = not bool(item.call_id and item.call_id.strip())
            call_id = (
                item.call_id.strip()
                if item.call_id and item.call_id.strip()
                else fallback_tool_call_id(f"oai_tool_{index}")
            )
            tool_calls.append(
                ProviderToolCall(
                    id=call_id,
                    function=ProviderFunctionCall(
                        name=item.name,
                        arguments=normalize_arguments(item.arguments, self.options.tool_argument_parse_mode),
                    ),
                    extra_content=self._build_openai_tool_call_extra_content(item, generated_call_id=generated_call_id),
                )
            )
        return tool_calls

    @staticmethod
    def _extract_embedding(payload: JsonObject) -> list[float]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(build_parse_error_message("OpenAI Embeddings", "缺少 embeddings 数据"))
        first_item = data[0]
        if not isinstance(first_item, dict):
            raise ValueError(build_parse_error_message("OpenAI Embeddings", "embedding 数据项不是 object"))
        raw_embedding = first_item.get("embedding")
        if not isinstance(raw_embedding, list):
            raise ValueError(build_parse_error_message("OpenAI Embeddings", "缺少 embedding 数组"))
        embedding: list[float] = []
        for index, item in enumerate(raw_embedding):
            try:
                value = float(item)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    build_parse_error_message(
                        "OpenAI Embeddings",
                        f"embedding[{index}] 无法转换为 float，类型为 {type(item).__name__}",
                    )
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    build_parse_error_message(
                        "OpenAI Embeddings",
                        f"embedding[{index}] 不是有限数值，类型为 {type(item).__name__}",
                    )
                )
            embedding.append(value)
        return embedding
