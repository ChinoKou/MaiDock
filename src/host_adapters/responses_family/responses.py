import logging
from collections.abc import Mapping
from typing import Literal

from ...core.common import (
    RuntimeOptionsView,
    build_usage_from_snapshot,
    message_text,
    read_model_identifier,
    tool_arguments_to_json,
)
from ...core.diagnostics import build_parse_error_message, sanitize_for_log, sanitize_upstream_detail
from ...core.json_types import (
    JsonValue,
    json_mapping_or_none,
)
from ...core.parameter_catalog import get_parameter_catalog
from ...core.parameter_policy import ProviderPolicyKey
from ...i18n import runtime_item, runtime_subject, translate
from ...schemas.host_snapshots import (
    MessageSnapshot,
    ResponseRequestSnapshot,
    ToolCallSnapshot,
    ToolOptionSnapshot,
)
from ...schemas.provider_contracts import ProviderResponse, ProviderToolCall
from ...schemas.responses_compat import (
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
)
from ...schemas.sdk_dump import SdkDumpAdapter
from ..common.parameter_translation import (
    NormalizedHostParameters,
    TranslationContext,
    TranslationEnvelope,
    build_normalized_host_parameters,
)
from ..common.reasoning import merge_reasoning_and_xml_tool_fallback
from .multimodal import convert_user_content_parts as convert_family_user_content_parts
from .multimodal import extract_reasoning_content as extract_family_reasoning_content
from .multimodal import extract_text_content as extract_family_text_content
from .parameter_translation import (
    apply_responses_parameters as apply_family_responses_parameters,
)
from .status import is_length_incomplete
from .tools import convert_tools as convert_family_tools
from .tools import extract_tool_calls as extract_family_tool_calls

_RESPONSES_CATALOG = get_parameter_catalog("openai_responses", "response")
RESPONSES_RESERVED_BODY_KEYS = set(_RESPONSES_CATALOG.reserved_body_keys)


class ResponsesMapper:
    """构建并解析 Responses 兼容的请求/响应负载。"""

    def __init__(
        self,
        *,
        options: RuntimeOptionsView,
        logger: logging.Logger,
        provider_label: str,
        raw_provider: str,
        policy_provider: ProviderPolicyKey,
    ) -> None:
        self.options = options
        self.logger = logger
        self.provider_label = provider_label
        self.raw_provider = raw_provider
        self.policy_provider: ProviderPolicyKey = policy_provider

    def build_request(self, request: object) -> OpenAIResponsesRequest:
        request_model = (
            request if isinstance(request, ResponseRequestSnapshot) else ResponseRequestSnapshot.model_validate(request)
        )
        overrides = self.options.parameter_overrides.get(self.policy_provider, "response")
        catalog = get_parameter_catalog(self.policy_provider, "response")
        normalized = build_normalized_host_parameters(
            request_model,
            overrides=overrides,
            catalog=catalog,
            provider_label=self.provider_label,
            capability="response",
        )
        return OpenAIResponsesRequest(
            model=read_model_identifier(request_model.model_info),
            input=self._convert_messages(request_model.message_list),
            tools=self._convert_tools(request_model.tool_options),
            normalized=normalized,
        )

    def build_http_body(
        self,
        upstream_request: OpenAIResponsesRequest,
        *,
        stream: bool,
    ) -> dict[str, JsonValue]:
        overrides = self.options.parameter_overrides.get(self.policy_provider, "response")
        normalized = upstream_request.normalized
        if not isinstance(normalized, NormalizedHostParameters):
            raise TypeError(
                translate(
                    "runtime.error.required",
                    subject="OpenAIResponsesRequest.normalized",
                    field="NormalizedHostParameters",
                )
            )

        input_params = upstream_request.input_params()
        tool_params = upstream_request.tool_params()
        body = {
            "model": upstream_request.model,
            "input": input_params,
            "stream": stream,
        }
        if upstream_request.tools:
            body["tools"] = tool_params
        envelope = TranslationEnvelope(body=body)
        context = TranslationContext(
            request=None,
            provider_label=self.provider_label,
            provider=self.policy_provider,
            capability="response",
            catalog=get_parameter_catalog(self.policy_provider, "response"),
            overrides=overrides,
            normalized=normalized,
            model=upstream_request.model,
        )
        self._apply_response_parameters(context, envelope)
        if envelope.headers:
            upstream_request.extra_headers.update(envelope.headers)
        if envelope.query:
            upstream_request.extra_query.update(envelope.query)
        return envelope.body

    def convert_response(self, response: object) -> ProviderResponse:
        raw_payload = SdkDumpAdapter.to_plain_dict(response)
        error_message = self._payload_error_message(raw_payload)
        if error_message is not None:
            raise ValueError(error_message)
        response_model = OpenAIResponseSnapshot.model_validate(raw_payload)
        self._raise_for_terminal_error(response_model, raw_payload)
        length_incomplete = is_length_incomplete(raw_payload)
        tool_calls = self._extract_tool_calls(response_model.output)
        text_content = self._extract_text_content(response_model)
        native_reasoning = self._extract_reasoning_content(response_model.output)
        reasoning_content, final_content = merge_reasoning_and_xml_tool_fallback(
            content=text_content,
            native_reasoning=native_reasoning,
            tool_calls=tool_calls,
            options=self.options,
        )

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
        if not final_content and not tool_calls and not (length_incomplete and reasoning_content):
            message = translate(
                "runtime.error.output_missing",
                item=runtime_item("output_text_reasoning_or_tools"),
            )
            raise ValueError(build_parse_error_message(self.provider_label, message))
        return ProviderResponse(
            content=final_content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
            raw_data=raw_data,
        )

    def _payload_error_message(self, payload: dict[str, JsonValue]) -> str | None:
        status = payload.get("status")
        if status in {"failed", "incomplete"}:
            return None
        error_payload = json_mapping_or_none(payload.get("error"))
        if error_payload is None:
            return None
        message = error_payload.get("message") or error_payload.get("code") or error_payload.get("type")
        return translate(
            "runtime.error.upstream_status",
            provider=self.provider_label,
            details=sanitize_for_log(message or error_payload),
        )

    def _raise_for_terminal_error(
        self, response_model: OpenAIResponseSnapshot, raw_payload: dict[str, JsonValue]
    ) -> None:
        if response_model.status not in {"failed", "incomplete"}:
            return
        if is_length_incomplete(raw_payload):
            self.logger.warning(translate("runtime.log.response_truncated", provider=self.provider_label))
            return
        error = raw_payload.get("error") or raw_payload.get("incomplete_details") or response_model.status
        message = translate(
            "runtime.error.response_invalid",
            provider=self.provider_label,
            field="status",
            details=f"{response_model.status}: {sanitize_upstream_detail(error)}",
        )
        raise ValueError(build_parse_error_message(self.provider_label, message))

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
                    raise ValueError(
                        translate(
                            "runtime.error.required",
                            subject=f"{self.provider_label} {runtime_subject('historical_tool_call')} {name}",
                            field="call_id",
                        )
                    )
                item_id, status = self._extract_tool_call_item_metadata(tool_call)
                converted.append(
                    OpenAIFunctionCallInputItem(
                        call_id=call_id,
                        name=name,
                        arguments=tool_arguments_to_json(
                            tool_call.function.arguments,
                            self.options.tool_argument_parse_mode,
                        ),
                        id=item_id,
                        status=status,
                    )
                )
                tool_call_names[call_id] = name
                emitted_function_call_ids.add(call_id)

        return converted

    def _convert_tools(self, tool_options: list[ToolOptionSnapshot]) -> list[OpenAIResponsesTool]:
        return convert_family_tools(tool_options)

    def _convert_user_content_parts(
        self,
        message: MessageSnapshot,
    ) -> list[OpenAIInputTextBlock | OpenAIInputImageBlock]:
        return convert_family_user_content_parts(message, logger=self.logger, options=self.options)

    def _apply_response_parameters(self, context: TranslationContext, envelope: TranslationEnvelope) -> None:
        apply_family_responses_parameters(context, envelope)

    def _extract_tool_calls(self, output: list[OpenAIResponseOutputItem]) -> list[ProviderToolCall]:
        return extract_family_tool_calls(
            output,
            options=self.options,
            raw_provider=self.raw_provider,
        )

    def _extract_text_content(self, response_model: OpenAIResponseSnapshot) -> str:
        return extract_family_text_content(response_model)

    def _extract_reasoning_content(self, output: list[OpenAIResponseOutputItem]) -> str | None:
        return extract_family_reasoning_content(output)

    @staticmethod
    def _extract_tool_call_item_metadata(
        tool_call: ToolCallSnapshot,
    ) -> tuple[str | None, Literal["in_progress", "completed", "incomplete"] | None]:
        extra_content = tool_call.extra_content.to_plain_dict()
        item_id = extra_content.get("item_id")
        status = extra_content.get("status")
        normalized_item_id = item_id.strip() if isinstance(item_id, str) and item_id.strip() else None
        normalized_status: Literal["in_progress", "completed", "incomplete"] | None = None
        if status == "in_progress":
            normalized_status = "in_progress"
        elif status == "completed":
            normalized_status = "completed"
        elif status == "incomplete":
            normalized_status = "incomplete"
        return normalized_item_id, normalized_status

    @staticmethod
    def _orphan_tool_result_message(message: MessageSnapshot, tool_call_names: Mapping[str, str]) -> OpenAIInputMessage:
        call_id = (message.tool_call_id or "").strip()
        tool_name = tool_call_names.get(call_id, message.tool_name or "tool")
        label = f"{tool_name} ({call_id or 'unknown'})"
        return OpenAIInputMessage(
            role="user",
            content=[
                OpenAIInputTextBlock(
                    text=f"工具调用结果（缺少可回放的 assistant function_call）：{label}: {message_text(message)}"
                )
            ],
        )
