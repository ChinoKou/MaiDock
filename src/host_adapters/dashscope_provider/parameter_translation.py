import re

from ...core.json_types import (
    JsonValue,
    json_list_or_none,
    json_mapping_or_none,
    mapping_to_json_object,
    normalize_json_value,
)
from ...i18n import runtime_expected, translate
from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    merge_body_object,
    normalize_dimensions,
    normalize_json_object_value,
    normalize_positive_int,
    normalize_temperature,
    plugin_header_value,
    run_translators,
    set_target_value,
)
from ..common.response_format import normalize_response_format_snapshot

_DASHSCOPE_BOOLEAN_PARAMETERS = frozenset(
    {
        "enable_code_interpreter",
        "enable_search",
        "enable_thinking",
        "incremental_output",
        "parallel_tool_calls",
        "stream",
        "tool_stream",
        "vl_high_resolution_images",
    }
)
_DASHSCOPE_POSITIVE_INTEGER_PARAMETERS = frozenset({"max_tokens", "max_completion_tokens", "thinking_budget"})
_DASHSCOPE_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_DASHSCOPE_QWEN_COMPLETION_PATTERN = re.compile(
    r"^qwen(?P<major>\d+)(?:\.(?P<minor>\d+))?-(?P<tier>max|plus|flash)(?:-|$)"
)
_DASHSCOPE_KIMI_COMPLETION_PATTERN = re.compile(r"^kimi-k(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:-|$)")
_DASHSCOPE_GLM_COMPLETION_PATTERN = re.compile(r"^glm-(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:-|$)")
_DASHSCOPE_MINIMAX_COMPLETION_PATTERN = re.compile(r"^minimax-m(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:-|$)")
_DASHSCOPE_DEEPSEEK_COMPLETION_PATTERN = re.compile(
    r"^deepseek-(?P<family>[vr])(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:-|$)"
)
_DASHSCOPE_QWEN_COMPLETION_THRESHOLDS = {
    "max": (3, 7),
    "plus": (3, 5),
    "flash": (3, 5),
}


def _matched_version(match: re.Match[str]) -> tuple[int, int]:
    return int(match.group("major")), int(match.group("minor") or 0)


def dashscope_supports_max_completion_tokens(model: str) -> bool:
    """按官方确认的模型系列判断新版完整输出预算参数。"""

    normalized = model.strip().lower()
    if not normalized or "/" in normalized:
        return False

    qwen_match = _DASHSCOPE_QWEN_COMPLETION_PATTERN.match(normalized)
    if qwen_match is not None:
        threshold = _DASHSCOPE_QWEN_COMPLETION_THRESHOLDS[qwen_match.group("tier")]
        return _matched_version(qwen_match) >= threshold

    kimi_match = _DASHSCOPE_KIMI_COMPLETION_PATTERN.match(normalized)
    if kimi_match is not None:
        return _matched_version(kimi_match) >= (2, 5)

    glm_match = _DASHSCOPE_GLM_COMPLETION_PATTERN.match(normalized)
    if glm_match is not None:
        return _matched_version(glm_match) >= (5, 0)

    minimax_match = _DASHSCOPE_MINIMAX_COMPLETION_PATTERN.match(normalized)
    if minimax_match is not None:
        return _matched_version(minimax_match) >= (2, 5)

    deepseek_match = _DASHSCOPE_DEEPSEEK_COMPLETION_PATTERN.match(normalized)
    if deepseek_match is None or "-distill-" in normalized:
        return False
    threshold = (3, 0) if deepseek_match.group("family") == "v" else (1, 0)
    return _matched_version(deepseek_match) >= threshold


def _has_explicit_dashscope_max_tokens(context: TranslationContext) -> bool:
    """区分用户显式覆写的原生旧字段与 Host 通用输出预算。

    只认覆写目录中的 max_tokens：Host 请求级类型字段属于通用输出预算，
    自动映射为 max_completion_tokens；用户显式覆写时才保留原生 max_tokens。
    """

    return "max_tokens" in context.overrides


def translate_dashscope_temperature(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "parameters", "temperature"), normalize_temperature(value))


def translate_dashscope_max_tokens(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(
        envelope,
        ("body", "parameters", "max_tokens"),
        normalize_positive_int(value, field_name="max_tokens"),
    )


def translate_dashscope_positive_integer(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(
            envelope,
            ("body", "parameters", target_name),
            normalize_positive_int(value, field_name=field_name),
        )

    _translator.__name__ = f"translate_dashscope_{field_name}"
    return _translator


def translate_dashscope_boolean(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        if not isinstance(value, bool):
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=field_name,
                    expected=runtime_expected("boolean"),
                    actual=type(value).__name__,
                )
            )
        set_target_value(envelope, ("body", "parameters", target_name), value)

    _translator.__name__ = f"translate_dashscope_{field_name}"
    return _translator


def translate_dashscope_reasoning_effort(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if not isinstance(value, str):
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="reasoning_effort",
                expected=runtime_expected("string"),
                actual=type(value).__name__,
            )
        )
    normalized = value.strip().lower()
    if normalized not in _DASHSCOPE_REASONING_EFFORTS:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="DashScope reasoning_effort",
                allowed="low/medium/high/xhigh/max",
            )
        )
    set_target_value(envelope, ("body", "parameters", "reasoning_effort"), normalized)


def translate_dashscope_search_options(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(
        envelope,
        ("body", "parameters", "search_options"),
        normalize_json_object_value(value, field_name="search_options"),
    )


def translate_dashscope_response_format(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    response_format = normalize_response_format_snapshot(value)
    format_type = response_format.format_type.strip().lower() if response_format.format_type is not None else None
    if format_type in {None, "text"}:
        return
    if format_type in {"json_object", "json_obj"}:
        set_target_value(envelope, ("body", "parameters", "response_format"), {"type": "json_object"})
        return
    if format_type == "json_schema":
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="DashScope native response_format json_schema",
                allowed="text/json_object",
            )
        )
    raise ValueError(
        translate(
            "runtime.error.unsupported_value",
            subject="DashScope response_format.format_type",
            allowed="text/json_object",
        )
    )


def translate_dashscope_parameters_identity(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, ("body", "parameters", target_name), value)

    _translator.__name__ = f"translate_dashscope_{field_name}"
    return _translator


def translate_dashscope_body_identity(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, ("body", target_name), value)

    _translator.__name__ = f"translate_dashscope_{field_name}"
    return _translator


def translate_dashscope_tool_choice(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "parameters", "tool_choice"), value)


def translate_dashscope_tools(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    """把目录工具追加到 Host 已注入的函数工具之后。"""

    del context
    parameters = merge_body_object(envelope, "parameters", {})
    existing_tools = json_list_or_none(parameters.get("tools")) or []
    incoming_tools = json_list_or_none(value)
    if incoming_tools is None:
        incoming_tools = [normalize_json_value(value)]
    parameters["tools"] = [*existing_tools, *incoming_tools]


def translate_dashscope_res_level(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(
        envelope,
        ("body", "parameters", "res_level"),
        normalize_positive_int(value, field_name="res_level"),
    )


def translate_dashscope_customized_model_id(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "input", "customized_model_id"), value)


def translate_dashscope_plugins(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("headers", "X-DashScope-Plugin"), plugin_header_value(value))


def translate_dashscope_embedding_dimensions(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    set_target_value(envelope, ("body", "parameters", "dimension"), normalize_dimensions(value))


DASHSCOPE_CHAT_TRANSLATORS: dict[str, FieldTranslator] = {
    "temperature": translate_dashscope_temperature,
    "max_tokens": translate_dashscope_max_tokens,
    "max_completion_tokens": translate_dashscope_positive_integer(
        "max_completion_tokens", field_name="max_completion_tokens"
    ),
    "thinking_budget": translate_dashscope_positive_integer("thinking_budget", field_name="thinking_budget"),
    "reasoning_effort": translate_dashscope_reasoning_effort,
    "response_format": translate_dashscope_response_format,
    "result_format": translate_dashscope_parameters_identity("result_format", field_name="result_format"),
    "top_p": translate_dashscope_parameters_identity("top_p", field_name="top_p"),
    "top_k": translate_dashscope_parameters_identity("top_k", field_name="top_k"),
    "enable_thinking": translate_dashscope_boolean("enable_thinking", field_name="enable_thinking"),
    "enable_search": translate_dashscope_boolean("enable_search", field_name="enable_search"),
    "incremental_output": translate_dashscope_boolean("incremental_output", field_name="incremental_output"),
    "stream": translate_dashscope_boolean("stream", field_name="stream"),
    "parallel_tool_calls": translate_dashscope_boolean("parallel_tool_calls", field_name="parallel_tool_calls"),
    "tool_stream": translate_dashscope_boolean("tool_stream", field_name="tool_stream"),
    "enable_code_interpreter": translate_dashscope_boolean(
        "enable_code_interpreter", field_name="enable_code_interpreter"
    ),
    "search_options": translate_dashscope_search_options,
    "vl_high_resolution_images": translate_dashscope_boolean(
        "vl_high_resolution_images", field_name="vl_high_resolution_images"
    ),
    "seed": translate_dashscope_parameters_identity("seed", field_name="seed"),
    "stop": translate_dashscope_parameters_identity("stop", field_name="stop"),
    "n": translate_dashscope_parameters_identity("n", field_name="n"),
    "presence_penalty": translate_dashscope_parameters_identity("presence_penalty", field_name="presence_penalty"),
    "repetition_penalty": translate_dashscope_parameters_identity(
        "repetition_penalty", field_name="repetition_penalty"
    ),
    "tool_choice": translate_dashscope_tool_choice,
    "tools": translate_dashscope_tools,
    "plugins": translate_dashscope_plugins,
    "customized_model_id": translate_dashscope_customized_model_id,
}

DASHSCOPE_EMBEDDING_TRANSLATORS: dict[str, FieldTranslator] = {
    "dimensions": translate_dashscope_embedding_dimensions,
    "output_type": translate_dashscope_parameters_identity("output_type", field_name="output_type"),
    "instruct": translate_dashscope_parameters_identity("instruct", field_name="instruct"),
    "text_type": translate_dashscope_parameters_identity("text_type", field_name="text_type"),
    "auto_truncation": translate_dashscope_parameters_identity("auto_truncation", field_name="auto_truncation"),
    "enable_fusion": translate_dashscope_parameters_identity("enable_fusion", field_name="enable_fusion"),
    "fps": translate_dashscope_parameters_identity("fps", field_name="fps"),
    "max_video_frames": translate_dashscope_parameters_identity("max_video_frames", field_name="max_video_frames"),
    "res_level": translate_dashscope_res_level,
}


def apply_dashscope_chat_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, DASHSCOPE_CHAT_TRANSLATORS)


def translate_dashscope_asr_option(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, ("body", "parameters", "asr_options", target_name), value)

    _translator.__name__ = f"translate_dashscope_asr_{field_name}"
    return _translator


DASHSCOPE_AUDIO_TRANSLATORS: dict[str, FieldTranslator] = {
    "language": translate_dashscope_asr_option("language", field_name="language"),
    "enable_itn": translate_dashscope_asr_option("enable_itn", field_name="enable_itn"),
    "format": translate_dashscope_body_identity("format", field_name="format"),
    "audio_format": translate_dashscope_body_identity("audio_format", field_name="audio_format"),
}


def apply_dashscope_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, DASHSCOPE_EMBEDDING_TRANSLATORS)


def apply_dashscope_audio_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, DASHSCOPE_AUDIO_TRANSLATORS)


def normalize_dashscope_chat_body(body: dict[str, JsonValue], *, context: TranslationContext) -> None:
    """在所有参数策略与 body 覆写完成后校验 DashScope 原生参数。"""

    raw_parameters = body.get("parameters")
    parameters_mapping = json_mapping_or_none(raw_parameters)
    if parameters_mapping is None:
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="DashScope parameters",
                expected=runtime_expected("object"),
                actual=type(raw_parameters).__name__,
            )
        )
    parameters = mapping_to_json_object(parameters_mapping)
    body["parameters"] = parameters

    for field_name in _DASHSCOPE_BOOLEAN_PARAMETERS:
        value = parameters.get(field_name)
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"DashScope {field_name}",
                    expected=runtime_expected("boolean"),
                    actual=type(value).__name__,
                )
            )
    for field_name in _DASHSCOPE_POSITIVE_INTEGER_PARAMETERS:
        value = parameters.get(field_name)
        if value is not None:
            parameters[field_name] = normalize_positive_int(value, field_name=field_name)

    reasoning_effort = parameters.get("reasoning_effort")
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str):
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject="DashScope reasoning_effort",
                    expected=runtime_expected("string"),
                    actual=type(reasoning_effort).__name__,
                )
            )
        normalized_effort = reasoning_effort.strip().lower()
        if normalized_effort not in _DASHSCOPE_REASONING_EFFORTS:
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject="DashScope reasoning_effort",
                    allowed="low/medium/high/xhigh/max",
                )
            )
        parameters["reasoning_effort"] = normalized_effort

    search_options = parameters.get("search_options")
    if search_options is not None:
        parameters["search_options"] = normalize_json_object_value(
            search_options,
            field_name="DashScope search_options",
        )

    _normalize_dashscope_token_limit(parameters, context=context)
    _normalize_dashscope_tool_choice(parameters)


def _normalize_dashscope_token_limit(parameters: dict[str, JsonValue], *, context: TranslationContext) -> None:
    legacy_value = parameters.get("max_tokens")
    completion_value = parameters.get("max_completion_tokens")
    explicit_legacy = _has_explicit_dashscope_max_tokens(context)

    if legacy_value is not None and completion_value is not None:
        if explicit_legacy:
            raise ValueError(
                translate(
                    "runtime.error.conflict",
                    left="parameters.max_tokens",
                    right="parameters.max_completion_tokens",
                )
            )
        parameters.pop("max_tokens")
        return

    if legacy_value is None or explicit_legacy:
        return
    if not dashscope_supports_max_completion_tokens(context.model):
        return

    parameters["max_completion_tokens"] = parameters.pop("max_tokens")


def _normalize_dashscope_tool_choice(parameters: dict[str, JsonValue]) -> None:
    tool_choice = parameters.get("tool_choice")
    if tool_choice is None:
        return

    if parameters.get("enable_thinking") is True:
        if isinstance(tool_choice, str) and tool_choice.strip().lower() in {"auto", "none"}:
            parameters["tool_choice"] = tool_choice.strip().lower()
            return
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="DashScope thinking mode tool_choice",
                allowed="auto/none",
            )
        )

    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in {"auto", "none"}:
            parameters["tool_choice"] = normalized
            return
        if normalized == "required":
            tool_names = _dashscope_tool_names(parameters.get("tools"))
            if len(tool_names) != 1:
                raise ValueError(
                    translate(
                        "runtime.error.unsupported_value",
                        subject="DashScope tool_choice=required",
                        allowed="exactly one final tool",
                    )
                )
            parameters["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_names[0]},
            }
            return
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject="DashScope tool_choice",
                allowed="auto/none/required/function object",
            )
        )

    tool_names = _dashscope_tool_names(parameters.get("tools"))
    choice_mapping = json_mapping_or_none(tool_choice)
    function = json_mapping_or_none(choice_mapping.get("function")) if choice_mapping is not None else None
    function_name = function.get("name") if function is not None else None
    choice_type = choice_mapping.get("type") if choice_mapping is not None else None
    if choice_type != "function" or not isinstance(function_name, str) or not function_name.strip():
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="DashScope tool_choice",
                expected="function selection object",
                actual=type(tool_choice).__name__,
            )
        )
    normalized_name = function_name.strip()
    if normalized_name not in tool_names:
        raise ValueError(
            translate(
                "runtime.error.unsupported_value",
                subject=f"DashScope tool_choice function {normalized_name}",
                allowed="one of the final tools",
            )
        )
    parameters["tool_choice"] = {
        "type": "function",
        "function": {"name": normalized_name},
    }


def _dashscope_tool_names(value: object) -> list[str]:
    if value is None:
        return []
    tools = json_list_or_none(value)
    if tools is None:
        raise TypeError(
            translate(
                "runtime.error.expected_type",
                subject="DashScope tools",
                expected=runtime_expected("list_of_objects"),
                actual=type(value).__name__,
            )
        )
    names: list[str] = []
    for index, tool in enumerate(tools):
        tool_mapping = json_mapping_or_none(tool)
        if tool_mapping is None:
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"DashScope tools[{index}]",
                    expected=runtime_expected("object"),
                    actual=type(tool).__name__,
                )
            )
        if tool_mapping.get("type") != "function":
            raise ValueError(
                translate(
                    "runtime.error.unsupported_value",
                    subject=f"DashScope tools[{index}].type",
                    allowed="function",
                )
            )
        function_value = tool_mapping.get("function")
        function = json_mapping_or_none(function_value)
        if function is None:
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"DashScope tools[{index}].function",
                    expected=runtime_expected("object"),
                    actual=type(function_value).__name__,
                )
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                translate(
                    "runtime.error.expected_type",
                    subject=f"DashScope tools[{index}].function.name",
                    expected=runtime_expected("non_empty_string"),
                    actual=type(name).__name__,
                )
            )
        names.append(name.strip())
    return names
