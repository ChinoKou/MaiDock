import logging

from ..common.parameter_translation import (
    FieldTranslator,
    TranslationContext,
    TranslationEnvelope,
    normalize_dimensions,
    normalize_positive_int,
    normalize_temperature,
    plugin_header_value,
    run_translators,
    set_target_value,
)
from ..common.response_format import normalize_response_format_snapshot


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
            "阿里云百炼 DashScope native response_format 暂未确认支持 json_schema，不能擅自转译 Host schema"
        )
    raise ValueError(f"阿里云百炼 DashScope 不支持的 response_format.format_type: {response_format.format_type}")


def translate_dashscope_parameters_identity(target_name: str, *, field_name: str) -> FieldTranslator:
    def _translator(context: TranslationContext, envelope: TranslationEnvelope, value: object) -> None:
        del context
        set_target_value(envelope, ("body", "parameters", target_name), value)

    _translator.__name__ = f"translate_dashscope_{field_name}"
    return _translator


_logger = logging.getLogger("maibot_plugin.maidock.dashscope")


def translate_dashscope_tool_choice(
    context: TranslationContext,
    envelope: TranslationEnvelope,
    value: object,
) -> None:
    del context
    if isinstance(value, str) and value == "required":
        _logger.warning("[dashscope] tool_choice='required' 不是有效值，已降级为 'auto'")
        set_target_value(envelope, ("body", "parameters", "tool_choice"), "auto")
        return
    set_target_value(envelope, ("body", "parameters", "tool_choice"), value)


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
    "response_format": translate_dashscope_response_format,
    "result_format": translate_dashscope_parameters_identity("result_format", field_name="result_format"),
    "top_p": translate_dashscope_parameters_identity("top_p", field_name="top_p"),
    "top_k": translate_dashscope_parameters_identity("top_k", field_name="top_k"),
    "enable_thinking": translate_dashscope_parameters_identity("enable_thinking", field_name="enable_thinking"),
    "enable_search": translate_dashscope_parameters_identity("enable_search", field_name="enable_search"),
    "incremental_output": translate_dashscope_parameters_identity(
        "incremental_output", field_name="incremental_output"
    ),
    "stream": translate_dashscope_parameters_identity("stream", field_name="stream"),
    "parallel_tool_calls": translate_dashscope_parameters_identity(
        "parallel_tool_calls", field_name="parallel_tool_calls"
    ),
    "seed": translate_dashscope_parameters_identity("seed", field_name="seed"),
    "stop": translate_dashscope_parameters_identity("stop", field_name="stop"),
    "n": translate_dashscope_parameters_identity("n", field_name="n"),
    "presence_penalty": translate_dashscope_parameters_identity("presence_penalty", field_name="presence_penalty"),
    "repetition_penalty": translate_dashscope_parameters_identity(
        "repetition_penalty", field_name="repetition_penalty"
    ),
    "tool_choice": translate_dashscope_tool_choice,
    "tools": translate_dashscope_parameters_identity("tools", field_name="tools"),
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
}


def apply_dashscope_embedding_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, DASHSCOPE_EMBEDDING_TRANSLATORS)


def apply_dashscope_audio_parameters(context: TranslationContext, envelope: TranslationEnvelope) -> None:
    run_translators(context, envelope, DASHSCOPE_AUDIO_TRANSLATORS)
