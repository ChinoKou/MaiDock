from dataclasses import dataclass

from ...core.common import (
    ArkBuiltinEndpointMode,
    ImageProcessingLimits,
    InvalidImagePolicy,
    ProviderRuntimeOptions,
)
from ...core.parameter_policy import ParameterOverrideRegistry
from ...core.parsing import ReasoningParseMode, ToolArgumentParseMode
from ...i18n import Locale


@dataclass(frozen=True, slots=True)
class HostCommonOptions:
    """仅包含 Host 请求映射与结果转换所需的通用选项。"""

    locale: Locale
    include_raw_data: bool
    log_payload_summary: bool
    log_payload_debug: bool
    tool_argument_parse_mode: ToolArgumentParseMode
    reasoning_parse_mode: ReasoningParseMode
    invalid_image_policy: InvalidImagePolicy
    image_limits: ImageProcessingLimits
    parameter_overrides: ParameterOverrideRegistry


@dataclass(frozen=True, slots=True)
class ConnectionOptions:
    """Host 用于构造单次不可变 Connection 的供应商默认值。"""

    user_agent: str
    max_retries: int
    force_max_retries: bool
    retry_interval: float
    force_retry_interval: bool


@dataclass(frozen=True, slots=True)
class OpenAIHostOptions:
    connection: ConnectionOptions


@dataclass(frozen=True, slots=True)
class AnthropicHostOptions:
    connection: ConnectionOptions


@dataclass(frozen=True, slots=True)
class DashScopeHostOptions:
    connection: ConnectionOptions
    force_official_endpoint: bool
    auto_detect_endpoint: bool


@dataclass(frozen=True, slots=True)
class BailianHostOptions:
    connection: ConnectionOptions


@dataclass(frozen=True, slots=True)
class SiliconFlowHostOptions:
    connection: ConnectionOptions
    force_official_endpoint: bool


@dataclass(frozen=True, slots=True)
class ArkHostOptions:
    connection: ConnectionOptions
    force_official_endpoint: bool
    builtin_endpoint_mode: ArkBuiltinEndpointMode
    prefix_cache_enabled: bool
    prefix_cache_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class MimoHostOptions:
    connection: ConnectionOptions
    reasoning_retention_days: int


def build_host_common_options(options: ProviderRuntimeOptions) -> HostCommonOptions:
    return HostCommonOptions(
        locale=options.locale,
        include_raw_data=options.include_raw_data,
        log_payload_summary=options.log_payload_summary,
        log_payload_debug=options.log_payload_debug,
        tool_argument_parse_mode=options.tool_argument_parse_mode,
        reasoning_parse_mode=options.reasoning_parse_mode,
        invalid_image_policy=options.invalid_image_policy,
        image_limits=options.image_limits,
        parameter_overrides=options.parameter_overrides,
    )


def _connection(
    *,
    user_agent: str,
    max_retries: int,
    force_max_retries: bool,
    retry_interval: float,
    force_retry_interval: bool,
) -> ConnectionOptions:
    return ConnectionOptions(
        user_agent=user_agent,
        max_retries=max_retries,
        force_max_retries=force_max_retries,
        retry_interval=retry_interval,
        force_retry_interval=force_retry_interval,
    )


def build_openai_host_options(options: ProviderRuntimeOptions) -> OpenAIHostOptions:
    return OpenAIHostOptions(
        connection=_connection(
            user_agent=options.openai_user_agent,
            max_retries=options.openai_max_retries,
            force_max_retries=options.openai_force_max_retries,
            retry_interval=options.openai_retry_interval,
            force_retry_interval=options.openai_force_retry_interval,
        )
    )


def build_anthropic_host_options(options: ProviderRuntimeOptions) -> AnthropicHostOptions:
    return AnthropicHostOptions(
        connection=_connection(
            user_agent=options.anthropic_user_agent,
            max_retries=options.anthropic_max_retries,
            force_max_retries=options.anthropic_force_max_retries,
            retry_interval=options.anthropic_retry_interval,
            force_retry_interval=options.anthropic_force_retry_interval,
        )
    )


def build_dashscope_host_options(options: ProviderRuntimeOptions) -> DashScopeHostOptions:
    return DashScopeHostOptions(
        connection=_connection(
            user_agent=options.dashscope_user_agent,
            max_retries=options.dashscope_max_retries,
            force_max_retries=options.dashscope_force_max_retries,
            retry_interval=options.dashscope_retry_interval,
            force_retry_interval=options.dashscope_force_retry_interval,
        ),
        force_official_endpoint=options.dashscope_force_official_endpoint,
        auto_detect_endpoint=options.dashscope_auto_detect_endpoint,
    )


def build_bailian_host_options(options: ProviderRuntimeOptions) -> BailianHostOptions:
    return BailianHostOptions(
        connection=_connection(
            user_agent=options.bailian_user_agent,
            max_retries=options.bailian_max_retries,
            force_max_retries=options.bailian_force_max_retries,
            retry_interval=options.bailian_retry_interval,
            force_retry_interval=options.bailian_force_retry_interval,
        )
    )


def build_siliconflow_host_options(options: ProviderRuntimeOptions) -> SiliconFlowHostOptions:
    return SiliconFlowHostOptions(
        connection=_connection(
            user_agent=options.siliconflow_user_agent,
            max_retries=options.siliconflow_max_retries,
            force_max_retries=options.siliconflow_force_max_retries,
            retry_interval=options.siliconflow_retry_interval,
            force_retry_interval=options.siliconflow_force_retry_interval,
        ),
        force_official_endpoint=options.siliconflow_force_official_endpoint,
    )


def build_ark_host_options(options: ProviderRuntimeOptions) -> ArkHostOptions:
    return ArkHostOptions(
        connection=_connection(
            user_agent=options.volcengine_user_agent,
            max_retries=options.volcengine_max_retries,
            force_max_retries=options.volcengine_force_max_retries,
            retry_interval=options.volcengine_retry_interval,
            force_retry_interval=options.volcengine_force_retry_interval,
        ),
        force_official_endpoint=options.volcengine_force_official_endpoint,
        builtin_endpoint_mode=options.volcengine_builtin_endpoint_mode,
        prefix_cache_enabled=options.volcengine_prefix_cache_enabled,
        prefix_cache_ttl_seconds=options.volcengine_prefix_cache_ttl_seconds,
    )


def build_mimo_host_options(options: ProviderRuntimeOptions) -> MimoHostOptions:
    return MimoHostOptions(
        connection=_connection(
            user_agent=options.mimo_user_agent,
            max_retries=options.mimo_max_retries,
            force_max_retries=options.mimo_force_max_retries,
            retry_interval=options.mimo_retry_interval,
            force_retry_interval=options.mimo_force_retry_interval,
        ),
        reasoning_retention_days=options.mimo_reasoning_retention_days,
    )
