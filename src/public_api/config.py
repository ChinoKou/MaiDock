from math import isfinite
from typing import Literal, Self
from urllib.parse import urlsplit
import json

from maibot_sdk import Field, PluginConfigBase
from pydantic import field_validator, model_validator

from .domain import MediaCapability, PublicJsonObject, PublicJsonValue
from .domain.json_types import normalize_public_json

type DashScopeProtocolFamily = Literal[
    "dashscope_multimodal_generation",
    "dashscope_image_generation",
    "dashscope_text2image_synthesis",
    "dashscope_image2image_synthesis",
    "dashscope_video_generation",
]
type ArkProtocolFamily = Literal[
    "ark_images_generations",
    "ark_content_generation_tasks",
]
type PublicParameterValueType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "json",
    "null",
]


def parse_parameter_value(value_type: PublicParameterValueType, value: str) -> PublicJsonValue:
    """按 WebUI 选定类型解析参数值，拒绝隐式类型转换。"""

    if value_type == "string":
        return value
    if value_type == "boolean":
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError("boolean 参数值只能是 true 或 false")
    if value_type == "null":
        if value:
            raise ValueError("null 参数值必须留空")
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{value_type} 参数值不是有效 JSON") from exc
    if value_type == "integer":
        if type(parsed) is not int:
            raise ValueError("integer 参数值必须是 JSON 整数")
        return parsed
    if value_type == "number":
        if isinstance(parsed, bool) or not isinstance(parsed, int | float):
            raise ValueError("number 参数值必须是 JSON 数字")
        try:
            number = float(parsed)
        except OverflowError as exc:
            raise ValueError("number 参数值必须是有限数字") from exc
        if not isfinite(number):
            raise ValueError("number 参数值必须是有限数字")
        return number
    if not isinstance(parsed, dict | list):
        raise ValueError("json 参数值必须是 JSON object 或 array")
    try:
        return normalize_public_json(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("json 参数值必须只包含有限 JSON 值") from exc


class PublicParameterEntryConfig(PluginConfigBase):
    """WebUI 中一个可读、可编辑的高级参数条目。"""

    name: str = Field(min_length=1)
    value_type: PublicParameterValueType = "string"
    value: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("value", mode="before")
    @classmethod
    def require_string_value(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("参数值必须是字符串")
        return value

    @model_validator(mode="after")
    def validate_typed_value(self) -> Self:
        parse_parameter_value(self.value_type, self.value)
        return self

    def parsed_value(self) -> PublicJsonValue:
        return parse_parameter_value(self.value_type, self.value)


def parameter_entries_to_object(entries: list[PublicParameterEntryConfig]) -> PublicJsonObject:
    """把参数条目转换为供应商请求使用的 JSON object。"""

    parameters: PublicJsonObject = {}
    for entry in entries:
        if entry.name in parameters:
            raise ValueError(f"参数列表存在重复名称: {entry.name}")
        parameters[entry.name] = entry.parsed_value()
    return parameters


def validate_parameter_entries(
    entries: list[PublicParameterEntryConfig],
) -> list[PublicParameterEntryConfig]:
    parameter_entries_to_object(entries)
    return entries


class PublicApiResourceConfig(PluginConfigBase):
    """跨插件 API 的通用队列、存储和保留期限制。"""

    max_concurrent_jobs: int = Field(default=2, ge=1, le=32)
    max_queued_jobs: int = Field(default=32, ge=1, le=1024)
    max_upload_mb: int = Field(default=512, ge=1, le=4096)
    max_artifact_mb: int = Field(default=512, ge=1, le=4096)
    storage_quota_gb: int = Field(default=10, ge=1, le=1024)
    incomplete_upload_ttl_hours: int = Field(default=24, ge=1, le=168)
    completed_upload_ttl_days: int = Field(default=7, ge=1, le=90)
    artifact_ttl_days: int = Field(default=7, ge=1, le=90)
    job_metadata_ttl_days: int = Field(default=30, ge=1, le=365)
    max_tracking_hours: int = Field(default=23, ge=1, le=72)


class DashScopeProtocolRouteConfig(PluginConfigBase):
    """DashScope 模型、模式与精确协议资源的显式路由。"""

    capability: MediaCapability
    model: str = Field(min_length=1)
    mode: str = ""
    protocol_family: DashScopeProtocolFamily

    @field_validator("model", "mode", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class DashScopePublicProfileConfig(PluginConfigBase):
    """Public API 使用的独立 DashScope 连接与参数配置。"""

    name: str = Field(min_length=1)
    api_key: str = Field(min_length=1, repr=False)
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    workspace_id: str = ""
    default_image_model: str = ""
    default_video_model: str = ""
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    request_timeout_seconds: float = Field(default=1800.0, gt=0, le=82800)
    safe_max_retries: int = Field(default=3, ge=0, le=10)
    retry_interval_seconds: float = Field(default=1.0, ge=0, le=30)
    image_default_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    image_override_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    video_default_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    video_override_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    protocol_routes: list[DashScopeProtocolRouteConfig] = Field(default_factory=list)

    @field_validator(
        "name",
        "api_key",
        "base_url",
        "workspace_id",
        "default_image_model",
        "default_video_model",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/api/v1")
        ):
            raise ValueError(
                "DashScope Public API base_url 必须是以 /api/v1 结尾且不含凭据/query/fragment 的 HTTPS URL"
            )
        return normalized

    @field_validator(
        "image_default_parameters",
        "image_override_parameters",
        "video_default_parameters",
        "video_override_parameters",
        mode="after",
    )
    @classmethod
    def validate_parameter_list(
        cls,
        value: list[PublicParameterEntryConfig],
    ) -> list[PublicParameterEntryConfig]:
        return validate_parameter_entries(value)

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        seen: set[tuple[MediaCapability, str, str]] = set()
        for route in self.protocol_routes:
            key = (route.capability, route.model, route.mode)
            if key in seen:
                raise ValueError(f"protocol_routes 存在重复路由: {key}")
            seen.add(key)
        return self


class DashScopePublicConfig(PluginConfigBase):
    """DashScope Public API 供应商配置。"""

    profiles: list[DashScopePublicProfileConfig] = Field(default_factory=list)


class ArkProtocolRouteConfig(PluginConfigBase):
    """Volcengine ARK 模型、模式与精确协议资源的显式路由。"""

    capability: MediaCapability
    model: str = Field(min_length=1)
    mode: str = ""
    protocol_family: ArkProtocolFamily

    @field_validator("model", "mode", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ArkPublicProfileConfig(PluginConfigBase):
    """Public API 使用的独立 Volcengine ARK 连接与参数配置。

    字段集与 DashScope 一致，只是没有 workspace_id——ARK 的多租户靠 API Key 本身区分。
    """

    name: str = Field(min_length=1)
    api_key: str = Field(min_length=1, repr=False)
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    default_image_model: str = ""
    default_video_model: str = ""
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    request_timeout_seconds: float = Field(default=1800.0, gt=0, le=82800)
    safe_max_retries: int = Field(default=3, ge=0, le=10)
    retry_interval_seconds: float = Field(default=1.0, ge=0, le=30)
    image_default_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    image_override_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    video_default_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    video_override_parameters: list[PublicParameterEntryConfig] = Field(default_factory=list)
    protocol_routes: list[ArkProtocolRouteConfig] = Field(default_factory=list)

    @field_validator(
        "name",
        "api_key",
        "base_url",
        "default_image_model",
        "default_video_model",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/api/v3")
        ):
            raise ValueError(
                "Volcengine ARK Public API base_url 必须是以 /api/v3 结尾且不含凭据/query/fragment 的 HTTPS URL"
            )
        return normalized

    @field_validator(
        "image_default_parameters",
        "image_override_parameters",
        "video_default_parameters",
        "video_override_parameters",
        mode="after",
    )
    @classmethod
    def validate_parameter_list(
        cls,
        value: list[PublicParameterEntryConfig],
    ) -> list[PublicParameterEntryConfig]:
        return validate_parameter_entries(value)

    @model_validator(mode="after")
    def validate_routes(self) -> Self:
        seen: set[tuple[MediaCapability, str, str]] = set()
        for route in self.protocol_routes:
            key = (route.capability, route.model, route.mode)
            if key in seen:
                raise ValueError(f"protocol_routes 存在重复路由: {key}")
            seen.add(key)
        return self


class ArkPublicConfig(PluginConfigBase):
    """Volcengine ARK Public API 供应商配置。"""

    profiles: list[ArkPublicProfileConfig] = Field(default_factory=list)


class PublicApiConfig(PluginConfigBase):
    """跨插件 Public API 的公共配置与供应商配置入口。"""

    enabled: bool = False
    default_image_profile: str = ""
    default_video_profile: str = ""
    resources: PublicApiResourceConfig = Field(default_factory=PublicApiResourceConfig)
    dashscope: DashScopePublicConfig = Field(default_factory=DashScopePublicConfig)
    volcengine_ark: ArkPublicConfig = Field(default_factory=ArkPublicConfig)

    @field_validator("default_image_profile", "default_video_profile", mode="before")
    @classmethod
    def strip_default_profile(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_profile_names(self) -> Self:
        # 唯一性必须跨供应商合并判断：Public API 的调用方只按 profile 名寻址，
        # 不带供应商前缀，所以 dashscope 和 volcengine_ark 各配一个同名 profile
        # 会让请求落到哪一家变得不确定。
        names = [
            *(profile.name for profile in self.dashscope.profiles),
            *(profile.name for profile in self.volcengine_ark.profiles),
        ]
        if len(names) != len(set(names)):
            raise ValueError("public_api 的 Profile name 必须全局唯一")
        return self
