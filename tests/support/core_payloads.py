from copy import deepcopy
from json import load
from pathlib import Path
from typing import Literal

from src.core.json_types import JsonValue, normalize_json_value

from .assertions import as_json_object

type CoreContractFixture = Literal[
    "response_request.v1.json",
    "embedding_request.v1.json",
    "audio_transcription_request.v1.json",
    "provider_result.v1.json",
]

_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "core_contract"


def load_core_contract(fixture: CoreContractFixture) -> dict[str, JsonValue]:
    """加载一份脱敏 Core 契约快照。"""
    with (_FIXTURE_ROOT / fixture).open(encoding="utf-8") as fixture_file:
        return as_json_object(normalize_json_value(load(fixture_file)))


def _request_payload(fixture: CoreContractFixture, overrides: dict[str, JsonValue]) -> dict[str, JsonValue]:
    payload = deepcopy(load_core_contract(fixture))
    payload.update(overrides)
    return payload


def build_response_payload(**overrides: JsonValue) -> dict[str, JsonValue]:
    """构造 Core response 请求，并允许测试覆盖顶层字段。"""
    return _request_payload("response_request.v1.json", overrides)


def build_embedding_payload(**overrides: JsonValue) -> dict[str, JsonValue]:
    """构造 Core embedding 请求，并允许测试覆盖顶层字段。"""
    return _request_payload("embedding_request.v1.json", overrides)


def build_audio_transcription_payload(**overrides: JsonValue) -> dict[str, JsonValue]:
    """构造 Core audio_transcription 请求，并允许测试覆盖顶层字段。"""
    return _request_payload("audio_transcription_request.v1.json", overrides)


def load_provider_result_payload() -> dict[str, JsonValue]:
    """加载 Provider 返回值与 Runner RPC 包装快照。"""
    return deepcopy(load_core_contract("provider_result.v1.json"))
