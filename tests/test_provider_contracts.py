from json import dumps, loads

from src.schemas import (
    ProviderFunctionCall,
    ProviderResponse,
    ProviderToolCall,
    ProviderUsage,
)
from tests.support.assertions import as_json_object
from tests.support.core_payloads import load_provider_result_payload


def test_provider_result_fixture_matches_runner_rpc_wrapper() -> None:
    fixture = load_provider_result_payload()
    provider_result = as_json_object(fixture["provider_result"])
    rpc_envelope = as_json_object(fixture["rpc_envelope"])

    assert rpc_envelope["success"] is True
    assert rpc_envelope["result"] == provider_result


def test_provider_response_validates_complete_core_return_contract() -> None:
    fixture = load_provider_result_payload()
    response = ProviderResponse.model_validate(as_json_object(fixture["provider_result"]))

    assert response.content == "脱敏响应正文"
    assert response.reasoning_content == "脱敏推理摘要"
    assert response.embedding == [0.125, -0.25, 0.5]
    assert response.tool_calls == [
        ProviderToolCall(
            id="call-contract-1",
            function=ProviderFunctionCall(name="lookup_weather", arguments={"city": "上海"}),
            extra_content={"provider_item_id": "item-redacted"},
        )
    ]
    assert response.usage == ProviderUsage(
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        prompt_cache_hit_tokens=5,
        prompt_cache_miss_tokens=7,
    )
    assert response.raw_data == {
        "id": "response-redacted",
        "model": "contract-response-model",
        "status": "completed",
    }


def test_provider_response_round_trips_through_json_without_contract_loss() -> None:
    fixture = load_provider_result_payload()
    response = ProviderResponse.model_validate(as_json_object(fixture["provider_result"]))
    serialized = dumps(response.to_host_dict(), ensure_ascii=False)
    restored = ProviderResponse.model_validate(loads(serialized))

    assert restored.to_host_dict() == response.to_host_dict()
    assert restored.model_dump_json() == response.model_dump_json()


def test_provider_response_default_factories_are_isolated() -> None:
    first = ProviderResponse()
    second = ProviderResponse()
    first.tool_calls.append(
        ProviderToolCall(
            id="call_1",
            function=ProviderFunctionCall(name="lookup", arguments={"q": "first"}),
        )
    )
    first.usage.prompt_tokens = 7

    assert second.tool_calls == []
    assert second.usage == ProviderUsage()


def test_provider_tool_call_default_factories_are_isolated() -> None:
    first_function = ProviderFunctionCall(name="lookup")
    second_function = ProviderFunctionCall(name="lookup")
    first_call = ProviderToolCall(id="call_1", function=first_function)
    second_call = ProviderToolCall(id="call_2", function=second_function)
    first_function.arguments["query"] = "first"
    first_call.extra_content["trace"] = "first"

    assert second_function.arguments == {}
    assert second_call.extra_content == {}


def test_provider_response_to_host_dict_excludes_none_and_uses_json_values() -> None:
    response = ProviderResponse(content=None, reasoning_content=None, embedding=None, raw_data=None)

    assert response.to_host_dict() == {
        "tool_calls": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        },
    }
