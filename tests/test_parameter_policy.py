import pytest

from src.config import MaiDockConfig, build_runtime_options
from src.core.common import split_request_overrides
from src.core.parameter_catalog import (
    field_enabled_key,
    field_override_enabled_key,
    field_override_value_key,
    get_parameter_catalog,
)
from src.core.parameter_policy import (
    ParameterPolicy,
    apply_transport_parameter_policy,
    resolve_request_parameter_policy,
)
from src.schemas import ResponseRequestSnapshot


def _request(
    *,
    model_extra_params: dict | None = None,
    request_extra_params: dict | None = None,
) -> ResponseRequestSnapshot:
    return ResponseRequestSnapshot.model_validate(
        {
            "model_info": {
                "model_identifier": "test-model",
                "extra_params": model_extra_params or {},
            },
            "api_provider": {"api_key": "test-key"},
            "extra_params": request_extra_params or {},
        }
    )


def test_default_policy_preserves_shallow_model_then_request_merge() -> None:
    request = _request(
        model_extra_params={
            "headers": {"X-Model": "1"},
            "body": {"model_value": True},
            "top_p": 0.7,
        },
        request_extra_params={
            "body": {"request_value": True},
            "top_p": 0.8,
        },
    )

    resolved = resolve_request_parameter_policy(
        request,
        policy=ParameterPolicy(),
        provider_label="Test Provider",
        capability="response",
        direct_body_keys={"top_p"},
    ).extra_params

    assert resolved == {
        "headers": {"X-Model": "1"},
        "body": {"request_value": True},
        "top_p": 0.8,
    }
    overrides = split_request_overrides(resolved, direct_body_keys={"top_p"})
    assert overrides.extra_headers == {"X-Model": "1"}
    assert overrides.extra_body == {"request_value": True}
    assert overrides.direct_params == {"top_p": 0.8}


def test_request_mirror_does_not_bypass_disabled_model_extra_params() -> None:
    request = _request(
        model_extra_params={"top_p": 0.7, "temperature": 0.3},
        request_extra_params={"top_p": 0.7, "temperature": 0.4, "seed": 1},
    )

    resolved = resolve_request_parameter_policy(
        request,
        policy=ParameterPolicy(accept_model_extra_params=False),
        provider_label="Test Provider",
        capability="response",
        direct_body_keys={"top_p", "temperature", "seed"},
    ).extra_params

    assert resolved == {"temperature": 0.4, "seed": 1}


def test_disabled_paths_remove_raw_nested_params_before_strict_unknown_policy() -> None:
    request = _request(
        model_extra_params={
            "unknown_field": 30,
            "headers": {"X-Debug": "1"},
            "body": {"parameters": {"enable_thinking": True, "top_p": 0.9}},
        }
    )

    resolved = resolve_request_parameter_policy(
        request,
        policy=ParameterPolicy(
            disabled_paths=(
                "unknown_field",
                "headers.X-Debug",
                "body.parameters.enable_thinking",
            ),
            unknown_extra_params="forward",
        ),
        provider_label="DashScope",
        capability="chat_completion",
    ).extra_params

    assert resolved == {"headers": {}, "body": {"parameters": {"top_p": 0.9}}}


def test_rejected_paths_include_provider_capability_and_path() -> None:
    request = _request(model_extra_params={"headers": {"Authorization": "secret"}})

    with pytest.raises(
        ValueError,
        match="Test Provider.*response.*headers.Authorization",
    ):
        resolve_request_parameter_policy(
            request,
            policy=ParameterPolicy(rejected_paths=("headers.Authorization",)),
            provider_label="Test Provider",
            capability="response",
        )


def test_defaults_are_low_priority_and_overrides_are_high_priority() -> None:
    request = _request(model_extra_params={"top_p": 0.8, "body": {"reasoning": {"effort": "high"}}})

    resolved = resolve_request_parameter_policy(
        request,
        policy=ParameterPolicy(
            default_params={"top_p": 0.3, "seed": 1},
            override_params={"top_p": 0.5, "body": {"reasoning": {"effort": "low"}}},
        ),
        provider_label="Test Provider",
        capability="response",
        direct_body_keys={"top_p", "seed"},
    ).extra_params

    assert resolved == {
        "top_p": 0.5,
        "seed": 1,
        "body": {"reasoning": {"effort": "low"}},
    }


def test_catalog_field_controls_drop_and_override_host_params() -> None:
    catalog = get_parameter_catalog("openai_responses", "response")
    top_p = catalog.field_by_safe_key("top_p")
    reasoning = catalog.field_by_safe_key("reasoning")
    assert top_p is not None
    assert reasoning is not None
    config = MaiDockConfig.model_validate(
        {
            "openai_responses": {
                "response": {
                    "fields": {
                        field_enabled_key(reasoning): False,
                        field_override_enabled_key(top_p): True,
                        field_override_value_key(top_p): "0.45",
                    }
                }
            }
        }
    )
    request = _request(model_extra_params={"top_p": 0.9, "reasoning": {"effort": "high"}})
    policy = build_runtime_options(config).parameter_policies.get("openai_responses", "response")

    from src.providers.common.parameter_translation import (
        build_normalized_host_parameters,
    )

    normalized = build_normalized_host_parameters(
        request,
        policy=policy,
        catalog=catalog,
        provider_label="OpenAI Responses",
        capability="response",
    )

    # reasoning field disabled → dropped from normalized
    assert "reasoning" not in normalized.fields
    # top_p override not applied at normalized level (happens at transport level)
    assert normalized.fields["top_p"] == 0.9
    assert normalized.sources["top_p"] == "model_info.extra_params.top_p"


def test_unknown_extra_params_drop_reject_and_override_passthrough() -> None:
    request = _request(model_extra_params={"future_host_param": True})

    dropped = resolve_request_parameter_policy(
        request,
        policy=ParameterPolicy(unknown_extra_params="drop", override_params={"future_config_param": "ok"}),
        provider_label="Test Provider",
        capability="response",
        direct_body_keys={"top_p"},
    ).extra_params
    assert dropped == {"future_config_param": "ok"}

    with pytest.raises(ValueError, match="future_host_param"):
        resolve_request_parameter_policy(
            request,
            policy=ParameterPolicy(unknown_extra_params="reject"),
            provider_label="Test Provider",
            capability="response",
            direct_body_keys={"top_p"},
        )


def test_final_transport_policy_removes_and_overrides_body_headers_query() -> None:
    transport = apply_transport_parameter_policy(
        body={
            "temperature": 0.3,
            "parameters": {"enable_thinking": True, "top_p": 0.9},
        },
        headers={"X-Debug": "1"},
        query={"debug": True},
        policy=ParameterPolicy(
            disabled_paths=(
                "body.temperature",
                "body.parameters.enable_thinking",
                "headers.X-Debug",
            ),
            override_params={
                "body": {"temperature": 0.2},
                "headers": {"X-Forced": "yes"},
                "query": {"trace": "on"},
            },
        ),
        provider_label="Test Provider",
        capability="response",
    )

    assert transport.body == {"parameters": {"top_p": 0.9}, "temperature": 0.2}
    assert transport.headers == {"X-Forced": "yes"}
    assert transport.query == {"debug": True, "trace": "on"}


def test_final_transport_policy_rejects_non_string_header_overrides() -> None:
    with pytest.raises(
        TypeError,
        match="parameter_policy.override_params.headers.X-Number",
    ):
        apply_transport_parameter_policy(
            body={},
            headers={},
            query={},
            policy=ParameterPolicy(override_params={"headers": {"X-Number": 1}}),
            provider_label="Test Provider",
            capability="response",
        )
