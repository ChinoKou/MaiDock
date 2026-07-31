from src.core.common import ProviderRuntimeOptions
from src.host_adapters.common.payloads import raw_data_or_none


def test_raw_data_or_none_sanitizes_when_enabled() -> None:
    raw_data = raw_data_or_none(
        {"api_key": "secret-key", "model": "m"},
        options=ProviderRuntimeOptions(include_raw_data=True),
    )

    assert raw_data == {"api_key": "***", "model": "m"}
    assert raw_data_or_none({"model": "m"}, options=ProviderRuntimeOptions(include_raw_data=False)) is None
