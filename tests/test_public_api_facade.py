import pytest

from src.public_api.api.facade import PublicApiFacade
from src.public_api.api.responses import CapabilityData, ErrorEnvelope
from src.public_api.application.service import PublicApiService


class FakeService(PublicApiService):
    def __init__(self) -> None:
        pass

    async def capabilities(self) -> CapabilityData:
        return CapabilityData(models=(), profiles=())


@pytest.mark.asyncio
async def test_resource_definitions_use_single_request_and_uniform_envelopes() -> None:
    facade = PublicApiFacade(FakeService())
    definitions = facade.definitions()
    assert [definition.name for definition in definitions] == [
        "media.capabilities",
        "media.jobs.create",
        "media.jobs.get",
        "media.jobs.cancel",
        "media.jobs.delete",
        "media.uploads.create",
        "media.uploads.upload",
        "media.uploads.get",
        "media.uploads.write_chunk",
        "media.uploads.complete",
        "media.uploads.delete",
        "media.artifacts.read",
    ]

    capabilities = await facade.capabilities({})
    assert capabilities == {
        "ok": True,
        "data": {"models": [], "profiles": [], "default_image_profile": None, "default_video_profile": None},
        "error": None,
    }

    for definition in definitions[1:]:
        result = await definition.handler({"unexpected": True})
        error = ErrorEnvelope.model_validate(result)
        assert error.ok is False
        assert error.data is None
        assert error.error.code == "INVALID_REQUEST"
