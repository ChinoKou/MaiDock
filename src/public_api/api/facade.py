from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

from ...i18n import format_validation_error, translate
from ..application.service import PublicApiService
from .requests import (
    CreateJobRequest,
    CreateUploadRequest,
    EmptyRequest,
    JobIdRequest,
    OneShotUploadRequest,
    ReadArtifactRequest,
    UploadIdRequest,
    WriteUploadChunkRequest,
)
from .responses import ApiResponseModel, ErrorData, error_response, success_response
from ..domain import PublicRpcObject
from ..errors import MediaApiError, public_media_error_message
from .definitions import PublicApiDefinition

_Command = TypeVar("_Command", bound=BaseModel)
_Response = TypeVar("_Response", bound=ApiResponseModel)


class PublicApiFacade:
    """SDK RPC 边界；不持有供应商分支，只编排 Command 和 Envelope。"""

    def __init__(self, service: PublicApiService) -> None:
        self._service = service

    def definitions(self) -> tuple[PublicApiDefinition, ...]:
        return (
            PublicApiDefinition(
                "media.capabilities", "public_api.description.get_media_capabilities", self.capabilities
            ),
            PublicApiDefinition("media.jobs.create", "public_api.description.submit_media_job", self.create_job),
            PublicApiDefinition("media.jobs.get", "public_api.description.get_media_job", self.get_job),
            PublicApiDefinition("media.jobs.cancel", "public_api.description.cancel_media_job", self.cancel_job),
            PublicApiDefinition("media.jobs.delete", "public_api.description.delete_media_job", self.delete_job),
            PublicApiDefinition(
                "media.uploads.create", "public_api.description.create_media_upload", self.create_upload
            ),
            PublicApiDefinition("media.uploads.upload", "public_api.description.upload_media", self.one_shot_upload),
            PublicApiDefinition("media.uploads.get", "public_api.description.get_media_upload", self.get_upload),
            PublicApiDefinition(
                "media.uploads.write_chunk", "public_api.description.write_media_upload_chunk", self.write_upload_chunk
            ),
            PublicApiDefinition(
                "media.uploads.complete", "public_api.description.complete_media_upload", self.complete_upload
            ),
            PublicApiDefinition(
                "media.uploads.delete", "public_api.description.delete_media_upload", self.delete_upload
            ),
            PublicApiDefinition(
                "media.artifacts.read", "public_api.description.read_media_artifact", self.read_artifact
            ),
        )

    async def capabilities(self, request: object) -> PublicRpcObject:
        return await self._call(request, EmptyRequest, self._capabilities)

    async def create_job(self, request: object) -> PublicRpcObject:
        return await self._call(request, CreateJobRequest, self._service.create_job)

    async def get_job(self, request: object) -> PublicRpcObject:
        return await self._call(request, JobIdRequest, self._service.get_job)

    async def cancel_job(self, request: object) -> PublicRpcObject:
        return await self._call(request, JobIdRequest, self._service.cancel_job)

    async def delete_job(self, request: object) -> PublicRpcObject:
        return await self._call(request, JobIdRequest, self._service.delete_job)

    async def create_upload(self, request: object) -> PublicRpcObject:
        return await self._call(request, CreateUploadRequest, self._service.create_upload)

    async def one_shot_upload(self, request: object) -> PublicRpcObject:
        return await self._call(request, OneShotUploadRequest, self._service.one_shot_upload)

    async def get_upload(self, request: object) -> PublicRpcObject:
        return await self._call(request, UploadIdRequest, self._service.get_upload)

    async def write_upload_chunk(self, request: object) -> PublicRpcObject:
        return await self._call(request, WriteUploadChunkRequest, self._service.write_upload_chunk)

    async def complete_upload(self, request: object) -> PublicRpcObject:
        return await self._call(request, UploadIdRequest, self._service.complete_upload)

    async def delete_upload(self, request: object) -> PublicRpcObject:
        return await self._call(request, UploadIdRequest, self._service.delete_upload)

    async def read_artifact(self, request: object) -> PublicRpcObject:
        return await self._call(request, ReadArtifactRequest, self._service.read_artifact)

    async def _call(
        self,
        request: object,
        command_type: type[_Command],
        operation: Callable[[_Command], Awaitable[_Response]],
    ) -> PublicRpcObject:
        try:
            command = command_type.model_validate(request)
            result = await operation(command)
            return success_response(result)
        except ValidationError as exc:
            return error_response(
                ErrorData(
                    code="INVALID_REQUEST",
                    message=format_validation_error(exc),
                )
            )
        except MediaApiError as exc:
            return error_response(
                ErrorData(
                    code=exc.code,
                    message=public_media_error_message(exc.code),
                    retryable=exc.retryable,
                    uncertain=exc.uncertain,
                    provider_request_id=exc.provider_request_id,
                )
            )
        except Exception:
            return error_response(ErrorData(code="INTERNAL_ERROR", message=translate("runtime.media_error.internal")))

    async def _capabilities(self, _command: EmptyRequest) -> ApiResponseModel:
        return await self._service.capabilities()
