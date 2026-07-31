from pydantic import BaseModel, ConfigDict

from ..domain import PublicJsonObject, VersionedOpaqueHandle


class OpaqueHandleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    driver_key: str
    payload_version: int
    payload: PublicJsonObject


class JobSerialization:
    """校验供应商 opaque handle 的持久化外壳。"""

    @staticmethod
    def handle_to_json(handle: VersionedOpaqueHandle) -> PublicJsonObject:
        record = OpaqueHandleRecord(
            driver_key=handle.driver_key,
            payload_version=handle.payload_version,
            payload=handle.payload,
        )
        return record.model_dump(mode="json")

    @staticmethod
    def handle_from_json(value: PublicJsonObject) -> VersionedOpaqueHandle:
        record = OpaqueHandleRecord.model_validate(value)
        return VersionedOpaqueHandle(
            driver_key=record.driver_key,
            payload_version=record.payload_version,
            payload=record.payload,
        )
