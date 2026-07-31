from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..domain import PublicRpcObject

type PublicApiHandler = Callable[[object], Awaitable[PublicRpcObject]]


@dataclass(frozen=True, slots=True)
class PublicApiDefinition:
    name: str
    description_key: str
    handler: PublicApiHandler
