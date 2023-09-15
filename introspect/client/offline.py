
from typing import TypedDict

from ..types import GenerateResponse, GenerateError
from ._abstract_client import AbstractClient

class OfflineInfo(TypedDict):
  pass

class OfflineResponse(GenerateResponse):
    pass

class OfflineError(GenerateError):
    pass

class OfflineClient(AbstractClient[OfflineInfo, OfflineResponse]):
    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> OfflineInfo:
       return {}

    async def _generate(self, prompt, config) -> OfflineResponse:
        raise OfflineError('An LLM client is not used')
