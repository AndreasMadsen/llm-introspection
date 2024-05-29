
from typing import TypedDict

from introspect.database import GenerationCache

from ..types import OfflineError, GenerateResponse
from ._abstract_client import AbstractClient

class OfflineInfo(TypedDict):
  pass

class OfflineClient(AbstractClient[OfflineInfo]):
    """This client does not connect to any server.

    This client allows the cache to be used, without connecting to a server.
    This is useful for debugging or refreshing the results.
    """
    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> OfflineInfo:
        return {}

    async def _generate(self, prompt, config) -> GenerateResponse:
        error = OfflineError('An LLM client is not used')
        error.add_note(f'The following prompt was not cached: {prompt}')
        raise error
