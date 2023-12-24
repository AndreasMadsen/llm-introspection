
from typing import TypedDict

from introspect.database import GenerationCache

from ..types import OfflineError, GenerateResponse
from ._abstract_client import AbstractClient

class OfflineInfo(TypedDict):
  pass

class OfflineClient(AbstractClient[OfflineInfo]):
    def __init__(self, base_url: str = 'http://127.0.0.0:0', cache: GenerationCache | None = None, connect_timeout_sec: int = 30 * 60) -> None:
        super().__init__(base_url, cache, connect_timeout_sec)

    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> OfflineInfo:
        return {}

    async def _generate(self, prompt, config) -> GenerateResponse:
        error = OfflineError('An LLM client is not used')
        error.add_note(f'The following prompt was not cached: {prompt}')
        raise error
