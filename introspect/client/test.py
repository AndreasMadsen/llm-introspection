
from typing import TypedDict, Callable, Awaitable, Iterable

from introspect.database import GenerationCache
from introspect.types import GenerateConfig, GenerateResponse

from ._abstract_client import AbstractClient

class TestInfo(TypedDict):
  pass

def _dict_to_callable(response: dict[str, str|None]) -> Callable[[str], Awaitable[str|None]]:
    async def fn(prompt: str) -> str|None:
        return response.get(prompt, None)
    return fn

class TestClient(AbstractClient[TestInfo]):
    _response: Callable[[str], Awaitable[str|None]]

    def __init__(self, response: Callable[[str], Awaitable[str|None]]|dict[str, str|None] = {},
                       cache: GenerationCache | None = None) -> None:
        self.log = []

        if isinstance(response, dict):
            self._response = _dict_to_callable(response)
        else:
            self._response = response
        super().__init__('http://127.0.0.0:0', cache, record=True)

    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> TestInfo:
       return {}

    async def _generate(self, prompt, config) -> GenerateResponse:
        response = await self._response(prompt)
        if response is None:
            response = f'[DEFAULT RESPONSE {len(self.log)}]'
        return {
           'response': response,
           'duration': 0
        }
