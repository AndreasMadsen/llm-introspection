
from typing import TypedDict

from introspect.database import GenerationCache
from introspect.types import GenerateResponse

from ._abstract_client import AbstractClient

class TestInfo(TypedDict):
  pass

class TestClient(AbstractClient[TestInfo]):
    log: list[str]
    response: dict[str, str|None]

    def __init__(self, response: dict[str, str|None] = {}, cache: GenerationCache | None = None) -> None:
       self.log = []
       self.response = response
       super().__init__('http://127.0.0.0:0', cache)

    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> TestInfo:
       return {}

    async def _generate(self, prompt, config) -> GenerateResponse:
        self.log.append(prompt)
        response = self.response.get(prompt, None)
        if response is None:
         response = f'[DEFAULT RESPONSE {len(self.log)}]'
        return {
           'response': response,
           'duration': 0
        }
