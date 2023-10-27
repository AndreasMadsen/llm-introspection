
from typing import TypedDict

from introspect.database import GenerationCache
from introspect.types import GenerateResponse

from ._abstract_client import AbstractClient

class TestInfo(TypedDict):
  pass

class TestClient(AbstractClient[TestInfo]):
    log: list[str]
    response: dict[str, str]

    def __init__(self, response: dict[str, str] = {}, cache: GenerationCache | None = None) -> None:
       self.log = []
       self.response = response
       super().__init__('http://127.0.0.0:0', cache)

    async def _try_connect(self) -> bool:
        return True

    async def _info(self) -> TestInfo:
       return {}

    async def _generate(self, prompt, config) -> GenerateResponse:
        self.log.append(prompt)
        return {
           'response': self.response.get(prompt, f'[DEFAULT RESPONSE {len(self.log)}]'),
           'duration': 0
        }
