
from introspect.types import ChatHistory
from introspect.model import AbstractModel

class RequestCapture:
    def __init__(self, model: AbstractModel) -> None:
        self.duration: float = 0
        self._model = model

    async def __call__(self, history: ChatHistory) -> str:
        answer = await self._model.generate_text(history)
        self.duration += answer['duration']
        return answer['response'].strip()
