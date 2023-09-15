
from contextlib import contextmanager
from introspect.types import ChatHistory, GenerateError
from introspect.model import AbstractModel

class RequestCapture:
    def __init__(self, model: AbstractModel) -> None:
        self.duration: float = 0
        self.error: None|GenerateError = None
        self._model = model

    async def __call__(self, history: ChatHistory) -> str:
        answer = await self._model.generate_text(history)
        self.duration += answer['duration']
        return answer['response'].strip()

@contextmanager
def request_capture_scope(model: AbstractModel):
    capture = RequestCapture(model)
    try:
        yield capture
    except GenerateError as error:
        capture.error = error
