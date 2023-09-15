
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Generic, TypedDict, Self
from traceback import TracebackException
from contextlib import contextmanager

from introspect.dataset._abstract_dataset import AbstractDataset
from introspect.model._abstract_model import AbstractModel
from introspect.types import ChatHistory, GenerateError

from ..types import DatasetCategories, AnswerableResult, PartialAnswerableResult

DatasetType = TypeVar('DatasetType', bound=AbstractDataset)
ObservationType = TypeVar('ObservationType', bound=TypedDict)

class RequestCapture:
    def __init__(self, model: AbstractModel) -> None:
        self.duration: float = 0
        self.error: None|GenerateError = None
        self._model = model

    async def __call__(self, history: ChatHistory) -> str:
        answer = await self._model.generate_text(history)
        self.duration += answer['duration']
        return answer['response']

@contextmanager
def _request_capture_scope(model: AbstractModel):
    capture = RequestCapture(model)
    try:
        yield capture
    except GenerateError as error:
        capture.error = error

class AbstractTasks(Generic[DatasetType, ObservationType], metaclass=ABCMeta):
    _category: DatasetCategories
    _dataset: DatasetType

    def __init__(self, dataset: DatasetType, model: AbstractModel) -> None:
        if dataset.category != self._category:
            raise ValueError(f'dataset has category "{dataset.category}", but expected "{self._category}"')

        self._dataset = dataset
        self._model = model

    @abstractmethod
    async def _answerable(self, observation: ObservationType, capture: RequestCapture) -> PartialAnswerableResult:
        pass

    async def answerable(self, observation: ObservationType) -> AnswerableResult:
        answer: PartialAnswerableResult = {
            "answer_ability": None,
            "answer_sentiment": None,
            "introspect": None,
            "correct": None
        }

        with _request_capture_scope(self._model) as capture:
            answer = await self._answerable(observation, capture)

        return {
            **answer,
            'duration': capture.duration,
            'error': capture.error,
        }
