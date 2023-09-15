
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, TypedDict

from introspect.dataset import AbstractDataset
from introspect.model import AbstractModel

from ..types import DatasetCategories, AnswerableResult, PartialAnswerableResult
from ._request_capture import RequestCapture, request_capture_scope

DatasetType = TypeVar('DatasetType', bound=AbstractDataset)
ObservationType = TypeVar('ObservationType', bound=TypedDict)

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

        with request_capture_scope(self._model) as capture:
            answer = await self._answerable(observation, capture)

        return {
            **answer,
            'duration': capture.duration,
            'error': capture.error,
        }
