
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, TypedDict

from introspect.dataset._abstract_dataset import AbstractDataset
from introspect.model._abstract_model import AbstractModel

from ..types import DatasetCategories, AnswerableResult

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
    async def answerable(self, observation: ObservationType) -> AnswerableResult:
        ...
