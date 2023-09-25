
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

from introspect.dataset import AbstractDataset
from introspect.model import AbstractModel

from ..types import DatasetCategories, TaskCategories, \
    Observation, TaskResult, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult

from ._request_capture import RequestCapture
from ._aggregator import AbstractAggregator, IntrospectAggregator, FaithfulAggregator

DatasetType = TypeVar('DatasetType', bound=AbstractDataset)
ObservationType = TypeVar('ObservationType', bound=Observation)
TaskResultType = TypeVar('TaskResultType', IntrospectResult, FaithfulResult)
PartialTaskResultType = TypeVar('PartialTaskResultType', PartialIntrospectResult, PartialFaithfulResult)

class AbstractTask(Generic[DatasetType, ObservationType, PartialTaskResultType, TaskResultType], metaclass=ABCMeta):
    _dataset: DatasetType
    dataset_category: DatasetCategories
    task_category: TaskCategories

    def __init__(self, dataset: DatasetType, model: AbstractModel, config: list[str] = []) -> None:
        self._dataset = dataset
        self._model = model
        self._config = set(config)

    def _is_enabled(self, option: str) -> bool:
        return option in self._config

    def _if_enabled(self, option: str, content: str) -> str:
        return content if self._is_enabled(option) else ''

    @abstractmethod
    def make_aggregator(self) -> AbstractAggregator:
        ...

    @abstractmethod
    async def _task(self, observation: ObservationType, capture: RequestCapture) -> PartialTaskResultType:
        ...

    @abstractmethod
    def _make_task_result(self, partial_result: PartialTaskResultType, default_result: TaskResult) -> TaskResultType:
        ...

    async def __call__(self, observation: ObservationType) -> TaskResultType:
        capture = RequestCapture(self._model)
        partial_result = await self._task(observation, capture)
        return self._make_task_result(partial_result, {
            'label': self._dataset.label_int2str[observation['label']],
            'duration': capture.duration
        })

class IntrospectTask(AbstractTask[DatasetType, ObservationType, PartialIntrospectResult, IntrospectResult]):
    def make_aggregator(self) -> IntrospectAggregator:
        return IntrospectAggregator()

    def _make_task_result(self, partial_result, default_result) -> IntrospectResult:
        return { **partial_result, **default_result }

class FaithfulTask(AbstractTask[DatasetType, ObservationType, PartialFaithfulResult, FaithfulResult]):
    def make_aggregator(self) -> FaithfulAggregator:
        return FaithfulAggregator()

    def _make_task_result(self, partial_result, default_result) -> FaithfulResult:
        return { **partial_result, **default_result }
