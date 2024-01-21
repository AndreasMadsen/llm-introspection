
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Sequence, Literal
from functools import cached_property

from introspect.dataset import AbstractDataset
from introspect.model import AbstractModel

from ..types import DatasetCategories, TaskCategories, \
    Observation, TaskResult, \
    PartialClassifyResult, ClassifyResult, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult

from ._request_capture import RequestCapture
from ._aggregator import AbstractAggregator, ClassifyAggregator, IntrospectAggregator, FaithfulAggregator

DatasetType = TypeVar('DatasetType', bound=AbstractDataset)
ObservationType = TypeVar('ObservationType', bound=Observation)
TaskResultType = TypeVar('TaskResultType', ClassifyResult, IntrospectResult, FaithfulResult)
PartialTaskResultType = TypeVar('PartialTaskResultType', PartialClassifyResult, PartialIntrospectResult, PartialFaithfulResult)

XstrType = TypeVar('XstrType', bound=str)
YstrType = TypeVar('YstrType', bound=str)

class AbstractTask(Generic[DatasetType, ObservationType, PartialTaskResultType, TaskResultType], metaclass=ABCMeta):
    dataset_category: DatasetCategories
    task_category: TaskCategories

    def __init__(self, model: AbstractModel, config: Sequence[str] = []) -> None:
        """Enables running a specific task.

        Each task is categorized by it's generalized dataset (e.g. SentimentDataset) and
        a task (answerable, counterfactual, or redacted). In addition to the overall
        task, additional configurations can be provided with the config option.

        A task will query the provided model, possibly several times. However, the
        queries are resicted to one query at a time. This is such that higher level
        paralization routines, can make accuate assumptions about the number of
        parallel queries.

        Args:
            model (AbstractModel): The model which is used to query prompts.
            config (Sequence[str], optional): Additional configurations. These options will
                make minor modifications to the prompts. Defaults to [].
        """
        self._model = model
        self._config = set(config)

    @cached_property
    def _mask_special_token(self) -> Literal['[REMOVED]', '[REDACTED]']:
        return self._ifelse_enabled('m-removed', '[REMOVED]', '[REDACTED]')

    def _is_enabled(self, option: str) -> bool:
        return option in self._config

    def _if_enabled(self, option: str, content: XstrType) -> XstrType|Literal['']:
        return content if self._is_enabled(option) else ''

    def _ifelse_enabled(self, option: str, true: XstrType, false: YstrType) -> XstrType|YstrType:
        return true if self._is_enabled(option) else false

    @abstractmethod
    def make_aggregator(self) -> AbstractAggregator:
        """Creates an aggregator, for collecting multiple task results.

        Example:
            agg = task.make_aggregator()
            for obs in tqdm(dataset, desc=agg.progress_description):
                answer = await task(obs)
                agg.add_answer(answer)
                pbar.set_description(agg.progress_description)
            print(agg.total_duration)
            print(agg.results)

        Returns:
            AbstractAggregator: aggregator object which collects task response statistics
        """
        ...

    @abstractmethod
    async def _task(self, observation: ObservationType, capture: RequestCapture) -> PartialTaskResultType:
        ...

    @abstractmethod
    def _make_task_result(self, partial_result: PartialTaskResultType, default_result: TaskResult) -> TaskResultType:
        ...

    async def __call__(self, observation: ObservationType) -> TaskResultType:
        """Run the task on a specific observation

        Args:
            observation (Observation): The dataset obsercation

        Returns:
            IntrospectResult | FaithfulResult: the task response.
        """
        capture = RequestCapture(self._model)
        partial_result = await self._task(observation, capture)
        return self._make_task_result(partial_result, {
            'label': observation['label'],
            'duration': capture.duration
        })

class ClassifyTask(AbstractTask[DatasetType, ObservationType, PartialClassifyResult, ClassifyResult]):
    def make_aggregator(self) -> ClassifyAggregator:
        return ClassifyAggregator()

    def _make_task_result(self, partial_result, default_result) -> ClassifyResult:
        return { **partial_result, **default_result }

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
