
__all__ = [
    'AbstractTask',
    'SentimentClassifyTask', 'SentimentAnswerableTask', 'SentimentCounterfactualTask', 'SentimentRedactedTask', 'SentimentImportanceTask',
    'MultiChoiceClassifyTask', 'MultiChoiceAnswerableTask', 'MultiChoiceCounterfactualTask', 'MultiChoiceRedactedTask', 'MultiChoiceImportanceTask',
    'EntailmentClassifyTask', 'EntailmentAnswerableTask', 'EntailmentCounterfactualTask', 'EntailmentRedactedTask', 'EntailmentImportanceTask',
    'tasks'
]

from typing import Type, Mapping

from ._abstract_tasks import AbstractTask
from .sentiment import SentimentClassifyTask, SentimentAnswerableTask, SentimentCounterfactualTask, SentimentRedactedTask, SentimentImportanceTask
from .multi_choice import MultiChoiceClassifyTask, MultiChoiceAnswerableTask, MultiChoiceCounterfactualTask, MultiChoiceRedactedTask, MultiChoiceImportanceTask
from .entailment import EntailmentClassifyTask, EntailmentAnswerableTask, EntailmentCounterfactualTask, EntailmentRedactedTask, EntailmentImportanceTask
from ..types import DatasetCategories, TaskCategories

tasks: Mapping[tuple[DatasetCategories, TaskCategories], Type[AbstractTask]] = {
    (Task.dataset_category, Task.task_category): Task
    for Task
    in [SentimentClassifyTask, SentimentAnswerableTask, SentimentCounterfactualTask, SentimentRedactedTask, SentimentImportanceTask,
        MultiChoiceClassifyTask, MultiChoiceAnswerableTask, MultiChoiceCounterfactualTask, MultiChoiceRedactedTask, MultiChoiceImportanceTask,
        EntailmentClassifyTask, EntailmentAnswerableTask, EntailmentCounterfactualTask, EntailmentRedactedTask, EntailmentImportanceTask]
}
