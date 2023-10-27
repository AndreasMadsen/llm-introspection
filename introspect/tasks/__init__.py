
__all__ = [
    'AbstractTask',
    'SentimentClassifyTask', 'SentimentAnswerableTask', 'SentimentCounterfactualTask', 'SentimentRedactedTask',
    'tasks'
]

from typing import Type, Mapping

from ._abstract_tasks import AbstractTask
from .sentiment import SentimentClassifyTask, SentimentAnswerableTask, SentimentCounterfactualTask, SentimentRedactedTask, SentimentImportanceTask
from ..types import DatasetCategories, TaskCategories

tasks: Mapping[tuple[DatasetCategories, TaskCategories], Type[AbstractTask]] = {
    (Task.dataset_category, Task.task_category): Task
    for Task
    in [SentimentClassifyTask, SentimentAnswerableTask, SentimentCounterfactualTask, SentimentRedactedTask, SentimentImportanceTask]
}
