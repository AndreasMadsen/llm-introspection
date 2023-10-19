
__all__ = ['GenerationCache', 'Answerable', 'Counterfactual', 'Redacted', 'result_databases']

from typing import Type, Mapping

from ..types import TaskCategories

from .generation_cache import GenerationCache
from ._result_dataset import ResultDatabase
from .task_results import Classify, Answerable, Counterfactual, Redacted, Important

result_databases: Mapping[TaskCategories, Type[ResultDatabase]] = {
    Database.task: Database
    for Database
    in [Classify, Answerable, Counterfactual, Redacted, Important]
}
