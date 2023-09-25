
from ..types import IntrospectResult, FaithfulResult, TaskCategories
from ._result_dataset import ResultDatabase

class Answerable(ResultDatabase[IntrospectResult]):
    task = TaskCategories.ANSWERABLE

    _result_type = IntrospectResult
    _table_name = 'Answerable'

class Counterfactual(ResultDatabase[FaithfulResult]):
    task = TaskCategories.COUNTERFACTUAL

    _result_type = FaithfulResult
    _table_name = 'Counterfactual'

class Redacted(ResultDatabase[FaithfulResult]):
    task = TaskCategories.REDACTED

    _result_type = FaithfulResult
    _table_name = 'Redacted'
