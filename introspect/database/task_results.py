
from ..types import ClassifyResult, IntrospectResult, FaithfulResult, TaskCategories
from ._result_dataset import ResultDatabase

class Classify(ResultDatabase[ClassifyResult]):
    task = TaskCategories.CLASSIFY

    _result_type = ClassifyResult
    _table_name = 'Classify'

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

class Importance(ResultDatabase[FaithfulResult]):
    task = TaskCategories.IMPORTANCE

    _result_type = FaithfulResult
    _table_name = 'importance'
