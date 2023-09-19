
from ..types import AnswerableResult
from ._result_dataset import ResultDatabase

class Answerable(ResultDatabase[AnswerableResult]):
    _result_type = AnswerableResult
    _table_name = 'Answerable'
