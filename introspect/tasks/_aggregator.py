
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Hashable, Literal
from collections import defaultdict

from ..types import GenerateError, \
    ClassifyAggregateAnswer, ClassifyAggregateResult, ClassifyResult, \
    IntrospectAggregateAnswer, IntrospectAggregateResult, IntrospectResult, \
    FaithfulAggregateAnswer, FaithfulAggregateResult, FaithfulResult

ResultAnswerType = TypeVar('ResultAnswerType', ClassifyResult, IntrospectResult, FaithfulResult)
AggregateAnswerType = TypeVar('AggregateAnswerType', ClassifyAggregateAnswer, IntrospectAggregateAnswer, FaithfulAggregateAnswer)
FeatureColumnType = TypeVar('FeatureColumnType', bound=str)

class TableCounter(Generic[FeatureColumnType, AggregateAnswerType, ResultAnswerType]):
    _feature_columns: tuple[FeatureColumnType, ...]
    _counts: dict[tuple[Hashable, ...], int]

    def __init__(self, feature_columns: tuple[FeatureColumnType, ...]) -> None:
        self._counts = defaultdict(int)
        self._feature_columns = feature_columns

    def increment(self, observation: ResultAnswerType) -> None:
        key = tuple(observation.get(c) for c in self._feature_columns)
        self._counts[key] += 1

    def as_table(self) -> list[AggregateAnswerType]:
        table: list[AggregateAnswerType] = []
        for column_values, count in self._counts.items():
            row: dict[str, Hashable] = { 'count': count }
            for column_name, column_value in zip(self._feature_columns, column_values):
                row[column_name] = column_value
            table.append(row) # type: ignore

        return table

AggregateResultType = TypeVar('AggregateResultType', ClassifyAggregateResult, IntrospectAggregateResult, FaithfulAggregateResult)

class AbstractAggregator(Generic[ResultAnswerType, AggregateResultType], metaclass=ABCMeta):
    _duration: float

    def __init__(self) -> None:
        self._duration = 0
        self._error_count = 0
        self._total_count = 0

    @abstractmethod
    def _add_answer(self, answer: ResultAnswerType):
        ...

    def add_answer(self, answer: ResultAnswerType|GenerateError) -> None:
        self._total_count += 1

        if isinstance(answer, GenerateError):
            self._error_count += 1
            return

        self._duration += answer['duration']
        self._add_answer(answer)

    @property
    @abstractmethod
    def progress_description(self) -> str:
        ...

    @property
    def total_duration(self) -> float:
        return self._duration

    @property
    @abstractmethod
    def results(self) -> AggregateResultType:
        ...

class ClassifyAggregator(AbstractAggregator[ClassifyResult, ClassifyAggregateResult]):
    _answer_counts: TableCounter[Literal['label', 'sentiment'], ClassifyAggregateAnswer, ClassifyResult]

    def __init__(self) -> None:
        super().__init__()

        self._answer_counts = TableCounter(('label', 'sentiment'))
        self._correct_count = 0
        self._missmatch_count = 0

    def _add_answer(self, answer: IntrospectResult):
        if answer['correct'] is None:
            self._missmatch_count += 1
        else:
            self._correct_count += answer['correct']
            self._answer_counts.increment(answer)

    @property
    def progress_description(self) -> str:
        return f'Processing[C={self._correct_count}, M={self._missmatch_count}, E={self._error_count}]'

    @property
    def results(self) -> ClassifyAggregateResult:
        return {
            'answer': self._answer_counts.as_table(),
            'correct': self._correct_count,
            'missmatch': self._missmatch_count,
            'error': self._error_count,
            'total': self._total_count
        }

class IntrospectAggregator(AbstractAggregator[IntrospectResult, IntrospectAggregateResult]):
    _answer_counts: TableCounter[Literal['label', 'predict', 'ability'], IntrospectAggregateAnswer, IntrospectResult]

    def __init__(self) -> None:
        super().__init__()

        self._answer_counts = TableCounter(('label', 'predict', 'ability'))
        self._introspect_count = 0
        self._correct_count = 0
        self._missmatch_count = 0

    def _add_answer(self, answer: IntrospectResult):
        if answer['introspect'] is None or answer['correct'] is None:
            self._missmatch_count += 1
        else:
            self._introspect_count += answer['introspect']
            self._correct_count += answer['correct']
            self._answer_counts.increment(answer)

    @property
    def progress_description(self) -> str:
        return f'Processing[C={self._correct_count}, I={self._introspect_count}, M={self._missmatch_count}, E={self._error_count}]'

    @property
    def results(self) -> IntrospectAggregateResult:
        return {
            'answer': self._answer_counts.as_table(),
            'introspect': self._introspect_count,
            'correct': self._correct_count,
            'missmatch': self._missmatch_count,
            'error': self._error_count,
            'total': self._total_count
        }

class FaithfulAggregator(AbstractAggregator[FaithfulResult, FaithfulAggregateResult]):
    _answer_counts: TableCounter[Literal['label', 'predict', 'explain_predict'], FaithfulAggregateAnswer, FaithfulResult]

    def __init__(self) -> None:
        super().__init__()

        self._answer_counts = TableCounter(('label', 'predict', 'explain_predict'))
        self._faithful_count = 0
        self._correct_count = 0
        self._missmatch_count = 0

    def _add_answer(self, answer: FaithfulResult):
        if answer['faithful'] is None or answer['correct'] is None:
            self._missmatch_count += 1
        else:
            self._faithful_count += answer['faithful']
            self._correct_count += answer['correct']
            self._answer_counts.increment(answer)

    @property
    def progress_description(self) -> str:
        return f'Processing[C={self._correct_count}, F={self._faithful_count}, M={self._missmatch_count}, E={self._error_count}]'

    @property
    def results(self) -> FaithfulAggregateResult:
        return {
            'answer': self._answer_counts.as_table(),
            'faithful': self._faithful_count,
            'correct': self._correct_count,
            'missmatch': self._missmatch_count,
            'error': self._error_count,
            'total': self._total_count
        }
