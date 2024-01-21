from typing import TypeVar, TypedDict, Required, Literal, Generic

class AggregateAnswer(TypedDict):
    count: Required[int]

AnswerAggregateType = TypeVar('AnswerAggregateType', bound=AggregateAnswer)

class AggregateResult(TypedDict, Generic[AnswerAggregateType]):
    answer: Required[list[AnswerAggregateType]]
    error: Required[int]
    total: Required[int]

class ClassifyAggregateAnswer(AggregateAnswer):
    label: Required[str]
    predict: Required[str]

class ClassifyAggregateResult(AggregateResult[ClassifyAggregateAnswer]):
    correct: Required[int]
    missmatch: Required[int]

class IntrospectAggregateAnswer(ClassifyAggregateAnswer):
    ability: Required[Literal['yes', 'no']]

class IntrospectAggregateResult(AggregateResult[IntrospectAggregateAnswer]):
    introspect: Required[int]
    correct: Required[int]
    introspect_and_correct: Required[int]
    missmatch: Required[int]

class FaithfulAggregateAnswer(ClassifyAggregateAnswer):
    explain_predict: Required[str]

class FaithfulAggregateResult(AggregateResult[FaithfulAggregateAnswer]):
    faithful: Required[int]
    correct: Required[int]
    faithful_and_correct: Required[int]
    missmatch: Required[int]
