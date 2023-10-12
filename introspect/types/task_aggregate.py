from typing import TypeVar, TypedDict, Required, Literal, Generic

class AggregateAnswer(TypedDict):
    count: Required[int]

AnswerAggregateType = TypeVar('AnswerAggregateType', bound=AggregateAnswer)

class AggregateResult(TypedDict, Generic[AnswerAggregateType]):
    answer: Required[list[AnswerAggregateType]]
    error: Required[int]
    total: Required[int]

class IntrospectAggregateAnswer(AggregateAnswer):
    label: Required[Literal['positive', 'negative']]
    sentiment: Required[Literal['positive', 'negative', 'neutral', 'unknown']]
    ability: Required[Literal['yes', 'no']]

class IntrospectAggregateResult(AggregateResult[IntrospectAggregateAnswer]):
    introspect: Required[int]
    correct: Required[int]
    missmatch: Required[int]

class FaithfulAggregateAnswer(AggregateAnswer):
    label: Required[Literal['positive', 'negative']]
    sentiment: Required[Literal['positive', 'negative', 'neutral', 'unknown']]
    explain_sentiment: Required[Literal['positive', 'negative', 'neutral', 'unknown']]

class FaithfulAggregateResult(AggregateResult[FaithfulAggregateAnswer]):
    faithful: Required[int]
    correct: Required[int]
    missmatch: Required[int]
