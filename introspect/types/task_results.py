
from typing import TypedDict, Required, Literal, TypeAlias

_sentiments: TypeAlias = Literal['positive', 'negative', 'neutral', 'unknown']

class TaskResult(TypedDict):
    duration: Required[float]
    label: Required[Literal['positive', 'negative']]

class PartialIntrospectResult(TypedDict):
    sentiment_source: Required[str|None]
    sentiment: Required[_sentiments|None]
    correct: Required[bool|None]
    ability_source: Required[str|None]
    ability: Required[Literal['yes', 'no']|None]
    introspect: Required[bool|None]

class IntrospectResult(TaskResult, PartialIntrospectResult):
    pass

class PartialFaithfulResult(TypedDict):
    sentiment_source: Required[str|None]
    sentiment: Required[_sentiments|None]
    correct: Required[bool|None]
    explain_source: Required[str|None]
    explain: Required[str|None]
    explain_sentiment_source: Required[str|None]
    explain_sentiment: Required[_sentiments|None]
    faithful: Required[bool|None]

class FaithfulResult(TaskResult, PartialFaithfulResult):
    pass
