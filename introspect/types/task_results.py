
from typing import TypedDict, Required, Literal

class TaskResult(TypedDict):
    duration: Required[float]

class PartialAnswerableResult(TypedDict):
    ability_source: Required[str|None]
    ability: Required[Literal['yes', 'no']|None]
    sentiment_source: Required[str|None]
    sentiment: Required[Literal['positive', 'negative', 'neutral', 'unknown']|None]
    correct: Required[bool|None]
    introspect: Required[bool|None]

class AnswerableResult(TaskResult, PartialAnswerableResult):
    pass
