
from typing import TypedDict, Required

class TaskResult(TypedDict):
    duration: Required[float]

class PartialAnswerableResult(TypedDict):
    answer_ability: Required[str|None]
    answer_sentiment: Required[str|None]
    introspect: Required[bool|None]
    correct: Required[bool|None]

class AnswerableResult(TaskResult, PartialAnswerableResult):
    pass
