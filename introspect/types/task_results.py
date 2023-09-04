
from typing import TypedDict, Required

class AnswerableResult(TypedDict):
    answer_ability: Required[str]
    answer_sentiment: Required[str]
    introspect: Required[bool|None]
    correct: Required[bool|None]
