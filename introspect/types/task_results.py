
from typing import TypedDict, Required
from .generate import GenerateError

class AnswerableResult(TypedDict):
    answer_ability: Required[str|None]
    answer_sentiment: Required[str|None]
    introspect: Required[bool|None]
    correct: Required[bool|None]
    error: Required[GenerateError|None]
