
from typing import TypeVar, Generic, TypedDict, Required, Literal

LabelType = TypeVar('LabelType', bound=str)

class Observation(TypedDict, Generic[LabelType]):
    label: Required[LabelType]
    idx: Required[int]

class SentimentObservation(Observation[Literal['negative', 'positive']]):
    text: Required[str]

class MultiChoiceObservation(Observation[str]):
    paragraph: Required[str]
    question: Required[str]
    choices: Required[list[str]]

class EntailmentObservation(Observation[Literal['yes', 'no']]):
    statement: Required[str]
    paragraph: Required[str]
