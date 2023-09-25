
from typing import TypedDict, Required

class Observation(TypedDict):
    label: Required[int]
    idx: Required[int]

class SentimentObservation(Observation):
    text: Required[str]
