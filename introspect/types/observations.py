
from typing import TypedDict, Required

class SentimentObservation(TypedDict):
    text: Required[str]
    label: Required[int]
    idx: Required[int]
