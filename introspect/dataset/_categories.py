
from typing import TypedDict

from ._abstract_dataset import AbstractDataset
from ..types import DatasetCategories, SentimentObservation

class SentimentLabels(TypedDict):
    positive: int
    negative: int

class SentimentDataset(AbstractDataset[SentimentObservation, SentimentLabels]):
    category = DatasetCategories.SENTIMENT
    labels: SentimentLabels
