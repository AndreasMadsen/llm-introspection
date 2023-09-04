
from enum import IntEnum
from typing import Type

from ._abstract_dataset import AbstractDataset
from ..types import DatasetCategories, SentimentObservation

class SentimentLabels(IntEnum):
    negative = 0
    positive = 1

class SentimentDataset(AbstractDataset[SentimentObservation]):
    category = DatasetCategories.SENTIMENT
    labels: Type[SentimentLabels] = SentimentLabels
