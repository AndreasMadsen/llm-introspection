
from typing import Literal

from ._abstract_dataset import AbstractDataset
from ..types import DatasetCategories, SentimentObservation

class SentimentDataset(AbstractDataset[SentimentObservation, Literal['positive', 'negative']]):
    category = DatasetCategories.SENTIMENT
