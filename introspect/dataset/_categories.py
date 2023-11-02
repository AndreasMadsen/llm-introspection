
from typing import Literal

import datasets

from ._abstract_dataset import AbstractDataset
from ..types import DatasetCategories, SentimentObservation, MultiChoiceObservation
from ..types import SentimentObservation

class SentimentDataset(AbstractDataset[SentimentObservation, Literal['positive', 'negative']]):
    category = DatasetCategories.SENTIMENT

    _features = datasets.Features({
        "text": datasets.Value("string"),
        "label": datasets.Value("string"),
        "idx": datasets.Value("int64"),
    })

class MultiChoiceDataset(AbstractDataset[MultiChoiceObservation, Literal['garden', 'hallway', 'kitchen', 'office', 'bedroom', 'bathroom']]):
    category = DatasetCategories.MULTI_CHOICE

    _features = datasets.Features({
        "paragraph": datasets.Value("string"),
        "question": datasets.Value("string"),
        "choices": datasets.Sequence(datasets.Value("string")),
        "label": datasets.Value("string"),
        "idx": datasets.Value("int64"),
    })
