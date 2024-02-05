
from typing import Literal

import datasets

from ._abstract_dataset import AbstractDataset
from ..types import DatasetCategories, SentimentObservation, MultiChoiceObservation, EntailmentObservation
from ..types import SentimentObservation

class SentimentDataset(AbstractDataset[SentimentObservation]):
    category = DatasetCategories.SENTIMENT

    _features = datasets.Features({
        "text": datasets.Value("string"),
        "label": datasets.Value("string"),
        "idx": datasets.Value("int64"),
    })

class MultiChoiceDataset(AbstractDataset[MultiChoiceObservation]):
    category = DatasetCategories.MULTI_CHOICE

    _features = datasets.Features({
        "paragraph": datasets.Value("string"),
        "question": datasets.Value("string"),
        "choices": datasets.Sequence(datasets.Value("string")),
        "label": datasets.Value("string"),
        "idx": datasets.Value("int64"),
    })

class EntailmentDataset(AbstractDataset[EntailmentObservation]):
    category = DatasetCategories.ENTAILMENT

    _features = datasets.Features({
        "hypothesis": datasets.Value("string"),
        "paragraph": datasets.Value("string"),
        "label": datasets.Value("string"),
        "idx": datasets.Value("int64"),
    })
