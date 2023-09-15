
import pathlib
from dataclasses import dataclass

import pytest

from introspect.dataset import datasets, SentimentDataset
from introspect.types import DatasetCategories, DatasetSplits

@dataclass
class DatasetExpectations:
    name: str
    train: int
    valid: int
    test: int
    num_classes: int

sentiment_datasets = [
    DatasetExpectations('IMDB', 20000, 5000, 25000, 2),
    DatasetExpectations('SST2', 53879, 13470, 872, 2),
]

@pytest.mark.parametrize("info", sentiment_datasets, ids=lambda info: info.name)
def test_dataset_sentiment(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert isinstance(dataset, SentimentDataset)

    assert dataset.name == info.name
    assert dataset.category == DatasetCategories.SENTIMENT

    assert isinstance(dataset.labels['negative'], int)
    assert isinstance(dataset.labels['positive'], int)
    assert dataset.labels['positive'] != dataset.labels['negative']

    assert dataset.train_num_examples == dataset.num_examples(DatasetSplits.TRAIN) == info.train
    assert dataset.valid_num_examples == dataset.num_examples(DatasetSplits.VALID) == info.valid
    assert dataset.test_num_examples == dataset.num_examples(DatasetSplits.TEST) == info.test

    for example in dataset.train():
        assert example.keys() == { 'text', 'label', 'idx' }
        break
