
import pathlib
from dataclasses import dataclass

import pytest

from introspect.dataset import datasets, SentimentDataset, MultiChoiceDataset, EntailmentDataset
from introspect.types import DatasetCategories, DatasetSplits

@dataclass
class DatasetExpectations:
    name: str
    train: int
    valid: int
    test: int

sentiment_datasets = [
    DatasetExpectations('IMDB', 20000, 5000, 25000),
    DatasetExpectations('SST2', 53879, 13470, 872),
]

multi_choice_datasets = [
    DatasetExpectations('bAbI-1', 8000, 2000, 1000),
    DatasetExpectations('bAbI-2', 8000, 2000, 1000),
    DatasetExpectations('bAbI-3', 8000, 2000, 1000),
    DatasetExpectations('MCTest', 1200, 200, 600),
]

entailment_datasets = [
    DatasetExpectations('RTE', 1992, 498, 277),
]

@pytest.mark.parametrize("info", sentiment_datasets, ids=lambda info: info.name)
def test_dataset_sentiment(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert isinstance(dataset, SentimentDataset)

    assert dataset.name == info.name
    assert dataset.category == DatasetCategories.SENTIMENT

    assert dataset.train_num_examples == dataset.num_examples(DatasetSplits.TRAIN) == info.train
    assert dataset.valid_num_examples == dataset.num_examples(DatasetSplits.VALID) == info.valid
    assert dataset.test_num_examples == dataset.num_examples(DatasetSplits.TEST) == info.test

    for example in dataset.train():
        assert example.keys() == { 'text', 'label', 'idx' }
        break

@pytest.mark.parametrize("info", multi_choice_datasets, ids=lambda info: info.name)
def test_dataset_multi_choice(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert isinstance(dataset, MultiChoiceDataset)

    assert dataset.name == info.name
    assert dataset.category == DatasetCategories.MULTI_CHOICE

    assert dataset.train_num_examples == dataset.num_examples(DatasetSplits.TRAIN) == info.train
    assert dataset.valid_num_examples == dataset.num_examples(DatasetSplits.VALID) == info.valid
    assert dataset.test_num_examples == dataset.num_examples(DatasetSplits.TEST) == info.test

    for example in dataset.train():
        assert example.keys() == { 'paragraph', 'question', 'choices', 'label', 'idx' }
        break

@pytest.mark.parametrize("info", entailment_datasets, ids=lambda info: info.name)
def test_dataset_entailment(info):
    dataset = datasets[info.name](persistent_dir=pathlib.Path('.'))

    assert isinstance(dataset, EntailmentDataset)

    assert dataset.name == info.name
    assert dataset.category == DatasetCategories.ENTAILMENT

    assert dataset.train_num_examples == dataset.num_examples(DatasetSplits.TRAIN) == info.train
    assert dataset.valid_num_examples == dataset.num_examples(DatasetSplits.VALID) == info.valid
    assert dataset.test_num_examples == dataset.num_examples(DatasetSplits.TEST) == info.test

    for example in dataset.train():
        assert example.keys() == { 'statement', 'paragraph', 'label', 'idx' }
        break
