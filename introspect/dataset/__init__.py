
__all__ = [
    'AbstractDataset', 'SentimentDataset', 'MultiChoiceDataset',
    'SST2Dataset', 'IMDBDataset',
    'Babi1Dataset', 'Babi2Dataset', 'Babi3Dataset',
    'datasets'
]

from typing import Type, Mapping

from ._abstract_dataset import AbstractDataset
from ._categories import SentimentDataset, MultiChoiceDataset

from .sst2 import SST2Dataset
from .imdb import IMDBDataset
from .babi import Babi1Dataset, Babi2Dataset, Babi3Dataset

datasets: Mapping[str, Type[AbstractDataset]] = {
    Dataset.name: Dataset
    for Dataset
    in [SST2Dataset, IMDBDataset, Babi1Dataset, Babi2Dataset, Babi3Dataset]
}
