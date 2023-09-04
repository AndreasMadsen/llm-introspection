
__all__ = [
    'SentimentDataset',
    'SST2Dataset', 'IMDBDataset',
    'datasets'
]

from typing import Type

from ._abstract_dataset import AbstractDataset
from ._categories import SentimentDataset

from .sst2 import SST2Dataset
from .imdb import IMDBDataset

datasets: dict[str, Type[AbstractDataset]] = {
    Dataset.name: Dataset
    for Dataset
    in [SST2Dataset, IMDBDataset]
}
