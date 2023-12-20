
__all__ = [
    'AbstractDataset', 'SentimentDataset', 'MultiChoiceDataset', 'EntailmentDataset',
    'SST2Dataset', 'IMDBDataset',
    'Babi1Dataset', 'Babi2Dataset', 'Babi3Dataset', 'MCTestDataset',
    'RTEDataset',
    'datasets'
]

from typing import Type, Mapping

from ._abstract_dataset import AbstractDataset
from ._categories import SentimentDataset, MultiChoiceDataset, EntailmentDataset

from .sst2 import SST2Dataset
from .imdb import IMDBDataset
from .babi import Babi1Dataset, Babi2Dataset, Babi3Dataset
from .mctest import MCTestDataset
from .rte import RTEDataset

datasets: Mapping[str, Type[AbstractDataset]] = {
    Dataset.name: Dataset
    for Dataset
    in [SST2Dataset, IMDBDataset,
        Babi1Dataset, Babi2Dataset, Babi3Dataset, MCTestDataset,
        RTEDataset]
}
