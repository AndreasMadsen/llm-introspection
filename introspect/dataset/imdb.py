
import datasets
from typing import Mapping, Literal
from functools import cached_property

from ._categories import SentimentDataset
from ..types import SentimentObservation

class IMDBDataset(SentimentDataset):
    name = 'IMDB'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    @cached_property
    def label_str2int(self) -> Mapping[Literal['negative', 'positive'], int]:
        return {
            'negative': self._label_def.names.index('neg'),
            'positive': self._label_def.names.index('pos')
        }

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('imdb', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> SentimentObservation:
        return {
            'text': obs['text'],
            'label': obs['label'],
            'idx': idx
        }
