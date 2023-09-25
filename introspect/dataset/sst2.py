
import datasets
from typing import Mapping, Literal
from functools import cached_property

from ._categories import SentimentDataset
from ..types import SentimentObservation

class SST2Dataset(SentimentDataset):
    name = 'SST2'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    @cached_property
    def label_str2int(self) -> Mapping[Literal['negative', 'positive'], int]:
        return {
            'negative': self._label_def.names.index('negative'),
            'positive': self._label_def.names.index('positive')
        }

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('glue', 'sst2', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> SentimentObservation:
        return {
            'text': obs['sentence'],
            'label': obs['label'],
            'idx': idx
        }
