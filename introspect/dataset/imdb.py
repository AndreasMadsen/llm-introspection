
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
    def _label_int2str(self) -> Mapping[int, Literal['negative', 'positive']]:
        label_def = self.info.features['label'] # type: ignore

        return {
            label_def.names.index('neg'): 'negative',
            label_def.names.index('pos'): 'positive'
        }

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('stanfordnlp/imdb', 'plain_text', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> SentimentObservation:
        return {
            'text': obs['text'].replace('<br />', '\n'),
            'label': self._label_int2str[obs['label']],
            'idx': idx
        }
