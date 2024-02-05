
import datasets
from typing import Mapping, Literal
from functools import cached_property

from ._categories import EntailmentDataset
from ..types import EntailmentObservation

class RTEDataset(EntailmentDataset):
    name = 'RTE'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'validation'

    @cached_property
    def _label_int2str(self) -> Mapping[int, Literal['yes', 'no']]:
        label_def = self.info.features['label'] # type: ignore

        return {
            label_def.names.index('entailment'): 'yes',
            label_def.names.index('not_entailment'): 'no'
        }

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('glue', 'rte', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> EntailmentObservation:
        return {
            'hypothesis': obs['sentence2'],
            'paragraph': obs['sentence1'],
            'label': self._label_int2str[obs['label']],
            'idx': idx
        }
