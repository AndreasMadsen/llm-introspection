
import datasets
from typing import Mapping, Literal
from functools import cached_property

from ._categories import MultiChoiceDataset
from ..types import MultiChoiceObservation

class MCTestDataset(MultiChoiceDataset):
    name = 'MCTest'

    _split_train = 'train'
    _split_valid = 'validation'
    _split_test = 'test'

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('sagnikrayc/mctest', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> MultiChoiceObservation:
        return {
            'paragraph': obs['story'].replace(r'\newline', '\n'),
            'question': obs['question'],
            'choices': list(obs['answer_options'].values()),
            'label': obs['answer_options'][obs['answer']],
            'idx': idx
        }
