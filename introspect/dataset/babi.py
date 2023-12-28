
from ._categories import MultiChoiceDataset
from ..types import MultiChoiceObservation
from .local import LocalBabiDataset

class GenraizedBabiDataset(MultiChoiceDataset):
    _babi_config: str

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _builder(self, cache_dir):
       return LocalBabiDataset(config_name=self._babi_config, cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> MultiChoiceObservation:
        return {
            'paragraph': obs['paragraph'],
            'question': obs['question'],
            'choices': sorted(obs['choices']),
            'label': obs['label'],
            'idx': idx
        }

class Babi1Dataset(GenraizedBabiDataset):
    _babi_config = 'en-10k-qa1'
    name = 'bAbI-1'

class Babi2Dataset(GenraizedBabiDataset):
    _babi_config = 'en-10k-qa2'
    name = 'bAbI-2'

class Babi3Dataset(GenraizedBabiDataset):
    _babi_config = 'en-10k-qa3'
    name = 'bAbI-3'
