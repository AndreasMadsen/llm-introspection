
import datasets

from ._categories import SentimentDataset
from ..types import SentimentObservation

class IMDBDataset(SentimentDataset):
    name = 'IMDB'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('imdb', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> SentimentObservation:
        return {
            'text': obs['text'],
            'label': obs['label'],
            'idx': idx
        }
