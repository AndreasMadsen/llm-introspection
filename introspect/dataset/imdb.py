
import datasets

from ._categories import SentimentDataset, SentimentLabels
from ..types import SentimentObservation

class IMDBDataset(SentimentDataset):
    name = 'IMDB'

    _split_train = 'train[:80%]'
    _split_valid = 'train[80%:]'
    _split_test = 'test'

    def _labels(self, label_def) -> SentimentLabels:
        return {
            'negative': label_def.names.index('neg'),
            'positive': label_def.names.index('pos')
        }

    def _builder(self, cache_dir):
       return datasets.load_dataset_builder('imdb', cache_dir=str(cache_dir))

    def _restructure(self, obs, idx) -> SentimentObservation:
        return {
            'text': obs['text'],
            'label': obs['label'],
            'idx': idx
        }
