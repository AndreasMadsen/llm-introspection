import pathlib
from functools import cached_property
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Iterable, Generic, Mapping
from collections.abc import Iterable

import datasets

from ..types import DatasetCategories, DatasetSplits, Observation

ObservationType = TypeVar('ObservationType', bound=Observation)
LabelNamesType = TypeVar('LabelNamesType', bound=str)

class AbstractDataset(Generic[ObservationType, LabelNamesType], metaclass=ABCMeta):
    name: str
    category: DatasetCategories

    _persistent_dir: pathlib.Path
    _features: datasets.Features

    _split_train: str
    _split_valid: str
    _split_test: str

    def __init__(self, persistent_dir: pathlib.Path, seed: int|None = None):
        """Class describing the semantics of a dataset.

        Args:
            persistent_dir (pathlib.Path): Persistent directory, used for storing the dataset.
            seed (int, optional): Seed used to shuffle the data. Default None.
        """
        self._builder_cache = self._builder(cache_dir=persistent_dir / 'cache' / 'datasets')
        self._persistent_dir = persistent_dir
        self._seed = seed

        if not isinstance(self.info.features, dict):
            raise ValueError('this dataset does not havce features defined')

    @abstractmethod
    def _builder(self, cache_dir: pathlib.Path) -> datasets.DatasetBuilder:
        ...

    @abstractmethod
    def _restructure(self, obs: dict[str, Any], idx: int) -> ObservationType:
        ...

    @cached_property
    def _datasets(self) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
        return self._builder_cache.as_dataset(split=(self._split_train, self._split_valid, self._split_test)) # type: ignore

    @property
    def info(self) -> datasets.DatasetInfo:
        """Standard information object
        """
        return self._builder_cache.info

    def download(self):
        """Downloads dataset
        """
        self._builder_cache.download_and_prepare()

    def _process_dataset(self, dataset: datasets.Dataset) -> Iterable[ObservationType]:
        return dataset \
            .map(self._restructure, with_indices=True, features=self._features, remove_columns=dataset.column_names) \
            .shuffle(seed=self._seed) # type: ignore

    def num_examples(self, split: DatasetSplits):
        """Number of observations for a given split"""
        match split:
            case 'train':
                return self.train_num_examples
            case 'valid':
                return self.valid_num_examples
            case 'test':
                return self.test_num_examples
            case _:
                raise ValueError(f'split {split} is not supported')

    def split(self, split: DatasetSplits) -> Iterable[ObservationType]:
        """Get observations for a given split"""
        match split:
            case 'train':
                return self.train()
            case 'valid':
                return self.valid()
            case 'test':
                return self.test()
            case _:
                raise ValueError(f'split {split} is not supported')

    def train(self) -> Iterable[ObservationType]:
        """Get training dataset
        """
        return self._process_dataset(self._datasets[0])

    @property
    def train_num_examples(self) -> int:
        """Number of training obsevations
        """
        return self._datasets[0].num_rows

    def valid(self) -> Iterable[ObservationType]:
        """Validation dataset
        """
        return self._process_dataset(self._datasets[1])
    @property
    def valid_num_examples(self) -> int:
        """Number of validation obsevations
        """
        return self._datasets[1].num_rows

    def test(self) -> Iterable[ObservationType]:
        """Test dataset
        """
        return self._process_dataset(self._datasets[2])

    @property
    def test_num_examples(self) -> int:
        """Number of test obsevations
        """
        return self._datasets[2].num_rows
