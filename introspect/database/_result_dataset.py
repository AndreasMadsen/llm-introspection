
import traceback
from pathlib import Path
from typing import Generic, TypeVar, TypedDict
from abc import abstractmethod

from ..types import DatasetSplits
from ._abstract_dataset import AbstractDatabase
from ..client import OfflineError

_split_to_id = { DatasetSplits.TRAIN: 0, DatasetSplits.VALID: 1, DatasetSplits.TEST: 2 }

def _idx_split_to_rowid(split: DatasetSplits, idx: int) -> int:
   return idx * 3 + _split_to_id[split]

ObservationType = TypeVar('ObservationType', bound=TypedDict)

class ResultDatabase(AbstractDatabase, Generic[ObservationType]):
    _put_sql: str
    _has_sql: str
    _get_sql: str

    def __init__(self, database: str, persistent_dir: Path|None=None, **kwargs) -> None:
        if persistent_dir is None:
            filepath = database
        else:
            filepath = (persistent_dir / 'database' / database).with_suffix('.sqlite')
        super().__init__(filepath, **kwargs)

    async def put(self, split: DatasetSplits, idx: int, data: ObservationType) -> None:
        """Add or update an entry to the database

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index
            data (ObservationType): data to add, input is a dictionary.
                The index of the observation is identified by the idx property.
        """
        rowid = _idx_split_to_rowid(split, idx)

        error = None
        # data['error'] is an Exception object and therefore needs to be converted to a string
        if 'error' in data:
            match data['error']:
                # In case of an OfflineError, preserve the original error message
                case OfflineError():
                    existing_data = await self.get(split, idx)
                    if existing_data is None:
                        error = ''.join(traceback.format_exception(data['error']))
                    else:
                        error = existing_data['error']

                # Stringify the error
                case Exception():
                    error = ''.join(traceback.format_exception(data['error']))

        await self._con.execute(self._put_sql, {
            **data,
            'error': error,
            'split': _split_to_id[split],
            'idx': idx,
            'rowid': rowid
        })
        self._transactions_queued += 1
        self._maybe_commit()

    async def has(self, split: DatasetSplits, idx: int) -> bool:
        """Check if observation exists

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index

        Returns:
            bool: True if the observation exists.
        """
        rowid = _idx_split_to_rowid(split, idx)
        cursor = await self._con.execute(self._has_sql, (rowid, ))
        exists, = await cursor.fetchone() # type: ignore
        return exists == 1

    @abstractmethod
    def _get_unpack(self, split: DatasetSplits, idx: int, *args) -> ObservationType:
        ...

    async def get(self, split: DatasetSplits, idx: int) -> ObservationType|None:
        """Get entry by index

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index

        Returns:
            ObservationType|None: Returns the entry if it exists.
                Otherwise, return None.
        """
        rowid = _idx_split_to_rowid(split, idx)
        cursor = await self._con.execute(self._get_sql, (rowid, ))
        results = await cursor.fetchone()
        if results is None:
            return None

        return self._get_unpack(*results)
