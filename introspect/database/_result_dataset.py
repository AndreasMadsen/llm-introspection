
from traceback import format_exception
from pathlib import Path
from typing import Generic, TypeVar
from abc import abstractmethod
import pickle

from ..types import DatasetSplits, OfflineError, TaskResult, GenerateError
from ._abstract_dataset import AbstractDatabase

_split_to_id = { DatasetSplits.TRAIN: 0, DatasetSplits.VALID: 1, DatasetSplits.TEST: 2 }

def _idx_split_to_rowid(split: DatasetSplits, idx: int) -> int:
   return idx * 3 + _split_to_id[split]

TaskResultType = TypeVar('TaskResultType', bound=TaskResult)

class ResultDatabase(AbstractDatabase, Generic[TaskResultType]):
    _put_sql: str
    _has_sql: str
    _get_sql: str

    def __init__(self, database: str, persistent_dir: Path|None=None, **kwargs) -> None:
        if persistent_dir is None:
            filepath = database
        else:
            filepath = (persistent_dir / 'database' / database).with_suffix('.sqlite')
        super().__init__(filepath, **kwargs)

    async def put(self, split: DatasetSplits, idx: int, data: TaskResultType) -> None:
        """Add or update an entry to the database

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index
            data (TaskResultType): data to add, input is a dictionary.
                The index of the observation is identified by the idx property.
        """
        rowid = _idx_split_to_rowid(split, idx)

        traceback: str|None = None
        error: bytes|None = None
        match data['error']:
            case OfflineError():
                # There is information value in saving an OfflineError
                return

            case GenerateError():
                error = pickle.dumps(data['error'])
                traceback = ''.join(format_exception(data['error']))

            case Exception():
                raise ValueError(
                    "data['error'] is an error, but not a GenerationError"
                ) from data['error'] # type:ignore

        await self._con.execute(self._put_sql, {
            **data,
            'error': error,
            'traceback': traceback,
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
    def _get_unpack(self, split: DatasetSplits, idx: int, *args) -> TaskResultType:
        ...

    async def get(self, split: DatasetSplits, idx: int) -> TaskResultType|None:
        """Get entry by index

        Args:
            split (DatasetSplits): Dataset split
            idx (int): Observation index

        Returns:
            TaskResultType|None: Returns the entry if it exists.
                Otherwise, return None.
        """
        rowid = _idx_split_to_rowid(split, idx)
        cursor = await self._con.execute(self._get_sql, (rowid, ))
        results = await cursor.fetchone()
        if results is None:
            return None

        return self._get_unpack(*results)
