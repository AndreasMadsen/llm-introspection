
import pathlib
from typing import Self, Generic, TypeVar, TypedDict
from abc import ABCMeta, abstractmethod
import os
import asyncio

import aiosqlite as sql

from ..types import DatasetSplits

ObservationType = TypeVar('ObservationType', bound=TypedDict)

_split_to_id = { DatasetSplits.TRAIN: 0, DatasetSplits.VALID: 1, DatasetSplits.TEST: 2 }
_id_to_split = [DatasetSplits.TRAIN, DatasetSplits.VALID, DatasetSplits.TEST]

def _idx_split_to_rowid(split: DatasetSplits, idx: int) -> int:
   return idx * 3 + _split_to_id[split]

class AbstractDatabase(Generic[ObservationType], metaclass=ABCMeta):
    _setup_sql: str
    _add_sql: str
    _has_sql: str
    _get_sql: str

    def __init__(self, filepath: pathlib.Path|str, min_commit_transactions=100) -> None:
        """Create Database to store results

        Example:
        async with Database(':memory:') as db:
            print(await db.has(0))

        Args:
            filepath (pathlib.Path | str): A filepath or in-memory address, where the database is tored
        """
        self._filepath = filepath
        self._min_commit_transactions = min_commit_transactions
        self._transactions_queued = 0
        self._commit_task = None

    def _schedule_commit(self):
        if self._commit_task is None:
            def task_done(_):
                self._commit_task = None
                self._maybe_commit()

            async def commit():
                self._transactions_queued = 0
                await self._con.commit()

            self._commit_task = asyncio.create_task(commit())
            self._commit_task.add_done_callback(task_done)

        return self._commit_task

    def _maybe_commit(self):
        if self._transactions_queued >= self._min_commit_transactions:
            self._schedule_commit()

    async def _ensure_commit(self):
        if self._commit_task is not None:
            await self._commit_task
        await self._schedule_commit()

    async def open(self):
        """Open connection and create databases

        Likely this should not be used directly. Instead, use `async with`.
        """
        self._con = await sql.connect(self._filepath)
        await self._con.execute(self._setup_sql)
        await self._ensure_commit()

    async def commit(self):
        """Commits  connection
        """
        await self._ensure_commit()

    async def close(self):
        """Close connection

        Likely this should not be used directly. Instead, use `async with`.
        """
        await self._con.close()
        del self._con

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._ensure_commit()
        await self.close()

    async def add(self, split: DatasetSplits, idx: int, data: ObservationType) -> None:
        """Add an entry to the database

        Args:
            data (ObservationType): data to add, input is a dictionary.
                The index of the observation is identified by the idx property.
        """
        rowid = _idx_split_to_rowid(split, idx)
        await self._con.execute(self._add_sql, {
            **data,
            'split': _split_to_id[split],
            'idx': idx,
            'rowid': rowid
        })
        self._transactions_queued += 1
        self._maybe_commit()

    async def has(self, split: DatasetSplits, idx: int) -> bool:
        """Check if observation exists

        Args:
            idx (int): Observation index

        Returns:
            bool: _description_
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

    async def backup(self, *args) -> None:
        """Make a backup of the database
        """
        await self._ensure_commit()
        await self._con.backup(*args)

    def remove(self) -> None:
        """Remove database
        """
        try:
            os.remove(self._filepath)
        except FileNotFoundError:
            pass
