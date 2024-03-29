
import pathlib
from typing import Self
from abc import ABCMeta
import os
import asyncio

import aiosqlite as sql

class AbstractDatabase(metaclass=ABCMeta):
    _setup_sql: str
    _con: sql.Connection

    def __init__(self, filepath: pathlib.Path|str, min_commit_transactions: int=100) -> None:
        """Create Database

        Args:
            filepath (pathlib.Path | str): A filepath or in-memory address, where the database is stored
            min_commit_transactions (int, optional): The minimum number of transactions before commiting
                to the database on disk. Default to 100.
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

    async def open(self) -> bool:
        """Open connection and create databases

        Likely this should not be used directly. Instead, use `async with`.

        Returns:
            bool: return true if a new database was created
        """
        is_new = (not self._filepath.exists()) if isinstance(self._filepath, pathlib.Path) else True
        self._con = await sql.connect(self._filepath)
        # Use WAL (write ahead log) with NORMAL synchronization, for best performance.
        if is_new:
            await self._con.execute('PRAGMA journal_mode = WAL;')
            await self._con.execute('PRAGMA synchronous = NORMAL;')

        await self._con.execute(self._setup_sql)
        await self._ensure_commit()

        return is_new

    async def commit(self) -> None:
        """Commits  connection
        """
        await self._ensure_commit()

    async def close(self) -> None:
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

    async def backup(self, target: Self, *args) -> None:
        """Make a backup of the database
        """
        await self._ensure_commit()
        await self._con.backup(target._con, *args)

    def remove(self) -> None:
        """Remove database
        """
        try:
            os.remove(self._filepath)
        except FileNotFoundError:
            pass
