
import pathlib
from typing import Self
from abc import ABCMeta
import os
import asyncio

import aiosqlite as sql

class AbstractDatabase(metaclass=ABCMeta):
    _setup_sql: str
    _con: sql.Connection

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

    async def open(self) -> None:
        """Open connection and create databases

        Likely this should not be used directly. Instead, use `async with`.
        """
        self._con = await sql.connect(self._filepath)
        await self._con.execute(self._setup_sql)
        await self._ensure_commit()

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
