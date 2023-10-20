
from pathlib import Path
import pickle
from traceback import format_exception
from typing import AsyncIterator, overload

from ._abstract_dataset import AbstractDatabase
from ..types import GenerateResponse, GenerateError

@overload
def _database_to_filepath(database: str, directory: Path) -> Path:
    ...

@overload
def _database_to_filepath(database: str, directory: None) -> str:
    ...

def _database_to_filepath(database, directory):
    if directory is None:
        filepath = database
    else:
        filepath = (directory / database).with_suffix('.sqlite')
    return filepath

class GenerationCache(AbstractDatabase):
    _setup_sql = '''
        CREATE TABLE IF NOT EXISTS Cache (
            prompt TEXT NOT NULL PRIMARY KEY,
            response TEXT,
            duration REAL,
            error BLOB,
            traceback TEXT
        ) STRICT, WITHOUT ROWID
    '''
    _put_sql = '''
        REPLACE INTO Cache(prompt, response, duration, error, traceback)
        VALUES (:prompt, :response, :duration, :error, :traceback)
    '''
    _has_sql = '''
        SELECT EXISTS(SELECT 1 FROM Cache WHERE prompt = ?)
    '''
    _get_sql = '''
        SELECT response, duration, error
        FROM Cache
        WHERE prompt = ?
    '''
    _iter_sql = '''
        SELECT prompt, response, duration, error
        FROM Cache
    '''

    def __init__(self, database: str, cache_dir: Path|None=None, deps: list[str]=[], **kwargs) -> None:
        self._database = database
        self._deps = deps
        self._cache_dir = cache_dir

        super().__init__(_database_to_filepath(database, cache_dir), **kwargs)

    async def open(self) -> bool:
        is_new = await super().open()

        if self._cache_dir is None:
            return is_new

        # if it is a new database, bootstrap the database using the dependencies
        for dep in self._deps:
            # prevent cloneing itself
            if dep == self._database:
                continue

            if not _database_to_filepath(dep, self._cache_dir).exists():
                continue

            # copy over content
            async with GenerationCache(dep, self._cache_dir) as source_db:
                async for prompt, answer in source_db:
                    await self.put(prompt, answer)

        return is_new

    async def __aiter__(self) -> AsyncIterator[tuple[str, GenerateResponse|GenerateError]]:
        cursor = await self._con.execute(self._iter_sql)
        async for row in cursor:
            prompt, response, duration, error = row
            yield (prompt, self._unpack_results(response, duration, error))

    async def put(self, prompt: str, answer: GenerateResponse|GenerateError) -> None:
        """Add or update an entry to the database

        Args:
            prompt (str): Prompt sent to generative model
            answer (GenerateResponse, GenerateError): Answer by generative model
        """
        match answer:
            case GenerateError():
                await self._con.execute(self._put_sql, {
                    'prompt': prompt,
                    'response': None,
                    'duration': None,
                    'error': pickle.dumps(answer),
                    'traceback': ''.join(format_exception(answer)),
                })

            case _:
                await self._con.execute(self._put_sql, {
                    'prompt': prompt,
                    'response': answer['response'],
                    'duration': answer['duration'],
                    'error': None,
                    'traceback': None
                })

        self._transactions_queued += 1
        self._maybe_commit()

    async def has(self, prompt: str) -> bool:
        """Check if observation exists

        Args:
            prompt (str): Prompt sent to generative model

        Returns:
            bool: True if the observation exists.
        """
        cursor = await self._con.execute(self._has_sql, (prompt, ))
        exists, = await cursor.fetchone() # type: ignore
        return exists == 1

    async def get(self, prompt: str) -> GenerateResponse|GenerateError|None:
        """Get entry by index

        Args:
            prompt (str): Prompt sent to generative model

        Returns:
            tuple[str, float]|None: If the entry exists, return
                a (answer, duration) tuple.
                Otherwise, return None.
        """
        cursor = await self._con.execute(self._get_sql, (prompt, ))
        results = await cursor.fetchone()
        if results is None:
            return None

        return self._unpack_results(*results)

    def _unpack_results(self, response: str|None, duration: float|None, error: bytes|None) -> GenerateResponse|GenerateError:
        if error is not None:
            return pickle.loads(error)
        if response is None or duration is None:
            raise IOError(f'unexpected database content: {response=}, {duration=}')

        return {
            'response': response,
            'duration': duration
        }
