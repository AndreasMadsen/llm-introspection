
from pathlib import Path
from ._abstract_dataset import AbstractDatabase
from ..types import GenerateResponse

class GenerationCache(AbstractDatabase):
    _setup_sql = '''
        CREATE TABLE IF NOT EXISTS Cache (
            prompt TEXT NOT NULL PRIMARY KEY,
            response TEXT NOT NULL,
            duration REAL NOT NULL
        ) WITHOUT ROWID
    '''
    _put_sql = '''
        REPLACE INTO Cache(prompt, response, duration)
        VALUES (:prompt, :response, :duration)
    '''
    _has_sql = '''
        SELECT EXISTS(SELECT 1 FROM Cache WHERE prompt = ?)
    '''
    _get_sql = '''
        SELECT response, duration
        FROM Cache
        WHERE prompt = ?
    '''

    def __init__(self, database: str, persistent_dir: Path|None=None, **kwargs) -> None:
        if persistent_dir is None:
            filepath = database
        else:
            filepath = (persistent_dir / 'database' / database).with_suffix('.sqlite')
        super().__init__(filepath, **kwargs)

    async def put(self, prompt: str, answer: GenerateResponse) -> None:
        """Add or update an entry to the database

        Args:
            prompt (str): Prompt sent to generative model
            answer (GenerateResponse): Answer by generative model
        """
        await self._con.execute(self._put_sql, {**answer, 'prompt': prompt})
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

    async def get(self, prompt: str) -> GenerateResponse|None:
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

        return {
            'response': results[0],
            'duration': results[1]
        }
