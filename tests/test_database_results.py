
import pytest
from traceback import format_exception

from introspect.database import Answerable
from introspect.types import IntrospectResult, DatasetSplits, GenerateError, OfflineError

@pytest.mark.asyncio
async def test_database_basic_put():
    obs: IntrospectResult = {
        'sentiment_source': 'positive',
        'sentiment': 'positive',
        'correct': False,
        'ability_source': 'yes',
        'ability': 'yes',
        'introspect': True,
        'duration': 10,
        'label': 'positive'
    }

    async with Answerable(':memory:') as db:
        assert not (await db.has(DatasetSplits.TRAIN, 1))
        assert not (await db.has(DatasetSplits.TRAIN, 2))
        assert not (await db.has(DatasetSplits.VALID, 1))
        assert not (await db.has(DatasetSplits.VALID, 2))
        assert not (await db.has(DatasetSplits.TEST, 1))
        assert not (await db.has(DatasetSplits.TEST, 2))

        await db.put(DatasetSplits.TRAIN, 1, obs)
        assert (await db.has(DatasetSplits.TRAIN, 1))
        assert not (await db.has(DatasetSplits.TRAIN, 2))
        assert not (await db.has(DatasetSplits.VALID, 1))
        assert not (await db.has(DatasetSplits.VALID, 2))
        assert not (await db.has(DatasetSplits.TEST, 1))
        assert not (await db.has(DatasetSplits.TEST, 2))

        assert await db.get(DatasetSplits.TRAIN, 1) == obs
        assert await db.get(DatasetSplits.TRAIN, 2) == None
        assert await db.get(DatasetSplits.VALID, 1) == None
        assert await db.get(DatasetSplits.VALID, 2) == None
        assert await db.get(DatasetSplits.TEST, 1) == None
        assert await db.get(DatasetSplits.TEST, 2) == None

@pytest.mark.asyncio
async def test_database_error_storage():
    obs_generate_error = GenerateError('generate error')
    obs_offline_error = OfflineError('offline error')

    async with Answerable(':memory:') as db:
        await db.put(DatasetSplits.TRAIN, 1, obs_generate_error)

        db_item = await db.get(DatasetSplits.TRAIN, 1)
        assert isinstance(db_item,GenerateError)
        assert format_exception(db_item) == format_exception(obs_generate_error)

        # If there is existing error, then don't save an OfflineError
        await db.put(DatasetSplits.TRAIN, 1, obs_offline_error)
        db_item = await db.get(DatasetSplits.TRAIN, 1)
        assert isinstance(db_item,GenerateError)
        assert format_exception(db_item) == format_exception(obs_generate_error)

        # If there is no existing error, then don't save an OfflineError
        await db.put(DatasetSplits.TRAIN, 2, obs_offline_error)
        db_item = await db.get(DatasetSplits.TRAIN, 2)
        assert db_item is None
