
import pytest
from traceback import format_exception

from introspect.database import Answerable
from introspect.types import AnswerableResult, DatasetSplits, GenerateError, OfflineError

@pytest.mark.asyncio
async def test_database_basic_put():
    obs: AnswerableResult = {
        'answer_ability': 'yes',
        'answer_sentiment': 'positive',
        'introspect': True,
        'correct': False,
        'duration': 10,
        'error': None
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
    obs_generate_error: AnswerableResult = {
        'answer_ability': None,
        'answer_sentiment': None,
        'introspect': None,
        'correct': None,
        'duration': None,
        'error': GenerateError('generate error')
    }
    obs_offline_error: AnswerableResult = {
        'answer_ability': None,
        'answer_sentiment': None,
        'introspect': None,
        'correct': None,
        'duration': None,
        'error': OfflineError('offline error')
    }

    async with Answerable(':memory:') as db:
        await db.put(DatasetSplits.TRAIN, 1, obs_generate_error)

        db_item = await db.get(DatasetSplits.TRAIN, 1)
        assert db_item is not None
        assert format_exception(db_item['error']) == format_exception(obs_generate_error['error'])

        # Check that offline errors do not overwrite existing errors
        await db.put(DatasetSplits.TRAIN, 1, obs_offline_error)
        db_item = await db.get(DatasetSplits.TRAIN, 1)
        assert db_item is not None
        assert format_exception(db_item['error']) == format_exception(obs_generate_error['error'])

        # If there is no existing error, then use the offline error
        await db.put(DatasetSplits.TRAIN, 2, obs_offline_error)
        db_item = await db.get(DatasetSplits.TRAIN, 2)
        assert db_item is not None
        assert format_exception(db_item['error']) == format_exception(obs_offline_error['error'])
