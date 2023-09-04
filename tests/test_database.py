
import asyncio

import pytest

from introspect.database import Answerable
from introspect.types import AnswerableResult, DatasetSplits

@pytest.mark.asyncio
async def test_database_answerable():
    obs: AnswerableResult = {
        'answer_ability': 'yes',
        'answer_sentiment': 'positive',
        'introspect': True,
        'correct': False
    }

    async with Answerable(':memory:') as db:
        assert not (await db.has(DatasetSplits.TRAIN, 1))
        assert not (await db.has(DatasetSplits.TRAIN, 2))
        assert not (await db.has(DatasetSplits.VALID, 1))
        assert not (await db.has(DatasetSplits.VALID, 2))
        assert not (await db.has(DatasetSplits.TEST, 1))
        assert not (await db.has(DatasetSplits.TEST, 2))

        await db.add(DatasetSplits.TRAIN, 1, obs)
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
