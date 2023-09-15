
import pytest

from introspect.database import GenerationCache
from introspect.types import GenerateResponse

@pytest.mark.asyncio
async def test_database_cache():
    obs: GenerateResponse = {
        'response': 'LLM response',
        'duration': 1
    }

    async with GenerationCache(':memory:') as db:
        assert not (await db.has('USER MESSAGE'))
        assert not (await db.has('ANOTHER USER MESSAGE'))

        # add response and check it exists
        await db.put('USER MESSAGE', obs)
        assert await db.has('USER MESSAGE')
        assert await db.get('USER MESSAGE') == obs

        # other responses are still missing
        assert not (await db.has('ANOTHER USER MESSAGE'))
