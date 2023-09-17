
import pytest
import asyncio

from introspect.client import OfflineClient
from introspect.database import GenerationCache
from introspect.types import GenerateResponse, SystemMessage, OfflineError
from introspect.model import FalconModel
from introspect.tasks._request_capture import RequestCapture

@pytest.mark.asyncio
async def test_request_capture_duration_accumulate():
    obs_1: GenerateResponse = {
        'response': 'LLM response 1',
        'duration': 1
    }
    obs_2: GenerateResponse = {
        'response': 'LLM response 2',
        'duration': 2
    }

    async with GenerationCache(':memory:') as cache:
        await cache.put('User: USER MESSAGE 1.\nFalcon:', obs_1)
        await cache.put('User: USER MESSAGE 2.\nFalcon:', obs_2)

        client = OfflineClient(cache=cache)
        model = FalconModel(client, system_message=SystemMessage.NONE)
        capture = RequestCapture(model)

        answer_1, answer_2 = await asyncio.gather(
            capture([{ 'user': 'USER MESSAGE 1.', 'assistant': None}]),
            capture([{ 'user': 'USER MESSAGE 2.', 'assistant': None}])
        )

        assert answer_1 == 'LLM response 1'
        assert answer_2 == 'LLM response 2'
        assert capture.duration == 3
