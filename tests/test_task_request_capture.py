
import pytest
import asyncio

from introspect.client import OfflineClient
from introspect.database import GenerationCache
from introspect.types import GenerateResponse, SystemMessage, OfflineError
from introspect.model import FalconModel
from introspect.tasks._request_capture import request_capture_scope

@pytest.mark.asyncio
async def test_request_capture_scope_duration_accumulate():
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

        with request_capture_scope(model) as scope:
            answer_1, answer_2 = await asyncio.gather(
                scope([{ 'user': 'USER MESSAGE 1.', 'assistant': None}]),
                scope([{ 'user': 'USER MESSAGE 2.', 'assistant': None}])
            )

            assert answer_1 == 'LLM response 1'
            assert answer_2 == 'LLM response 2'
        assert scope.duration == 3
        assert scope.error == None

@pytest.mark.asyncio
async def test_request_capture_scope_error():
    obs_1: GenerateResponse = {
        'response': 'LLM response 1',
        'duration': 1
    }

    async with GenerationCache(':memory:') as cache:
        await cache.put('User: USER MESSAGE 1.\nFalcon:', obs_1)

        client = OfflineClient(cache=cache)
        model = FalconModel(client, system_message=SystemMessage.NONE)

        with request_capture_scope(model) as scope:
            answer_1, answer_2 = await asyncio.gather(
                scope([{ 'user': 'USER MESSAGE 1.', 'assistant': None}]),
                scope([{ 'user': 'USER MESSAGE 2.', 'assistant': None}])
            )

        # If answer_2 errors first, then 0
        # If answer_1 completes first, then 1
        assert scope.duration in [0, 1]

        # Check that the GenerateError was captured
        assert isinstance(scope.error, OfflineError)
