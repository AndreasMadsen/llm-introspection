
from pytest_httpserver import HTTPServer
import pytest

from introspect.database import GenerationCache
from introspect.types import OfflineError, GenerateResponse
from introspect.client import OfflineClient, TGIClient, VLLMClient

@pytest.mark.asyncio
async def test_client_offline_error():
    client = OfflineClient()
    with pytest.raises(OfflineError):
        await client.generate('Hello', {})

@pytest.mark.asyncio
async def test_client_offline_cache():
    cached_answer: GenerateResponse = {
        'response': 'ASSISTANT MESSAGE ANSWER',
        'duration': 1
    }

    async with GenerationCache(':memory:') as cache:
        await cache.put('USER MESSAGE PROMPT', cached_answer)

        client = OfflineClient(cache=cache)

        # Return cache if exists
        answer = await client.generate('USER MESSAGE PROMPT', {})
        assert answer == cached_answer

        # Raise OfflineError if not
        with pytest.raises(OfflineError):
            await client.generate('MISSING USER MESSAGE PROMPT: ', {})

@pytest.mark.asyncio
async def test_client_tgi_request(httpserver: HTTPServer):
    httpserver.expect_request("/health").respond_with_data('')
    httpserver.expect_request("/").respond_with_json([{
        'generated_text': 'MOCK RESPONSE'
    }], headers={
        'X-Inference-Time': '12'
    })

    client = TGIClient(httpserver.url_for(""))
    answer = await client.generate('MISSING USER MESSAGE PROMPT: ', {})
    assert answer == {
        'response': 'MOCK RESPONSE',
        'duration': 12
    }

@pytest.mark.asyncio
async def test_client_tgi_info(httpserver: HTTPServer):
    httpserver.expect_request("/health").respond_with_data('')
    httpserver.expect_request("/info").respond_with_json({
        'model_id': 'mock'
    })

    client = TGIClient(httpserver.url_for(""))
    assert await client.info() == {
        'model_id': 'mock'
    }

@pytest.mark.asyncio
async def test_client_vllm_request(httpserver: HTTPServer):
    httpserver.expect_request("/generate").respond_with_json({
        'text': ['MISSING USER MESSAGE PROMPT: MOCK RESPONSE']
    })

    client = VLLMClient(httpserver.url_for(""))
    answer = await client.generate('MISSING USER MESSAGE PROMPT: ', {})
    assert answer == {
        'response': 'MOCK RESPONSE',
        'duration': answer['duration']
    }
    assert answer['duration'] > 0

@pytest.mark.asyncio
async def test_client_vllm_info(httpserver: HTTPServer):
    client = VLLMClient(httpserver.url_for(""))
    httpserver.expect_request("/generate").respond_with_json({ 'text': [''] })
    assert await client.info() == { }
