
from typing import TypedDict, Required, NotRequired
from timeit import default_timer as timer

import aiohttp
import asyncio

from ..types import GenerateResponse, GenerateConfig, GenerateError
from ._abstract_client import AbstractClient

class VLLMInfo(TypedDict):
    pass

class VLLMGenerateConfig(GenerateConfig):
    # n: Number of output sequences to return for the given prompt.
    n: NotRequired[int]

    # best_of: Number of output sequences that are generated from the prompt.
    #     From these `best_of` sequences, the top `n` sequences are returned.
    #     `best_of` must be greater than or equal to `n`. This is treated as
    #     the beam width when `use_beam_search` is True. By default, `best_of`
    #     is set to `n`.
    best_of: NotRequired[int]

    # presence_penalty: Float that penalizes new tokens based on whether they
    #     appear in the generated text so far. Values > 0 encourage the model
    #     to use new tokens, while values < 0 encourage the model to repeat
    #     tokens.
    presence_penalty: NotRequired[float]

    # frequency_penalty: Float that penalizes new tokens based on their
    #     frequency in the generated text so far. Values > 0 encourage the
    #     model to use new tokens, while values < 0 encourage the model to
    #     repeat tokens.
    frequency_penalty: NotRequired[float]

    # temperature: Float that controls the randomness of the sampling. Lower
    #     values make the model more deterministic, while higher values make
    #     the model more random. Zero means greedy sampling.
    temperature: NotRequired[float]

    # top_p: Float that controls the cumulative probability of the top tokens
    #     to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
    top_p: NotRequired[float]

    # top_k: Integer that controls the number of top tokens to consider. Set
    #     to -1 to consider all tokens.
    top_k: NotRequired[int]

    # use_beam_search: Whether to use beam search instead of sampling.
    use_beam_search: NotRequired[bool]

    # length_penalty: Float that penalizes sequences based on their length.
    #     Used in beam search.
    length_penalty: NotRequired[float]

    # early_stopping: Controls the stopping condition for beam search. It
    #     accepts the following values: `True`, where the generation stops as
    #     soon as there are `best_of` complete candidates; `False`, where an
    #     heuristic is applied and the generation stops when is it very
    #     unlikely to find better candidates; `"never"`, where the beam search
    #     procedure only stops when there cannot be better candidates
    #     (canonical beam search algorithm).
    early_stopping: NotRequired[bool|str]

    # stop: List of strings that stop the generation when they are generated.
    #     The returned output will not contain the stop strings.
    stop: NotRequired[list[str]]

    # ignore_eos: Whether to ignore the EOS token and continue generating
    #     tokens after the EOS token is generated.
    ignore_eos: NotRequired[bool]

    # max_tokens: Maximum number of tokens to generate per output sequence.
    max_tokens: NotRequired[int]

    # logprobs: Number of log probabilities to return per output token.
    logprobs: NotRequired[int]

class VLLMGeneratePayload(VLLMGenerateConfig):
    prompt: Required[str]

class VLLMError(Exception):
    pass

class VLLMClient(AbstractClient[VLLMInfo]):
    """VLLM Client.

    Although this client works, with the `python -m vllm.entrypoints.api_server` endpoint,
    the throughput is about 3x slower compared to TGI. The generation quality is also much
    worse.
    """
    async def _try_connect(self) -> bool:
        payload: VLLMGeneratePayload = {
            'prompt': 'Alive?',
            'max_tokens': 1,
            'best_of': 1,
            'stop': [],
            'temperature': 1,
            'top_k': 50,
            'top_p': 1
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            try:
                async with session.post(f'{self._base_url}/generate', json=payload) as response:
                    return response.status == 200
            except (aiohttp.ClientOSError, asyncio.TimeoutError):
                return False

    async def _info(self) -> VLLMInfo:
        return {}

    async def _generate(self, prompt: str, config: GenerateConfig) -> GenerateResponse:
        payload: VLLMGeneratePayload = {
            'prompt': prompt,
            'max_tokens': config.get('max_new_tokens', 0),
            'best_of': config.get('best_of', 0),
            'stop': config.get('stop', []),
            'temperature': config.get('temperature', 0),
            'top_k': config.get('top_k', 0),
            'top_p': config.get('top_p', 0),
            # Not exactly the same: https://github.com/vllm-project/vllm/issues/712#issuecomment-1672739448
            'presence_penalty': config.get('repetition_penalty', 0) - 1
        }

        try:
            request_start_time = timer()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(5 * 60)) as session:
                async with session.post(f'{self._base_url}/generate', json=payload) as response:
                    answer = await response.json()

                    if response.status != 200:
                        raise VLLMError(f'unexpected status code {response.status}')

                    response = answer['text'][0][len(prompt):]
                    durration = timer() - request_start_time
                    return {
                        'response': response,
                        'duration': durration
                    }

        except (VLLMError, asyncio.TimeoutError) as err:
            raise GenerateError('LLM generate failed') from err
