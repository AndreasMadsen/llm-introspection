
from typing import TypedDict, Literal, Required, NotRequired

import aiohttp
import asyncio
from text_generation.types import Details

from ..types import GenerateConfig, GenerateResponse
from ._abstract_client import AbstractClient
from text_generation.errors import parse_error
from text_generation.types import Response


class TGIInfo(TypedDict):
  docker_label: Required[str|None]
  max_batch_total_tokens: Required[int]
  max_best_of: Required[int]
  max_concurrent_requests: Required[int]
  max_input_length: Required[int]
  max_stop_sequences: Required[int]
  max_total_tokens: Required[int]
  max_waiting_tokens: Required[int]
  model_device_type: Required[Literal['cuda', 'cpu']]
  model_dtype: Required[Literal['torch.float16', 'torch.bfloat16']]
  model_id: Required[str]
  model_pipeline_tag: Required[str|None]
  model_sha: Required[str|None]
  sha: Required[str]
  validation_workers: Required[int]
  version: Required[str]
  waiting_served_ratio: Required[float]

class TGIResponse(GenerateResponse):
    details: Required[Details|None]

class TGIGenerateConfig(GenerateConfig):
    # Default values:
    # https://github.com/huggingface/text-generation-inference/blob/0a63e9ab688cf715d31574ee5bb31025ff22ceec/router/src/main.rs#L29
    # https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/generation/configuration_utils.py#L39
    # https://github.com/huggingface/text-generation-inference/blob/main/router/src/lib.rs#L65

    # Activate logits sampling
    do_sample: NotRequired[bool]

    # Maximum number of generated tokens
    max_new_tokens: NotRequired[int]

    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: NotRequired[int]

    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: NotRequired[float]

    # Whether to prepend the prompt to the generated text
    return_full_text: NotRequired[bool]

    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: NotRequired[list[str]]

    # The value used to module the logits distribution.
    temperature: NotRequired[float]

    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: NotRequired[int]

    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: NotRequired[float]

    # truncate inputs tokens to the given size
    truncate: NotRequired[int]

    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: NotRequired[float]

    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: NotRequired[bool]

    # Get decoder input token logprobs and ids
    decoder_input_details: NotRequired[bool]

    # Get generation details
    details: NotRequired[bool]

class TGIGeneratePayload(TypedDict):
    inputs: Required[str]
    parameters: NotRequired[TGIGenerateConfig]
    stream: Literal[False]

class TGIClient(AbstractClient[TGIInfo, TGIResponse]):
    async def _try_connect(self) -> bool:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            try:
                async with session.get(f'{self._base_url}/health') as response:
                    return response.status == 200
            except (aiohttp.ClientOSError, asyncio.TimeoutError):
                return False

    async def _info(self) -> TGIInfo:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            async with session.get(f'{self._base_url}/info') as response:
                if response.status != 200:
                    raise RuntimeError(f'unexpected status code {response.status}')

                return await response.json()

    async def _generate(self, prompt, config) -> TGIResponse:
        payload: TGIGeneratePayload = {
            'inputs': prompt,
            'parameters': { **config },
            'stream': False
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            async with session.post(self._base_url, json=payload) as response:
                answer = await response.json()

                if response.status != 200:
                    raise parse_error(response.status, answer)

                details = None
                if 'details' in answer[0]:
                    details = Response(**answer[0]).details

                return {
                    'text': answer[0]['generated_text'],
                    'details': details
                }
