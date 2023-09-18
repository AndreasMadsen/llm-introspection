from abc import ABCMeta, abstractmethod
import asyncio
import time
from timeit import default_timer as timer
from typing import TypedDict, Generic, TypeVar

from ..types import GenerateConfig, GenerateResponse, GenerateError, OfflineError
from ..database import GenerationCache

InfoType = TypeVar('InfoType', bound=TypedDict)

class AbstractClient(Generic[InfoType], metaclass=ABCMeta):
    def __init__(self, base_url: str, cache: GenerationCache|None = None, connect_timeout_sec: int=30*60) -> None:
        """Create a client that can be used to run a generative inference

        Args:
            base_url (str): The url to the endpoint. For example: "http://localhost:8080"
            cache (GenerationCache | None, optional): Cache where generation outputs are stored. Defaults to None.
            connect_timeout_sec (int, optional): _description_. Defaults to 30*60.
        """
        self._base_url = base_url
        self._connect_timeout_sec = connect_timeout_sec
        self._cache = cache
        self._is_connected = False
        self._on_connection = None

    async def _get_cache(self, prompt) -> None|GenerateResponse|GenerateError:
        if self._cache is None:
            return None
        return await self._cache.get(prompt)

    async def _put_cache(self, prompt, answer: GenerateResponse|GenerateError) -> None:
        if self._cache is None:
            return
        await self._cache.put(prompt, answer)

    @abstractmethod
    async def _try_connect(self) -> bool:
        ...

    @abstractmethod
    async def _info(self) -> InfoType:
        ...

    @abstractmethod
    async def _generate(self, prompt: str, config: GenerateConfig) -> str:
        ...

    async def _await_connection(self):
        start_time = time.time()

        while time.time() < start_time + self._connect_timeout_sec:
            if await self._try_connect():
                self._is_connected = True
                return

            await asyncio.sleep(10)

        raise IOError(f'Could not connect to {self._base_url}')

    async def connect(self):
        """Complete when server is running
        """
        if self._on_connection is None:
            self._on_connection = asyncio.create_task(self._await_connection())
        await self._on_connection

    async def info(self) -> InfoType:
        """Get info about server
        """
        if not self._is_connected:
            await self.connect()

        return await self._info()

    async def generate(self, prompt: str, config: GenerateConfig) -> GenerateResponse:
        """Run inference on the generative model.

        Args:
            prompt (str): The prompt to generate from.
            config (GenerateConfig): The configuration which controls the generative algorithm
                (e.g. beam-search) and the response format.

        Returns:
            Response: The generated content, including optional details.
        """
        config_with_defaults: GenerateConfig = {
            **config,
            'max_new_tokens': config.get('max_new_tokens', 20),
            'best_of': config.get('best_of', 1),
            'stop': config.get('stop', []),
            'temperature': config.get('temperature', 1),
            'top_k': config.get('top_k', 50),
            'top_p': config.get('top_p', 1),
            'repetition_penalty': config.get('repetition_penalty', 1)
        }

        # Return valid response from cache, if it exists
        cached_answer = await self._get_cache(prompt)
        if cached_answer is not None and not isinstance(cached_answer, GenerateError):
            return cached_answer

        # No valid response in cache (might not exists, might be an error).
        # Attempt to compute response.

        if not self._is_connected:
            await self.connect()

        # compute response
        try:
            request_start_time = timer()
            response = await self._generate(prompt, config_with_defaults)
            durration = timer() - request_start_time
            computed_answer: GenerateResponse|GenerateError = {
                'response': response,
                'duration': durration
            }
        except GenerateError as error:
            computed_answer: GenerateResponse|GenerateError = error

        match computed_answer:
            case OfflineError():
                raise computed_answer from cached_answer

            case GenerateError():
                await self._put_cache(prompt, computed_answer)
                raise computed_answer

            case _:
                await self._put_cache(prompt, computed_answer)
                return computed_answer
