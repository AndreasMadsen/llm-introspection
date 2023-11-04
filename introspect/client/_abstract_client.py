from abc import ABCMeta, abstractmethod
import asyncio
import time
from typing import TypedDict, Generic, TypeVar

from ..types import GenerateConfig, GenerateResponse, GenerateError, OfflineError
from ..database import GenerationCache

InfoType = TypeVar('InfoType', bound=TypedDict)

class RetryRequest(Exception):
    pass

class AbstractClient(Generic[InfoType], metaclass=ABCMeta):
    def __init__(self, base_url: str, cache: GenerationCache|None = None,
                 connect_timeout_sec: int=30*60, max_reconnects: int=5) -> None:
        """Create a client that can be used to run a generative inference

        Args:
            base_url (str): The url to the endpoint. For example: "http://localhost:8080"
            cache (GenerationCache | None, optional): Cache where generation outputs are stored. Defaults to None.
            connect_timeout_sec (int, optional): How long to wait for the server to start. Defaults to 30*60.
            max_reconnects (int, optional): The number of times the connection can be lost. Default to 3.
        """
        self._base_url = base_url
        self._connect_timeout_sec = connect_timeout_sec
        self._cache = cache
        self._is_connected = False
        self._on_connection = None
        self._remaning_reconnects = max_reconnects

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
    async def _generate(self, prompt: str, config: GenerateConfig) -> GenerateResponse:
        ...

    async def _await_connection(self, presleep=0):
        start_time = time.time()

        if presleep > 0:
            await asyncio.sleep(presleep)

        while time.time() < start_time + self._connect_timeout_sec:
            if await self._try_connect():
                self._is_connected = True
                print('connection made')
                return

            await asyncio.sleep(10)

        raise IOError(f'Could not connect to {self._base_url}')

    async def connect(self):
        """Complete when server is running
        """
        if self._on_connection is None:
            self._on_connection = asyncio.create_task(self._await_connection())
        await self._on_connection

    def _handle_disconnect(self):
        """Renew state assuming the connection is lost
        """
        print('handle disconnect')
        if self._remaning_reconnects <= 0:
            raise IOError('Exhaused all allowed reconnection attempts')

        if self._is_connected:
            self._is_connected = False
            self._remaning_reconnects -= 1
            self._on_connection = asyncio.create_task(self._await_connection(presleep=10))

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
            computed_answer = await self._generate(prompt, config_with_defaults)
        except GenerateError as error:
            computed_answer: GenerateResponse|GenerateError = error
        except RetryRequest as error:
            # The _generate failed and requested a reconnect.
            self._handle_disconnect()
            return await self.generate(prompt, config)

        match computed_answer:
            case OfflineError():
                raise computed_answer from cached_answer

            case GenerateError():
                await self._put_cache(prompt, computed_answer)
                raise computed_answer

            case _:
                await self._put_cache(prompt, computed_answer)
                return computed_answer
