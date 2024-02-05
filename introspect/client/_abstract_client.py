from abc import ABCMeta, abstractmethod
import asyncio
import time
from typing import TypedDict, Generic, TypeVar, Iterable

from ..types import GenerateConfig, GenerateResponse, GenerateError, OfflineError
from ..database import GenerationCache

InfoType = TypeVar('InfoType', bound=TypedDict)

class RetryRequest(Exception):
    pass

class AbstractClient(Generic[InfoType], metaclass=ABCMeta):
    _record: list[tuple[str, GenerateResponse]]

    def __init__(self, base_url: str, cache: GenerationCache|None = None,
                 connect_timeout_sec: int=60*60, max_reconnects: int=5,
                 record=False) -> None:
        """Create a client that can be used to run a generative inference

        Note that the client is backed by a cache. This cache is checked for the prompt first
        and only if there is no entry in the cache, does the request a computed response. This
        computed response is then cached afterwards.

        Args:
            base_url (str): The url to the endpoint. For example: "http://localhost:8080"
            cache (GenerationCache | None, optional): Cache where generation outputs are stored. Defaults to None.
            connect_timeout_sec (int, optional): How long to wait for the server to start. Defaults to 30*60.
            max_reconnects (int, optional): The number of times the connection can be lost. Default to 3.
            record (bool, optional). Record inputs and outputs, this is only useful for testing or debugging. Default False.
        """
        self._base_url = base_url
        self._connect_timeout_sec = connect_timeout_sec
        self._cache = cache
        self._is_connected = False
        self._on_connection = None
        self._remaning_reconnects = max_reconnects

        self._record_enabled = record
        self._record = []

    @property
    def record(self) -> Iterable[tuple[str, GenerateResponse]]:
        """If recording is enabled, this returns a list of prompts and responses

        Raises:
            ValueError: raises an error if recording is disabled

        Returns:
            Iterable[tuple[str, GenerateResponse]]: Iterable of (prompt, response).
        """
        if not self._record_enabled:
            raise ValueError('recording is not enabled')

        return ((prompt, response) for prompt, response in self._record)

    @property
    def prompt_record(self) -> Iterable[str]:
        """If recording is enabled, this returns a list of prompts queried

        Raises:
            ValueError: raises an error if recording is disabled

        Returns:
            Iterable[str]: Iterable of prompts
        """
        if not self._record_enabled:
            raise ValueError('recording is not enabled')

        return (prompt for prompt, response in self._record)

    @property
    def response_record(self) -> Iterable[GenerateResponse]:
        """If recording is enabled, this returns a list of responses

        Raises:
            ValueError: raises an error if recording is disabled

        Returns:
            Iterable[str]: Iterable of responses
        """
        if not self._record_enabled:
            raise ValueError('recording is not enabled')

        return (response for prompt, response in self._record)

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
        """This future returns when a connection is establed.

        The function tries to connect to the server several times, until
        `_connect_timeout_sec` has expired. Each connnection attempt is
        handled by `_try_connect.

        Args:
            presleep (int, optional): Time to sleep before first connection attempt. Defaults to 0.

        Raises:
            IOError: raises if the allowed time expired.
        """
        start_time = time.time()

        if presleep > 0:
            await asyncio.sleep(presleep)

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

    def _handle_disconnect(self):
        """Renew state assuming the connection is lost
        """
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

        # Query a resonse and manage the record if recording is enabled
        response = await self._read_cache_or_generate(prompt, config_with_defaults)
        if self._record_enabled:
            self._record.append((prompt, response))
        return response

    async def _read_cache_or_generate(self, prompt: str, config: GenerateConfig) -> GenerateResponse:
        # Return valid response from cache, if it exists
        cached_answer = await self._get_cache(prompt)
        if cached_answer is not None and not isinstance(cached_answer, GenerateError):
            return cached_answer

        # No valid response in cache (might not exists, might be an previous error).
        # Attempt to compute response.
        if not self._is_connected:
            await self.connect()

        # compute response
        try:
            computed_answer = await self._generate(prompt, config)
        except GenerateError as error:
            # A GenerateError is often because the prompt is too long for the model.
            # These are are errors that do not indicate an issue with the server and
            # should not crash the client.
            computed_answer: GenerateResponse|GenerateError = error
        except RetryRequest as error:
            # A RetryRequest indicates that the server crashed, maybe due to a OOM bug.
            # Such errors are handled by a server wrapper, which will restart the server.
            # The RetryRequest request indicates that the server is disconnected, and we
            # need to wait until the server has restarted.
            self._handle_disconnect()
            # Retry this function. This will wait until the server has restarted.
            return await self.generate(prompt, config)

        match computed_answer:
            case OfflineError():
                # In case of an OfflineError, relay the cached error if such an error exist.
                raise computed_answer from cached_answer

            case GenerateError():
                # Save a GenerateError to the cache, such it can be relayed if an Offlineclient is used.
                await self._put_cache(prompt, computed_answer)
                raise computed_answer

            case _:
                # There were no error, update the cache and return the regular response
                await self._put_cache(prompt, computed_answer)
                return computed_answer
