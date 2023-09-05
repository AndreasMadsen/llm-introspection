
import asyncio
import time

import aiohttp
from text_generation import AsyncClient
from text_generation.types import Response

from ..types import GenerateConfig

class TGIClient:
    def __init__(self, base_url: str, *args, connect_timeout_sec: int=30*60, **kwargs) -> None:
        """Create a client that can be used to run a generative inference

        Args:
            base_url (str): The url to the endpoint. For example: "http://localhost:8080"
            connect_timeout_sec (int, optional): _description_. Defaults to 30*60.
        """
        self._base_url = base_url
        self._connect_timeout_sec = connect_timeout_sec
        self._client = AsyncClient(base_url, *args, **kwargs)

        self._is_connected = False
        self._on_connection = None

    async def _await_connection(self):
        start_time = time.time()

        while time.time() < start_time + self._connect_timeout_sec:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
                try:
                    async with session.get(f'{self._base_url}/health') as response:
                        if response.status == 200:
                            self._is_connected = True
                            return
                except aiohttp.ClientOSError:
                    pass

                await asyncio.sleep(10)

        raise IOError(f'Could not connect to {self._base_url}')

    async def connect(self):
        if self._on_connection is None:
            self._on_connection = asyncio.create_task(self._await_connection())
        await self._on_connection

    async def generate(self, prompt: str, config: GenerateConfig) -> Response:
        """Run inference on the generative model.

        Args:
            prompt (str): The prompt to generate from.
            config (GenerateConfig): The configuration which controls the generative algorithm
                (e.g. beam-search) and the response format.

        Returns:
            Response: The generated content, including optional details.
        """
        if not self._is_connected:
            await self.connect()

        return await self._client.generate(prompt, **config)
