
__all__ = ['TGIClient', 'VLLMClient', 'AbstractClient', 'clients']

from typing import Type

from .tgi import TGIClient
from .vllm import VLLMClient
from ._abstract_client import AbstractClient

clients: dict[str, Type[AbstractClient]] = {
    'TGI': TGIClient,
    'VLLM': VLLMClient
}
