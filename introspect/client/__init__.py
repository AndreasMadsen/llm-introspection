
__all__ = ['TGIClient', 'VLLMClient', 'OfflineClient', 'AbstractClient', 'OfflineError', 'clients']

from typing import Type

from .tgi import TGIClient
from .vllm import VLLMClient
from .offline import OfflineError, OfflineClient
from ._abstract_client import AbstractClient

clients: dict[str, Type[AbstractClient]] = {
    'TGI': TGIClient,
    'VLLM': VLLMClient,
    'Offline': OfflineClient
}
