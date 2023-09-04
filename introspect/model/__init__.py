
__all__ = ['FalconModel', 'Llama2Model',
           'models']

from typing import Type

from ._abstract_model import AbstractModel
from .falcon import FalconModel
from .llama2 import Llama2Model

models: dict[str, Type[AbstractModel]] = {
    Model._name: Model
    for Model
    in [FalconModel, Llama2Model]
}
