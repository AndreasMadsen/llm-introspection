
__all__ = ['FalconModel', 'Llama2Model',
           'models', 'AbstractModel']

from typing import Type, Mapping

from ._abstract_model import AbstractModel
from .falcon import FalconModel
from .llama2 import Llama2Model

models: Mapping[str, Type[AbstractModel]] = {
    Model._name: Model
    for Model
    in [FalconModel, Llama2Model]
}
