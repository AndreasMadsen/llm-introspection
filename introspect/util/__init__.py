
__all__ = ['AsyncMap', 'generate_experiment_id',
           'default_model_id', 'default_model_type', 'default_system_message',
           'cancel_eventloop_on_signal']

import sys as _sys

from .experiment_id import generate_experiment_id
from .default_args import default_model_id, default_model_type, default_system_message

# On the login node, python is not new enough to support some features
# required for these packages. We anyway only need generate_experiment_id,
# on login nodes for the `experiment_id.py` script. So just avoid importing
# them.
if _sys.version_info >= (3, 11):
    from .async_map import AsyncMapIterable as AsyncMap
    from .signal_handler import cancel_eventloop_on_signal
