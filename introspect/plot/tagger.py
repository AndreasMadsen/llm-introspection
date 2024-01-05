
import re
from typing import Literal

def classify_persona(task_config: list[str]) -> Literal['human', 'you', 'objective']:
    if 'c-persona-human' in task_config:
        return 'human'
    elif 'c-persona-you' in task_config:
        return 'you'
    else:
        return 'objective'

def classify_redact(task_config: list[str]) -> Literal['no-redact', 'removed', 'redacted']:
    if 'c-no-redacted' in task_config:
        return 'no-redact'
    elif 'm-removed' in task_config:
        return 'removed'
    else:
        return 'redacted'

def explain_persona(task_config: list[str]) -> Literal['human', 'you', 'objective']|None:
    if 'e-persona-human' in task_config and 'c-persona-human' in task_config:
        return 'human'
    elif 'e-persona-you' in task_config and 'c-persona-you' in task_config:
        return 'you'
    elif ('e-persona-you' not in task_config and 'c-persona-you' not in task_config and
            'e-persona-human' not in task_config and 'c-persona-human' not in task_config):
        return 'objective'

    return None

def explain_counterfactual_target(task_config: list[str]) -> Literal['implicit', 'explicit']:
    if 'e-implcit-target' in task_config:
        return 'implicit'
    else:
        return 'explicit'

def explain_redact(task_config: list[str]) -> Literal['no-redact', 'removed', 'redacted']:
    return classify_redact(task_config)

def model_size(model_name: str) -> int|None:
    if m := re.match(r'([a-z0-9-]+)-([1-9][0-9]*)b$', model_name, flags=re.IGNORECASE):
        return int(m.group(2))
    else:
        return None

def model_name(model_name: str) -> str|None:
    if m := re.match(r'([a-z0-9-]+)-([1-9][0-9]*)b$', model_name, flags=re.IGNORECASE):
        return m.group(1)
    else:
        return None
