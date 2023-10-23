
from enum import StrEnum

class TaskCategories(StrEnum):
    CLASSIFY = 'classify'
    ANSWERABLE = 'answerable'
    COUNTERFACTUAL = 'counterfactual'
    REDACTED = 'redacted'
    IMPORTANCE = 'importance'
