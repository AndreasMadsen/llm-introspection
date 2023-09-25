
from enum import StrEnum

class TaskCategories(StrEnum):
    ANSWERABLE = 'answerable'
    COUNTERFACTUAL = 'counterfactual'
    REDACTED = 'redacted'
