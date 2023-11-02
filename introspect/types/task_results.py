
from typing import TypeVar, Generic, TypedDict, Required, Literal

LabelType = TypeVar('LabelType', bound=str)
PredictType = TypeVar('PredictType', bound=str)

class TaskResult(TypedDict, Generic[LabelType]):
    duration: Required[float]
    label: Required[LabelType]

class PartialClassifyResult(TypedDict, Generic[PredictType]):
    paragraph: Required[str|None]
    predict_source: Required[str|None]
    predict: Required[PredictType|None]
    correct: Required[bool|None]

class ClassifyResult(TaskResult[LabelType], PartialClassifyResult[PredictType]):
    pass

class PartialIntrospectResult(PartialClassifyResult[PredictType], Generic[PredictType]):
    paragraph: Required[str|None]
    ability_source: Required[str|None]
    ability: Required[Literal['yes', 'no']|None]
    introspect: Required[bool|None]

class IntrospectResult(TaskResult[LabelType], PartialIntrospectResult[PredictType]):
    pass

class PartialFaithfulResult(PartialClassifyResult[PredictType], Generic[PredictType]):
    paragraph: Required[str|None]
    explain_source: Required[str|None]
    explain: Required[str|None]
    explain_predict_source: Required[str|None]
    explain_predict: Required[PredictType|None]
    faithful: Required[bool|None]

class FaithfulResult(TaskResult[LabelType], PartialFaithfulResult[PredictType]):
    pass
