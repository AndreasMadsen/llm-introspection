
from functools import cached_property


class _AnnotationMapping(dict):
    @cached_property
    def breaks(self):
        return list(self.keys())

    @cached_property
    def labels(self):
        return list(self.values())

    def labeller(self, key):
        return self.get(key, key)

    def __or__(self, other):
        return _AnnotationMapping(dict.__or__(self, other))

    def __ror__(self, other):
        return _AnnotationMapping(dict.__ror__(self, other))

    def __ior__(self, other):
        return super.__ior__(self, other) # type: ignore


class _AllAnnotations():
    def __init__(self) -> None:
        self.predicted_sentiment = _AnnotationMapping({
            'negative': 'Negative',
            'neutral': 'Neutral',
            'positive': 'Positive',
            'unknown': 'Unknown'
        })
        self.target_sentiment = _AnnotationMapping({
            'negative': 'Negative',
            'positive': 'Positive'
        })
        self.persona = _AnnotationMapping({
            'human-persona': 'Human persona',
            'you-persona': 'You persona',
            'no-persona': 'No persona'
        })
        self.redact = _AnnotationMapping({
            'no-redact': 'No redact desc.',
            'redact': 'Has redact desc.'
        })
        self.counterfactual_target = _AnnotationMapping({
            'explicit': 'Explicit target',
            'implicit': 'Implicit target'
        })
        self.prompt_length = _AnnotationMapping({
            'short': 'Short variation',
            'long': 'Long variation'
        })
        self.ability = _AnnotationMapping({
            'yes': 'Yes',
            'no': 'No'
        })
        self.answerable_options = _AnnotationMapping({
            'options': 'Choices given',
            'no-options': 'No choices given'
        })

annotation = _AllAnnotations()
