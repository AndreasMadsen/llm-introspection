
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


predicted_sentiment = _AnnotationMapping({
    'negative': 'Negative',
    'neutral': 'Neutral',
    'positive': 'Positive',
    'unknown': 'Unknown'
})
target_sentiment = _AnnotationMapping({
    'negative': 'Negative',
    'positive': 'Positive'
})
persona = _AnnotationMapping({
    'human': 'Human',
    'you': 'You',
    'objective': 'Objective'
})
redact = _AnnotationMapping({
    'no-redact': 'None',
    'redacted': '"redacted"',
    'removed': '"removed"'
})
redact_token = _AnnotationMapping({
    'redacted': '"redacted"',
    'removed': '"removed"'
})
counterfactual_target = _AnnotationMapping({
    'explicit': 'Explicit target',
    'implicit': 'Implicit target'
})
prompt_length = _AnnotationMapping({
    'short': 'Short variation',
    'long': 'Long variation'
})
ability = _AnnotationMapping({
    'yes': 'Yes',
    'no': 'No'
})
answerable_options = _AnnotationMapping({
    'options': 'Choices given',
    'no-options': 'No choices given'
})
