
import pathlib

import pytest

from introspect.model import Llama2Model
from introspect.client import OfflineClient
from introspect.dataset import IMDBDataset
from introspect.tasks import SentimentClassifyTask

@pytest.fixture
def task() -> SentimentClassifyTask:
    return SentimentClassifyTask(
        Llama2Model(OfflineClient())
    )

def test_task_sentiment_process_redact_words(task: SentimentClassifyTask):
    r = lambda content, words: task._process_redact_words(
        {'text': content, 'label': 'negative', 'idx': 0},
        words
    )

    p = ('I opted to see the film at the recent Dubai Film Festival because'
         ' it had been selected to the Cannes film festival\'s prestigious'
         ' Competition section. I was surprised that Cannes could be so off'
         ' the mark in judging quality.<br /><br />The film, some reviewers,'
         ' have noted does not have too much of gunfire--but the inherent'
         ' violence is repulsive. Imagine killing your enemy/competitor in'
         ' front of your young son..or forcing someone to eat a porcelain'
         ' spoon to prove loyalty. There are some hints of the contrasting'
         ' Corleone sons in Copolla\'s "Godfather" that seem to resurface'
         ' here in this Chinese/Hong Kong film but the quality of the two'
         ' are as distinctly different as chalk and cheese.<br /><br />This'
         ' film is only recommended for violence junkies..there is no great'
         ' cinema here. At best it might be considered to be better than'
         ' the usual Run Run Shaw production for production values.')
    w = ["prestigious", "off the mark", "repulsive", "violence junkies",
         "no great cinema", "Run Run Shaw production"]
    e = ('I opted to see the film at the recent Dubai Film Festival because'
         ' it had been selected to the Cannes film festival\'s [REDACTED]'
         ' Competition section. I was surprised that Cannes could be so'
         ' [REDACTED] in judging quality.<br /><br />The film, some reviewers,'
         ' have noted does not have too much of gunfire--but the inherent'
         ' violence is [REDACTED]. Imagine killing your enemy/competitor'
         ' in front of your young son..or forcing someone to eat a porcelain'
         ' spoon to prove loyalty. There are some hints of the contrasting'
         ' Corleone sons in Copolla\'s "Godfather" that seem to resurface'
         ' here in this Chinese/Hong Kong film but the quality of the two'
         ' are as distinctly different as chalk and cheese.<br /><br />This'
         ' film is only recommended for [REDACTED]..there is [REDACTED] here.'
         ' At best it might be considered to be better than the usual [REDACTED]'
         ' for production values.')
    assert r(p, w) == e

def test_task_sentiment_process_redact_words_synthetic(task: SentimentClassifyTask):
    r = lambda content, words: task._process_redact_words(
        {'text': content, 'label': 'negative', 'idx': 0},
        words
    )

    p = ('First word, isn\'t always the "first" word')
    # check first word and qouted context
    assert r(p, ['first']) == '[REDACTED] word, isn\'t always the "[REDACTED]" word'
    # check cass insensitive
    assert r(p, ['First']) == '[REDACTED] word, isn\'t always the "[REDACTED]" word'
    # check last word
    assert r(p, ['word']) == 'First [REDACTED], isn\'t always the "first" [REDACTED]'
    # check case insensitive
    assert r(p, ['WORD']) == 'First [REDACTED], isn\'t always the "first" [REDACTED]'
    # check bounding word
    assert r(p, ['is']) == 'First word, isn\'t always the "first" word'
    # check non-word charecter in 'word'
    assert r(p, ['isn\'t']) == 'First word, [REDACTED] always the "first" word'
    # check space charecter in 'word'
    assert r(p, ['isn\'t always']) == 'First word, [REDACTED] the "first" word'
