
import pathlib

import pytest

from introspect.model import Llama2Model
from introspect.client import OfflineClient
from introspect.dataset import IMDBDataset
from introspect.tasks import SentimentClassifyTask

@pytest.fixture
def task() ->SentimentClassifyTask:
    return SentimentClassifyTask(
        IMDBDataset(persistent_dir=pathlib.Path('.')),
        Llama2Model(OfflineClient())
    )

def test_task_sentiment_extract_sentiment(task: SentimentClassifyTask):
    c = task._extract_sentiment

    # positive
    assert c('sentiment: positive') == 'positive'
    assert c('The sentiment of the paragraph is: positive') == 'positive'
    assert c('the sentiment of the paragraph is: positive') == 'positive'
    assert c('the sentiment of the paragraph is: positive.') == 'positive'
    assert c('the sentiment of the paragraph is positive.') == 'positive'
    assert c('the sentiment of the paragraph is positive') == 'positive'
    assert c('the sentiment of this review is positive.') == 'positive'

    # negative
    assert c('negative') == 'negative'
    assert c('sentiment: negative') == 'negative'
    assert c('The sentiment of the paragraph is: negative') == 'negative'
    assert c('the sentiment of the paragraph is: negative') == 'negative'
    assert c('the sentiment of the paragraph is: negative.') == 'negative'
    assert c('the sentiment of the paragraph is negative.') == 'negative'
    assert c('the sentiment of the paragraph is negative') == 'negative'
    assert c('the sentiment of the review is negative.') == 'negative'
    assert c('The sentiment of the review is negative.'
             ' While the reviewer acknowledges that the film is'
             ' visually arresting and offers a unique take on the opera,'
             ' they ultimately find it overwhelming and unsatisfying,'
             ' criticizing the director\'s use of excessive symbolism'
             ' and failure to stay true to Wagner\'s original vision.'
             ' They also express frustration with the film\'s unevenness'
             ' and lack of coherence, suggesting that it fails to effectively'
             ' convey the depth and complexity of the opera. Overall, the'
             ' review is written in a dismissive tone, advising readers'
             ' to approach the film with caution and recommending alternative'
             ' recordings of the opera instead.') == 'negative'

    # neutral
    assert c('Mixed') == 'neutral'
    assert c('mixed') == 'neutral'
    assert c('The sentiment of the paragraph is "mixed".') == 'neutral'
    assert c('The sentiment of the paragraph is "neutral"'
             ' as the author expresses mixed feelings towards'
             ' the TV show "Father of the Pride." They mention both'
             ' positive and negative aspects of the show, but ultimately'
             ' conclude that they do not have strong feelings'
             ' towards it one way or the other.')

    # unknown
    assert c('unknown') == 'unknown'
    assert c('The sentiment of the paragraph is unknown.') == 'unknown'
    assert c('the sentiment of the paragraph is unknown.') == 'unknown'
    assert c('The sentiment of the paragraph is unknown.'
             ' It describes both positive and negative aspects'
             ' of the movie, making it difficult to determine'
             ' the overall sentiment.') == 'unknown'
    assert c('The sentiment of the paragraph is unknown.'
             ' While the writer mentions some positive aspects of the movie,'
             ' such as the music and the storyline, they also mention negative'
             ' emotions they experienced while watching it, including sadness'
             ' and melancholy. Therefore, it cannot be classified as purely'
             ' positive or negative.') == 'unknown'

    # failure
    assert c('yes') == None
    assert c('no') == None
