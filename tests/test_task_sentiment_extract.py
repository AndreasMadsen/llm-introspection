
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

def test_task_sentiment_extract_ability(task: SentimentClassifyTask):
    c = task._extract_ability

    # yes
    assert c('Yes') == 'yes'
    assert c('yes') == 'yes'
    assert c('Yes.') == 'yes'
    assert c('yes.') == 'yes'

    # no
    assert c('No') == 'no'
    assert c('no') == 'no'
    assert c('No.') == 'no'
    assert c('no.') == 'no'

    # failure
    assert c('positive') == None
    assert c('negative') == None

def test_task_sentiment_extract_paragraph(task: SentimentClassifyTask):
    c = task._extract_paragraph

    p = ('This film is reminiscent of Godard\'s Masculin, féminin, and I highly'
         ' recommend checking out both films. The film boasts two standout'
         ' elements - the natural acting and the striking photography. While some'
         ' may find certain aspects of the film silly, I believe it\'s a charming'
         ' and thought-provoking exploration of human relationships. Lena Nyman'
         ' delivers an endearing performance, bringing depth and nuance to her'
         ' character. Unlike Godard\'s work, which can sometimes feel overly'
         ' cerebral, this film offers a refreshing change of pace. It\'s a'
         ' delightful reflection of the era and culture that produced it.'
         ' Rating: 8/10.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Extra double new-line
    p = ('If the talented team behind "Zombie Chronicles" reads this, here\'s some feedback:\n'
         '\n'
         '1. Kudos for incorporating creative close-ups in the opening credits!'
         ' However, consider saving some surprises for later to enhance the twists.\n'
         '2. Your cast did an admirable job with the resources available. For'
         ' future projects, investing in acting workshops or casting experienced'
         ' actors could elevate the overall performance.'
         '3. The historical setting was a unique touch. To further enhance the'
         ' immersion, consider adding more period-specific details in set design'
         ' and costuming.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Contains quoutes
    p = ('I was very disappointed in [REDACTED]. I had been waiting a really long time'
         ' to see it and I finally got the chance when it re-aired Thursday night on'
         ' [REDACTED]. I love the first three "[REDACTED]" movies but this one was'
         ' nothing like I thought it was going to be. The whole movie was [REDACTED]'
         ' and depressing, there were way too many [REDACTED], and the editing was'
         ' very poor - too many scenes out of context. I also think the death of'
         ' [REDACTED] happened way too soon and [REDACTED]\'s appearance in the movie'
         ' just didn\'t seem to fit. It seemed like none of the actors really wanted'
         ' to be there - they were all lacking [REDACTED]. There seemed to be no'
         ' interaction between [REDACTED] and [REDACTED] at all.')
    assert c(f'Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p

def test_task_sentiment_extract_list_content(task: SentimentClassifyTask):
    c = task._extract_list_content

    # ordered
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '1. Awful\n'
             '2. Worst') == ['Awful', 'Worst']

    # unordered
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* Awful\n'
             '* Worst') == ['Awful', 'Worst']

    # qouted
    assert c('Sure! Here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* "fun" (appears twice)\n'
             '* "great" (appears three times)\n'
             '* "spectacular"\n'
             '* "superb"\n'
             '* "totally recommend"\n'
             '* "talent"\n') == ['fun', 'great', 'spectacular', 'superb', 'totally recommend', 'talent']

    # Fancy dots and tabs
    assert c('Sure! Here are the most important words for determining the sentiment of the given paragraph:\n'
             '\n'
             '•\tFugly\n'
             '•\tAnnoying\n'
             '•\tAwful\n') == ['Fugly', 'Annoying', 'Awful']
