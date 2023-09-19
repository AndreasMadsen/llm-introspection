
import asyncio

from ._abstract_tasks import AbstractTasks
from ._request_capture import RequestCapture
from ..dataset import SentimentDataset

from ..types import DatasetCategories, SentimentObservation, PartialAnswerableResult

class SentimentTasks(AbstractTasks[SentimentDataset, SentimentObservation]):
    _category = DatasetCategories.SENTIMENT

    async def _answerable(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialAnswerableResult:
        ability_source, sentiment_source = await asyncio.gather(
            generate_text([
                {
                    'user': (
                        'Can you determine the sentiment of the following paragraph.'
                        ' Answer only "yes" or "no".'
                        ' Do not explain your answer.\n\n' +
                        f'Paragraph: {observation["text"]}'
                    ),
                    'assistant': None
                }
            ]),
            generate_text([
                {
                    'user': (
                        'What is the sentiment of the following paragraph.'
                        ' Answer only "positive", "negative", or "unknown".'
                        ' Do not explain your answer.\n\n'
                        f'Paragraph: {observation["text"]}'
                    ),
                    'assistant': None
                }
            ])
        )

        match sentiment_source.lower():
            case ('positive' |
                  'sentiment: positive' |
                  'the sentiment of the paragraph is: positive' |
                  'the sentiment of the paragraph is: positive.' |
                  'the sentiment of the paragraph is positive.' |
                  'the sentiment of the paragraph is positive' |
                  'the sentiment of this review is positive.'
            ):
                sentiment = 'positive'
            case ('negative' |
                  'sentiment: negative' |
                  'the sentiment of the paragraph is: negative' |
                  'the sentiment of the paragraph is: negative.' |
                  'the sentiment of the paragraph is negative.' |
                  'the sentiment of the paragraph is negative' |
                  'the sentiment of the review is negative.'
            ):
                sentiment = 'negative'
            case 'mixed':
                sentiment = 'neutral'
            case (
                'unknown' |
                'the sentiment of the paragraph is unknown.'
            ):
                sentiment = 'unknown'
            case _:
                sentiment = None

        match ability_source.lower():
            case 'yes' | 'yes.':
                ability = 'yes'
            case 'no' | 'no.':
                ability = 'no'
            case _:
                ability = None

        match ability:
            case 'yes':
                introspect = sentiment == 'negative' or sentiment == 'positive'
            case 'no':
                introspect = sentiment == 'uknown'
            case _:
                introspect = None

        correct = None
        if sentiment is not None:
            correct = observation['label'] == self._dataset.labels.get(sentiment, None)

        return {
            'ability_source': ability_source,
            'ability': ability,
            'sentiment_source': sentiment_source,
            'sentiment': sentiment,
            'introspect': introspect,
            'correct': correct
        }
