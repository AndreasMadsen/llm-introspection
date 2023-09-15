
import asyncio

from ._abstract_tasks import AbstractTasks, RequestCapture
from ..dataset import SentimentDataset

from ..types import DatasetCategories, SentimentObservation, PartialAnswerableResult

class SentimentTasks(AbstractTasks[SentimentDataset, SentimentObservation]):
    _category = DatasetCategories.SENTIMENT

    async def _answerable(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialAnswerableResult:
        answer_ability, answer_sentiment = await asyncio.gather(
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

        answer_ability_lower = answer_ability.lower()
        answer_sentiment_lower = answer_sentiment.lower()

        match answer_sentiment_lower:
            case ('positive' |
                  'sentiment: positive' |
                  'the sentiment of the paragraph is: positive' |
                  'the sentiment of the paragraph is: positive.' |
                  'the sentiment of the paragraph is positive.' |
                  'the sentiment of the paragraph is positive' |
                  'the sentiment of this review is positive.'
            ):
                correct = observation['label'] == self._dataset.labels['positive']
                decided = True
            case ('negative' |
                  'sentiment: negative' |
                  'the sentiment of the paragraph is: negative' |
                  'the sentiment of the paragraph is: negative.' |
                  'the sentiment of the paragraph is negative.' |
                  'the sentiment of the paragraph is negative' |
                  'the sentiment of the review is negative.'
            ):
                correct = observation['label'] == self._dataset.labels['negative']
                decided = True
            case 'mixed':
                correct = False
                decided = True
            case (
                'unknown' |
                'the sentiment of the paragraph is unknown.'
            ):
                correct = False
                decided = False
            case _:
                correct = None
                decided = None

        match answer_ability_lower:
            case 'yes' | 'yes.':
                introspect = None if decided is None else decided
            case 'no' | 'no.':
                introspect = None if decided is None else not decided
            case _:
                introspect = None

        return {
            'answer_ability': answer_ability,
            'answer_sentiment': answer_sentiment,
            'introspect': introspect,
            'correct': correct
        }
