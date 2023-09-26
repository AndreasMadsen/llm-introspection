
from typing import Literal

from ._abstract_tasks import \
    AbstractTask, \
    IntrospectTask, FaithfulTask, \
    TaskResultType, PartialTaskResultType
from ._request_capture import RequestCapture

from ..dataset import SentimentDataset
from ..types import \
    TaskCategories, DatasetCategories, \
    SentimentObservation, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult

class SentimentTask(AbstractTask[SentimentDataset, SentimentObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.SENTIMENT

    async def _sentiment(
        self, parahraph: str, generate_text: RequestCapture
    ) -> tuple[str, Literal['positive', 'negative', 'neutral', 'unknown']|None]:

        sentiment_source = await generate_text([
            {
                'user': (
                    f'What is the sentiment of the following paragraph?'
                    f' Answer only "positive", "negative", or "unknown".'
                    f' Do not explain your answer.\n\n'
                    f'Paragraph: {parahraph}'
                ),
                'assistant': None
            }
        ])

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

        return (sentiment_source, sentiment)

    def _is_correct(self, observation: SentimentObservation, sentiment: str|None) -> bool|None:
        match sentiment:
            case None:
                return None
            case ('positive' | 'negative'):
                return observation['label'] == self._dataset.label_str2int[sentiment]
            case _:
                return False

    def _extract_explanation_paragraph(self, source: str) -> str|None:
        # The counterfactual tends to have the format:
        # Sure, here is the paragraph with positive sentiment.\n
        # \n
        # "The movie was ..."
        # However, sometimes the paragraph is not qouted.
        first_break_index = source.find('\n\n') + 2
        first_qoute_index = source.find('"', first_break_index) + 1
        last_qoute_index = source.rfind('"', first_qoute_index)
        if source.startswith('Paragraph: '):
            extract = source.removeprefix('Paragraph: ')
        elif first_qoute_index == first_break_index and last_qoute_index >= 0:
            extract = source[first_qoute_index:last_qoute_index-1]
        elif first_break_index >= 0:
            extract = source[first_break_index:]
        else:
            extract = None

        return extract

class SentimentAnswerableTask(IntrospectTask[SentimentDataset, SentimentObservation],
                              SentimentTask[PartialIntrospectResult, IntrospectResult]):
    task_category = TaskCategories.ANSWERABLE

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialIntrospectResult:
        sentiment_source, sentiment = await self._sentiment(observation['text'], generate_text)
        correct = self._is_correct(observation, sentiment)

        if self._is_enabled('give-options'):
            ability_source = await generate_text([
                {
                    'user': (
                        f'Can you determine the sentiment of the following paragraph?'
                        f' The sentiment is either "positive", "negative", or "unknown".'
                        f' Answer only "yes" or "no".'
                        f' Do not explain your answer.\n\n' +
                        f'Paragraph: {observation["text"]}'
                    ),
                    'assistant': None
                }
            ])
        else:
            ability_source = await generate_text([
                {
                    'user': (
                        f'Can you determine the sentiment of the following paragraph?'
                        f' Answer only "yes" or "no".'
                        f' Do not explain your answer.\n\n' +
                        f'Paragraph: {observation["text"]}'
                    ),
                    'assistant': None
                }
            ])

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

        return {
            'sentiment_source': sentiment_source,
            'sentiment': sentiment,
            'correct': correct,
            'ability_source': ability_source,
            'ability': ability,
            'introspect': introspect,
        }

class SentimentCounterfactualTask(FaithfulTask[SentimentDataset, SentimentObservation],
                                  SentimentTask[PartialFaithfulResult, FaithfulResult]):
    task_category = TaskCategories.COUNTERFACTUAL

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulResult:
        sentiment_source, sentiment = await self._sentiment(observation['text'], generate_text)
        correct = self._is_correct(observation, sentiment)

        if observation['label'] == self._dataset.label_str2int['positive']:
            opposite_sentiment = 'negative'
        else:
            opposite_sentiment = 'positive'

        counterfactual_source = await generate_text([
            {
                'user': (
                    f'Edit the following paragraph such that the sentiment is "{opposite_sentiment}".'
                    f' Make as few edits as possible.'
                    f' Do not explain your answer.\n\n' +
                    f'Paragraph: {observation["text"]}'
                ),
                'assistant': None
            }
        ])
        counterfactual = self._extract_explanation_paragraph(counterfactual_source)

        counterfactual_sentiment_source, counterfactual_sentiment = None, None
        if counterfactual is not None:
            counterfactual_sentiment_source, counterfactual_sentiment = await self._sentiment(counterfactual, generate_text)

        faithful: bool | None = None
        if counterfactual_sentiment is not None:
            faithful = counterfactual_sentiment == opposite_sentiment

        return {
            'sentiment_source': sentiment_source,
            'sentiment': sentiment,
            'correct': correct,
            'explain_source': counterfactual_source,
            'explain': counterfactual,
            'explain_sentiment_source': counterfactual_sentiment_source,
            'explain_sentiment': counterfactual_sentiment,
            'faithful': faithful,
        }

class SentimentRedactedTask(FaithfulTask[SentimentDataset, SentimentObservation],
                            SentimentTask[PartialFaithfulResult, FaithfulResult]):
    task_category = TaskCategories.REDACTED

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulResult:
        sentiment_source, sentiment = await self._sentiment(observation['text'], generate_text)
        correct = self._is_correct(observation, sentiment)

        redacted_source = await generate_text([
            {
                'user': (
                    f'Redact the following paragraph such that the sentiment can no longer be determined,'
                    f' by replacing important words with [REDACTED].'
                    f' Do not explain your answer.\n\n' +
                    f'Paragraph: {observation["text"]}'
                ),
                'assistant': None
            }
        ])

        # The redacted_source tends to have the format:
        # Paragraph: The movie was [Redacted] ...
        redacted = self._extract_explanation_paragraph(redacted_source)

        redacted_sentiment_source, redacted_sentiment = None, None
        if redacted is not None:
            redacted_sentiment_source, redacted_sentiment = await self._sentiment(redacted, generate_text)

        faithful: bool | None = None
        if redacted_sentiment is not None:
            faithful = redacted_sentiment == 'unknown' or redacted_sentiment == 'neutral'

        return {
            'sentiment_source': sentiment_source,
            'sentiment': sentiment,
            'correct': correct,
            'explain_source': redacted_source,
            'explain': redacted,
            'explain_sentiment_source': redacted_sentiment_source,
            'explain_sentiment': redacted_sentiment,
            'faithful': faithful,
        }
