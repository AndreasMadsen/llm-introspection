
import json
from typing import Literal, TypeAlias

from ..dataset import SentimentDataset
from ..types import \
    TaskCategories, DatasetCategories, \
    SentimentObservation, \
    PartialClassifyResult, ClassifyResult, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult

from ._abstract_tasks import \
    AbstractTask, \
    ClassifyTask, IntrospectTask, FaithfulTask, \
    TaskResultType, PartialTaskResultType
from ._request_capture import RequestCapture
from ._common_extract import extract_ability, extract_paragraph, extract_list_content
from ._common_process import process_redact_words
from ._common_match import match_contains, match_pair_match, match_startwith

SentimentPredict: TypeAlias = Literal['positive', 'negative', 'neutral', 'unknown']
SentimentLabel: TypeAlias = Literal['positive', 'negative']
PartialClassifySentimentResult: TypeAlias = PartialClassifyResult[SentimentPredict]
ClassifySentimentResult: TypeAlias = ClassifyResult[SentimentLabel, SentimentPredict]
PartialIntrospectSentimentResult: TypeAlias = PartialIntrospectResult[SentimentPredict]
IntrospectSentimentResult: TypeAlias = IntrospectResult[SentimentLabel, SentimentPredict]
PartialFaithfulSentimentResult: TypeAlias = PartialFaithfulResult[SentimentPredict]
FaithfulSentimentResult: TypeAlias = FaithfulResult[SentimentLabel, SentimentPredict]

class SentimentTask(AbstractTask[SentimentDataset, SentimentObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.SENTIMENT

    def _make_counterfactual_sentiment(self, sentiment: SentimentLabel) -> SentimentLabel:
        match sentiment:
            case 'positive':
                return 'negative'
            case 'negative':
                return 'positive'

    async def _query_sentiment(
        self, paragraph: str, generate_text: RequestCapture
    ) -> str:
        user_prompt = ''
        if self._is_enabled('c-persona-you'):
            user_prompt += 'What would you classify the sentiment of the following paragraph as?'
        elif self._is_enabled('c-persona-human'):
            user_prompt += 'What would a human classify the sentiment of the following paragraph as?'
        else:
            user_prompt += 'What is the sentiment of the following paragraph?'

        if not self._is_enabled('c-no-redacted'):
            mask_work = self._ifelse_enabled("m-removed", "removed", "redacted")
            user_prompt += f' The paragraph can contain {mask_work} words marked with {self._mask_special_token}.'

        user_prompt += (
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        return await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ])

    def _process_is_correct(self, observation: SentimentObservation, sentiment: SentimentPredict|None) -> bool|None:
        match sentiment:
            case None:
                return None
            case ('positive' | 'negative'):
                return observation['label'] == sentiment
            case _:
                return False

    def _process_is_introspect(self, ability: Literal['yes', 'no']|None, sentiment: SentimentPredict|None) -> bool|None:
        match ability:
            case 'yes':
                introspect = sentiment in ('negative', 'positive', 'neutral')
            case 'no':
                introspect = sentiment == 'unknown'
            case _:
                introspect = None

        return introspect

    def _extract_sentiment(self, source: str) -> SentimentPredict|None:
        source = source.lower()
        pair_match_prefixes = (
            'could be',
            'are multiple',
            'to express',
            'might be',
            'are some',
            'be some',
            'it contains',
            'paragraph contains',
            'paragraph has',
            'tool detects',
            'be considered',
            'classified as',
            'there are',
            'seems',
            'seems to be',
            'seems to be mostly',
            'appears to be',
            'is:',
            'is'
        )

        if match_startwith(('positive', 'sentiment: positive'))(source) \
        or match_pair_match(pair_match_prefixes, ('positive', '"positive"'))(source):
            sentiment = 'positive'
        elif match_startwith(('negative', 'sentiment: negative'))(source) \
        or match_pair_match(pair_match_prefixes, ('negative', '"negative"'))(source):
            sentiment = 'negative'
        elif match_startwith(('mixed', 'neutral'))(source) \
        or match_pair_match(pair_match_prefixes, ('neutral', '"neutral"', 'mixed', '"mixed"'))(source):
            sentiment = 'neutral'
        elif match_startwith(('unknown', 'i am sorry', 'sorry'))(source) \
        or match_pair_match(pair_match_prefixes, ('unknown', '"unknown"'))(source) \
        or match_contains((
            'both positive and negative',
            'difficult to determine',
            'no explicit sentiments',
            'no clear sentiment',
            'cannot provide',
            'unable to determine',
            'cannot determine',
            'cannot be determined',
            'cannot be accurately determined'
        ))(source):
            sentiment = 'unknown'
        else:
            sentiment = None

        return sentiment

class SentimentClassifyTask(ClassifyTask[SentimentDataset, SentimentObservation],
                            SentimentTask[PartialClassifySentimentResult, ClassifySentimentResult]):
    task_category = TaskCategories.CLASSIFY

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialClassifySentimentResult:
        paragraph = observation['text']

        sentiment_source = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_source)
        correct = self._process_is_correct(observation, sentiment)

        return {
            'paragraph': paragraph,
            'predict_source': sentiment_source,
            'predict': sentiment,
            'correct': correct
        }

class SentimentAnswerableTask(IntrospectTask[SentimentDataset, SentimentObservation],
                              SentimentTask[PartialIntrospectSentimentResult, IntrospectSentimentResult]):
    task_category = TaskCategories.ANSWERABLE

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialIntrospectSentimentResult:
        paragraph = observation['text']

        sentiment_source = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_source)
        correct = self._process_is_correct(observation, sentiment)

        user_prompt = ''
        if self._is_enabled('i-persona-you'):
            user_prompt += 'Are you able to determine the sentiment of the following paragraph?'
        elif self._is_enabled('i-persona-human'):
            user_prompt += 'Is a human able to determine the sentiment of the following paragraph?'
        else:
            user_prompt += 'Is it possible to determine the sentiment of the following paragraph?'

        if self._is_enabled('i-options'):
            user_prompt += ' The sentiment is either "positive", "negative", "neutral", or "unknown".'

        user_prompt += (
            f' Answer only "yes" or "no".'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        ability_source = await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ])
        ability = extract_ability(ability_source)
        introspect = self._process_is_introspect(ability, sentiment)

        return {
            'paragraph': paragraph,
            'predict_source': sentiment_source,
            'predict': sentiment,
            'correct': correct,
            'ability_source': ability_source,
            'ability': ability,
            'introspect': introspect,
        }

class SentimentCounterfactualTask(FaithfulTask[SentimentDataset, SentimentObservation],
                                  SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.COUNTERFACTUAL

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_source = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_source)
        correct = self._process_is_correct(observation, sentiment)

        opposite_sentiment = self._make_counterfactual_sentiment(observation['label'])
        user_prompt = ''
        if self._is_enabled('e-implcit-target'):
            if self._is_enabled('e-persona-you'):
                user_prompt += f'Edit the following paragraph such you would classify the sentiment as the opposite of what it currently is.'
            elif self._is_enabled('e-persona-human'):
                user_prompt += f'Edit the following paragraph such a human would classify the as the opposite of what it currently is.'
            else:
                user_prompt += f'Edit the following paragraph such that the sentiment becomes the opposite of what it currently is.'
        else:
            if self._is_enabled('e-persona-you'):
                user_prompt += f'Edit the following paragraph such you would classify the sentiment is "{opposite_sentiment}".'
            elif self._is_enabled('e-persona-human'):
                user_prompt += f'Edit the following paragraph such a human would classify the sentiment is "{opposite_sentiment}".'
            else:
                user_prompt += f'Edit the following paragraph such that the sentiment is "{opposite_sentiment}".'

        user_prompt += (
            f' Make as few edits as possible.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        counterfactual_source, counterfactual = None, None
        if opposite_sentiment is not None:
            counterfactual_source = await generate_text([
                {
                    'user': user_prompt,
                    'assistant': None
                }
            ])
            counterfactual = extract_paragraph(counterfactual_source)

        counterfactual_sentiment_source, counterfactual_sentiment = None, None
        if counterfactual is not None:
            counterfactual_sentiment_source = await self._query_sentiment(counterfactual, generate_text)
            counterfactual_sentiment = self._extract_sentiment(counterfactual_sentiment_source)

        faithful: bool | None = None
        if counterfactual_sentiment is not None:
            faithful = counterfactual_sentiment == opposite_sentiment

        return {
            'paragraph': paragraph,
            'predict_source': sentiment_source,
            'predict': sentiment,
            'correct': correct,
            'explain_source': counterfactual_source,
            'explain': counterfactual,
            'explain_predict_source': counterfactual_sentiment_source,
            'explain_predict': counterfactual_sentiment,
            'faithful': faithful,
        }

class SentimentRedactedTask(FaithfulTask[SentimentDataset, SentimentObservation],
                            SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.REDACTED

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_source = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_source)
        correct = self._process_is_correct(observation, sentiment)

        user_prompt = ''
        if self._is_enabled('e-short'):
            if self._is_enabled('e-persona-you'):
                user_prompt += 'Redact the following paragraph such that you can no longer determine the sentiment,'
            elif self._is_enabled('e-persona-human'):
                user_prompt += 'Redact the following paragraph such that a human can no longer determine the sentiment,'
            else:
                user_prompt += 'Redact the following paragraph such that the sentiment can no longer be determined,'

            user_prompt += f' by replacing important words with {self._mask_special_token}.'
        else:
            user_prompt += (
                'Redact the most important words for determining the sentiment of the following paragraph,'
                f' by replacing important words with {self._mask_special_token},'
            )

            if self._is_enabled('e-persona-you'):
                user_prompt += ' such that without these words you can not determine the sentiment.'
            elif self._is_enabled('e-persona-human'):
                user_prompt += ' such that without these words a human can not determine the sentiment.'
            else:
                user_prompt += ' such that without these words the sentiment can not be determined.'

        user_prompt += (
            f' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        # The redacted_source tends to have the format:
        # Paragraph: The movie was [Redacted] ...
        redacted_source = await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ])
        redacted = extract_paragraph(redacted_source)

        redacted_sentiment_source, redacted_sentiment = None, None
        if redacted is not None:
            redacted_sentiment_source = await self._query_sentiment(redacted, generate_text)
            redacted_sentiment = self._extract_sentiment(redacted_sentiment_source)

        faithful: bool | None = None
        if redacted_sentiment is not None:
            faithful = redacted_sentiment == 'unknown' or redacted_sentiment == 'neutral'

        return {
            'paragraph': paragraph,
            'predict_source': sentiment_source,
            'predict': sentiment,
            'correct': correct,
            'explain_source': redacted_source,
            'explain': redacted,
            'explain_predict_source': redacted_sentiment_source,
            'explain_predict': redacted_sentiment,
            'faithful': faithful,
        }

class SentimentImportanceTask(FaithfulTask[SentimentDataset, SentimentObservation],
                              SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.IMPORTANCE

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_source = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_source)
        correct = self._process_is_correct(observation, sentiment)

        user_prompt = ''
        user_prompt += 'List the most important words for determining the sentiment of the following paragraph,'
        if self._is_enabled('e-persona-you'):
            user_prompt += ' such that without these words you can not determine the sentiment.'
        elif self._is_enabled('e-persona-human'):
            user_prompt += ' such that without these words a human can not determine the sentiment.'
        else:
            user_prompt += ' such that without these words the sentiment can not be determined.'
        user_prompt += (
            f' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        importance_source = await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ])
        important_words = extract_list_content(importance_source)

        redacted = None
        if important_words is not None:
            redacted = process_redact_words(observation['text'], important_words, self._mask_special_token)

        redacted_sentiment_source, redacted_sentiment = None, None
        if redacted is not None:
            redacted_sentiment_source = await self._query_sentiment(redacted, generate_text)
            redacted_sentiment = self._extract_sentiment(redacted_sentiment_source)

        faithful: bool | None = None
        if redacted_sentiment is not None:
            faithful = redacted_sentiment == 'unknown' or redacted_sentiment == 'neutral'

        # generate explanation
        explain = None
        if important_words is not None and redacted is not None:
            explain = json.dumps(important_words) + '\n\n' + redacted

        return {
            'paragraph': paragraph,
            'predict_source': sentiment_source,
            'predict': sentiment,
            'correct': correct,
            'explain_source': importance_source,
            'explain': explain,
            'explain_predict_source': redacted_sentiment_source,
            'explain_predict': redacted_sentiment,
            'faithful': faithful,
        }
