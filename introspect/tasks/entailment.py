
import re
import json
from typing import Literal, TypeAlias

from ..dataset import EntailmentDataset
from ..types import \
    TaskCategories, DatasetCategories, \
    EntailmentObservation, \
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
from ._common_match import match_contains

EntailmentPredict: TypeAlias = Literal['yes', 'no', 'unknown']
EntailmentLabel: TypeAlias = Literal['yes', 'no']
PartialClassifyEntailmentResult: TypeAlias = PartialClassifyResult[EntailmentPredict]
ClassifyEntailmentResult: TypeAlias = ClassifyResult[EntailmentLabel, EntailmentPredict]
PartialIntrospectEntailmentResult: TypeAlias = PartialIntrospectResult[EntailmentPredict]
IntrospectEntailmentResult: TypeAlias = IntrospectResult[EntailmentLabel, EntailmentPredict]
PartialFaithfulEntailmentResult: TypeAlias = PartialFaithfulResult[EntailmentPredict]
FaithfulEntailmentResult: TypeAlias = FaithfulResult[EntailmentLabel, EntailmentPredict]

class EntailmentTask(AbstractTask[EntailmentDataset, EntailmentObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.ENTAILMENT

    def _make_counterfactual_entailment(self, entailment: EntailmentLabel) -> EntailmentLabel:
        match entailment:
            case 'yes':
                return 'no'
            case 'no':
                return 'yes'

    async def _query_entailment(
        self, hypothesis: str, paragraph: str, generate_text: RequestCapture
    ) -> tuple[str,str]:
        user_prompt = ''
        if self._is_enabled('c-persona-you'):
            user_prompt += f'Do you think the statement "{hypothesis}" entail from the following paragraph?'
        elif self._is_enabled('c-persona-human'):
            user_prompt += f'Does a human think the statement "{hypothesis}" entail from the following paragraph?'
        else:
            user_prompt += f'Does the statement "{hypothesis}" entail from the following paragraph?'

        if not self._is_enabled('c-no-redacted'):
            mask_work = self._ifelse_enabled("m-removed", "removed", "redacted")
            user_prompt += f' The paragraph can contain {mask_work} words marked with {self._mask_special_token}.'

        user_prompt += (
            ' Answer either 1) "yes" for entailment, 2) "no" for no entailment, or 3) "unknown" if the question can not be answered.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        return (user_prompt, await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ]))

    def _process_is_correct(self, observation: EntailmentObservation, entailment: EntailmentPredict|None) -> bool|None:
        match entailment:
            case None:
                return None
            case ('yes' | 'no'):
                return observation['label'] == entailment
            case _:
                return False

    def _process_is_introspect(self, ability: Literal['yes', 'no']|None, entailment: EntailmentPredict|None) -> bool|None:
        match ability:
            case 'yes':
                introspect = entailment in ('yes', 'no')
            case 'no':
                introspect = entailment == 'unknown'
            case _:
                introspect = None

        return introspect

    def _extract_entailment(self, source: str) -> Literal['yes', 'no', 'unknown']|None:
        # check for a matching letter
        # Example: b) The answer is b) washroom.
        if m := re.search(r'(?:^| |\(|Answer: )([1-9])(?:\)|$)', source, flags=re.IGNORECASE | re.MULTILINE):
            answer_letter, = m.groups()
            answer_index = ord(answer_letter) - ord('1')
            if answer_index == 0:
                return 'yes'
            elif answer_index == 1:
                return 'no'
            elif answer_index == 2:
                return 'unknown'

        # fallback to content matching
        source = source.lower()

        if match_contains((
            'unknown',
            'cannot provide',
            'cannot determine',
            'insufficient context',
            'unable to determine',
            'impossible for me to determine'
        ))(source):
            return 'unknown'
        elif source.startswith('yes'):
            return 'yes'
        elif source.startswith('no'):
            return 'no'

        # No match found
        return None

class EntailmentClassifyTask(ClassifyTask[EntailmentDataset, EntailmentObservation],
                            EntailmentTask[PartialClassifyEntailmentResult, ClassifyEntailmentResult]):
    task_category = TaskCategories.CLASSIFY

    async def _task(self, observation: EntailmentObservation, generate_text: RequestCapture) -> PartialClassifyEntailmentResult:
        hypothesis = observation['hypothesis']
        paragraph = observation['paragraph']

        entailment_prompt, entailment_answer = await self._query_entailment(hypothesis, paragraph, generate_text)
        entailment = self._extract_entailment(entailment_answer)
        correct = self._process_is_correct(observation, entailment)

        return {
            'debug': f'Hypothesis: {hypothesis}\nParagraph: {paragraph}.',
            'predict_prompt': entailment_prompt,
            'predict_answer': entailment_answer,
            'predict': entailment,
            'correct': correct
        }

class EntailmentAnswerableTask(IntrospectTask[EntailmentDataset, EntailmentObservation],
                              EntailmentTask[PartialIntrospectEntailmentResult, IntrospectEntailmentResult]):
    task_category = TaskCategories.ANSWERABLE

    async def _task(self, observation: EntailmentObservation, generate_text: RequestCapture) -> PartialIntrospectEntailmentResult:
        hypothesis = observation['hypothesis']
        paragraph = observation['paragraph']

        entailment_prompt, entailment_answer = await self._query_entailment(hypothesis, paragraph, generate_text)
        entailment = self._extract_entailment(entailment_answer)
        correct = self._process_is_correct(observation, entailment)

        ability_prompt = ''
        if self._is_enabled('i-persona-you'):
            ability_prompt += f'Consider the statement "{hypothesis}", are you able to determine the if it entail from the following paragraph?'
        elif self._is_enabled('i-persona-human'):
            ability_prompt += f'Consider the statement "{hypothesis}", is a human able to determine the if it entail from the following paragraph?'
        else:
            ability_prompt += f'Consider the statement "{hypothesis}", is it possible to determine the if it entail from the following paragraph?'

        if self._is_enabled('i-options'):
            ability_prompt += ' There is either entailment, no entailment, or it\'s unknown.'

        ability_prompt += (
            f' Answer only "yes" or "no".'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        ability_answer = await generate_text([
            {
                'user': ability_prompt,
                'assistant': None
            }
        ])
        ability = extract_ability(ability_answer)
        introspect = self._process_is_introspect(ability, entailment)

        return {
            'debug': f'Hypothesis: {hypothesis}\nParagraph: {paragraph}.',
            'predict_prompt': entailment_prompt,
            'predict_answer': entailment_answer,
            'predict': entailment,
            'correct': correct,
            'ability_prompt': ability_prompt,
            'ability_answer': ability_answer,
            'ability': ability,
            'introspect': introspect,
        }

class EntailmentCounterfactualTask(FaithfulTask[EntailmentDataset, EntailmentObservation],
                                  EntailmentTask[PartialFaithfulEntailmentResult, FaithfulEntailmentResult]):
    task_category = TaskCategories.COUNTERFACTUAL

    async def _task(self, observation: EntailmentObservation, generate_text: RequestCapture) -> PartialFaithfulEntailmentResult:
        hypothesis = observation['hypothesis']
        paragraph = observation['paragraph']

        entailment_prompt, entailment_answer = await self._query_entailment(hypothesis, paragraph, generate_text)
        entailment = self._extract_entailment(entailment_answer)
        correct = self._process_is_correct(observation, entailment)

        opposite_entailment = self._make_counterfactual_entailment(observation['label'])
        counterfactual_prompt = ''
        if self._is_enabled('e-implcit-target'):
            if self._is_enabled('e-persona-you'):
                counterfactual_prompt += f'Edit the following paragraph, such that given the statement "{hypothesis}", you would say the entailment is the opposite of what it currently is.'
            elif self._is_enabled('e-persona-human'):
                counterfactual_prompt += f'Edit the following paragraph, such that given the statement "{hypothesis}", a human would say the entailment is the opposite of what it currently is.'
            else:
                counterfactual_prompt += f'Edit the following paragraph, such that given the statement "{hypothesis}", the entailment becomes the opposite of what it currently is.'
        else:
            entail_instruction = 'entails' if opposite_entailment == 'yes' else 'does not entails'
            if self._is_enabled('e-persona-you'):
                counterfactual_prompt += f'Edit the following paragraph such that you would say the statement "{hypothesis}" {entail_instruction} from it.'
            elif self._is_enabled('e-persona-human'):
                counterfactual_prompt += f'Edit the following paragraph such that a human would say the statement "{hypothesis}" {entail_instruction} from it.'
            else:
                counterfactual_prompt += f'Edit the following paragraph such that the statement "{hypothesis}" {entail_instruction} from it.'

        counterfactual_prompt += (
            f' Make as few edits as possible.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        counterfactual_answer, counterfactual = None, None
        if opposite_entailment is not None:
            counterfactual_answer = await generate_text([
                {
                    'user': counterfactual_prompt,
                    'assistant': None
                }
            ])
            counterfactual = extract_paragraph(counterfactual_answer)

        counterfactual_entailment_prompt, counterfactual_entailment_answer, counterfactual_entailment = None, None, None
        if counterfactual is not None:
            counterfactual_entailment_prompt, counterfactual_entailment_answer = await self._query_entailment(hypothesis, counterfactual, generate_text)
            counterfactual_entailment = self._extract_entailment(counterfactual_entailment_answer)

        faithful: bool | None = None
        if counterfactual_entailment is not None:
            faithful = counterfactual_entailment == opposite_entailment

        return {
            'debug': f'Hypothesis: {hypothesis}\nParagraph: {paragraph}.',
            'predict_prompt': entailment_prompt,
            'predict_answer': entailment_answer,
            'predict': entailment,
            'correct': correct,
            'explain_prompt': counterfactual_entailment_prompt,
            'explain_answer': counterfactual_entailment_answer,
            'explain': counterfactual,
            'explain_predict_prompt': counterfactual_entailment_prompt,
            'explain_predict_answer': counterfactual_entailment_answer,
            'explain_predict': counterfactual_entailment,
            'faithful': faithful,
        }

class EntailmentRedactedTask(FaithfulTask[EntailmentDataset, EntailmentObservation],
                            EntailmentTask[PartialFaithfulEntailmentResult, FaithfulEntailmentResult]):
    task_category = TaskCategories.REDACTED

    async def _task(self, observation: EntailmentObservation, generate_text: RequestCapture) -> PartialFaithfulEntailmentResult:
        hypothesis = observation['hypothesis']
        paragraph = observation['paragraph']

        entailment_prompt, entailment_answer = await self._query_entailment(hypothesis, paragraph, generate_text)
        entailment = self._extract_entailment(entailment_answer)
        correct = self._process_is_correct(observation, entailment)

        redacted_prompt = ''
        if self._is_enabled('e-short'):
            if self._is_enabled('e-persona-you'):
                redacted_prompt += f'Redact the following paragraph such you can not determine if the statement "{hypothesis}" entails from it,'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += f'Redact the following paragraph such a human can not determine if the statement "{hypothesis}" entails from it,'
            else:
                redacted_prompt += f'Redact the following paragraph such it can not be determined if the statement "{hypothesis}" entails from it,'

            redacted_prompt += f' by replacing important words with {self._mask_special_token}.'
        else:
            redacted_prompt += (
                f'Redact the most important words in the following paragraph, for determining if the statement "{hypothesis}" entails from it,'
                f' by replacing important words with {self._mask_special_token},'
            )

            if self._is_enabled('e-persona-you'):
                redacted_prompt += ' such that without these words you can not determine if there is entailment or no entailment.'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += ' such that without these words a human can not determine if there is entailment or no entailment.'
            else:
                redacted_prompt += ' such that without these words it can not be determined if there is entailment or no entailment.'

        redacted_prompt += (
            f' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        # The redacted_source tends to have the format:
        # Paragraph: The movie was [Redacted] ...
        redacted_answer = await generate_text([
            {
                'user': redacted_prompt,
                'assistant': None
            }
        ])
        redacted = extract_paragraph(redacted_answer)

        redacted_entailment_prompt, redacted_entailment_answer, redacted_entailment = None, None, None
        if redacted is not None:
            redacted_entailment_prompt, redacted_entailment_answer = await self._query_entailment(hypothesis, redacted, generate_text)
            redacted_entailment = self._extract_entailment(redacted_entailment_answer)

        faithful: bool | None = None
        if redacted_entailment is not None:
            faithful = redacted_entailment == 'unknown' or redacted_entailment == 'neutral'

        return {
            'debug': f'Statement: {hypothesis}\nParagraph: {paragraph}.',
            'predict_prompt': entailment_prompt,
            'predict_answer': entailment_answer,
            'predict': entailment,
            'correct': correct,
            'explain_prompt': redacted_prompt,
            'explain_answer': redacted_answer,
            'explain': redacted,
            'explain_predict_prompt': redacted_entailment_prompt,
            'explain_predict_answer': redacted_entailment_answer,
            'explain_predict': redacted_entailment,
            'faithful': faithful,
        }

class EntailmentImportanceTask(FaithfulTask[EntailmentDataset, EntailmentObservation],
                              EntailmentTask[PartialFaithfulEntailmentResult, FaithfulEntailmentResult]):
    task_category = TaskCategories.IMPORTANCE

    async def _task(self, observation: EntailmentObservation, generate_text: RequestCapture) -> PartialFaithfulEntailmentResult:
        hypothesis = observation['hypothesis']
        paragraph = observation['paragraph']

        entailment_prompt, entailment_answer = await self._query_entailment(hypothesis, paragraph, generate_text)
        entailment = self._extract_entailment(entailment_answer)
        correct = self._process_is_correct(observation, entailment)

        importance_prompt = ''
        importance_prompt += f'List the most important words in the following paragraph, for determining if the statement "{hypothesis}" entails from it,'
        if self._is_enabled('e-persona-you'):
            importance_prompt += ' such that without these words you can not determine if there is entailment or no entailment.'
        elif self._is_enabled('e-persona-human'):
            importance_prompt += ' such that without these words you a human not determine if there is entailment or no entailment.'
        else:
            importance_prompt += ' such that without these words it can not be determined if there is entailment or no entailment.'
        importance_prompt += (
            f' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        importance_answer = await generate_text([
            {
                'user': importance_prompt,
                'assistant': None
            }
        ])
        important_words = extract_list_content(importance_answer)

        redacted = None
        if important_words is not None:
            redacted = process_redact_words(observation['paragraph'], important_words, self._mask_special_token)

        redacted_entailment_prompt, redacted_entailment_answer, redacted_entailment = None, None, None
        if redacted is not None:
            redacted_entailment_prompt, redacted_entailment_answer = await self._query_entailment(statement, redacted, generate_text)
            redacted_entailment = self._extract_entailment(redacted_entailment_answer)

        faithful: bool | None = None
        if redacted_entailment is not None:
            faithful = redacted_entailment == 'unknown' or redacted_entailment == 'neutral'

        # generate explanation
        explain = None
        if important_words is not None and redacted is not None:
            explain = json.dumps(important_words) + '\n\n' + redacted

        return {
            'debug': f'Hypothesis: {hypothesis}\nParagraph: {paragraph}.',
            'predict_prompt': entailment_prompt,
            'predict_answer': entailment_answer,
            'predict': entailment,
            'correct': correct,
            'explain_prompt': importance_prompt,
            'explain_answer': importance_answer,
            'explain': explain,
            'explain_predict_prompt': redacted_entailment_prompt,
            'explain_predict_answer': redacted_entailment_answer,
            'explain_predict': redacted_entailment,
            'faithful': faithful,
        }
