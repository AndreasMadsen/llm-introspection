
from typing import TypeAlias, Literal
import re
import json

from ._abstract_tasks import \
    AbstractTask, \
    ClassifyTask, IntrospectTask, FaithfulTask, \
    TaskResultType, PartialTaskResultType
from ._request_capture import RequestCapture

from ..dataset import MultiChoiceDataset
from ..types import \
    TaskCategories, DatasetCategories, \
    MultiChoiceObservation, \
    PartialClassifyResult, ClassifyResult, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult
from ._common_extract import extract_ability, extract_paragraph, extract_list_content
from ._common_process import process_redact_words
from ._common_match import match_contains

PartialClassifyMultiChoiceResult: TypeAlias = PartialClassifyResult[str]
ClassifyMultiChoiceResult: TypeAlias = ClassifyResult[str, str]
PartialIntrospectMultiChoiceResult: TypeAlias = PartialIntrospectResult[str]
IntrospectMultiChoiceResult: TypeAlias = IntrospectResult[str, str]
PartialFaithfulMultiChoiceResult: TypeAlias = PartialFaithfulResult[str]
FaithfulMultiChoiceResult: TypeAlias = FaithfulResult[str, str]

class MultiChoiceTask(AbstractTask[MultiChoiceDataset, MultiChoiceObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.MULTI_CHOICE

    def _make_answer_choices(self, choices: list[str]):
        choices_prefixed = [f'{chr(ord("a") + choice_i)}) "{choice}"' for choice_i, choice in enumerate(choices)]
        options_string = ', '.join(choices_prefixed[:-1])
        options_string += f', or {choices_prefixed[-1]}'
        return options_string

    def _make_alternative_choice(self, choices: list[str], choice: str|None) -> None|str:
        if choice is None:
            return None
        if choice not in choices:
            return None

        choice_idx = choices.index(choice)
        alternative_choice_idx = (choice_idx + 1) % len(choices)
        return choices[alternative_choice_idx]

    async def _query_choice(
        self, question: str, choices: list[str], paragraph: str, generate_text: RequestCapture
    ) -> tuple[str, str]:
        user_prompt = ''
        if self._is_enabled('c-persona-you'):
            user_prompt += f'Consideing the following paragraph, how would you answer the question: "{question}"'
        elif self._is_enabled('c-persona-human'):
            user_prompt += f'Consideing the following paragraph, how would a human answer the question: "{question}"'
        else:
            user_prompt += f'Consider the following paragraph, and answer the question: "{question}"'

        if not self._is_enabled('c-no-redacted'):
            mask_work = self._ifelse_enabled("m-removed", "removed", "redacted")
            user_prompt += f' The paragraph can contain {mask_work} words marked with {self._mask_special_token}.'

        user_prompt += (
            f' Answer either {self._make_answer_choices(choices + ["unknown"])} if the question can not be answered.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        return (user_prompt, await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ]))

    def _extract_choice(self, choices: list[str], choice_source: str) -> str|None:
        # check for a matching letter
        # Example: b) The answer is b) washroom.
        if m := re.search(r'(?:^| |\(|")([a-z])(?:\.|,|:|\)|"|$)', choice_source, flags=re.IGNORECASE | re.MULTILINE):
            answer_letter, = m.groups()
            answer_index = ord(answer_letter.lower()) - ord('a')
            if 0 <= answer_index < len(choices):
                return choices[answer_index]
            if answer_index == len(choices):
                return 'unknown'

        # fallback to content matching
        choice_source = choice_source.lower()
        matched_choices = []
        for possible_choice in choices:
            idx = choice_source.find(possible_choice.lower())
            if idx >= 0:
                matched_choices.append((idx, possible_choice))
        matched_choices = list(map(lambda item: item[1], sorted(matched_choices)))

        if len(matched_choices) > 0:
            return matched_choices[0]

        if match_contains((
            'unknown',
            'cannot provide',
            'cannot determine',
            'insufficient context'
        ))(choice_source):
            return 'unknown'

        # No match found
        return None

    def _process_is_correct(self, observation: MultiChoiceObservation, choice: str|None) -> bool|None:
        if choice is None:
            return None
        return choice == observation['label']

    def _process_is_introspect(self, ability: Literal['yes', 'no']|None, choice: str|None) -> bool|None:
        match ability:
            case 'yes':
                introspect = choice != 'unknown'
            case 'no':
                introspect = choice == 'unknown'
            case _:
                introspect = None

        return introspect

class MultiChoiceClassifyTask(ClassifyTask[MultiChoiceDataset, MultiChoiceObservation],
                              MultiChoiceTask[PartialClassifyMultiChoiceResult, ClassifyMultiChoiceResult]):
    task_category = TaskCategories.CLASSIFY

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialClassifyMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_prompt, choice_answer = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_answer)
        correct = self._process_is_correct(observation, choice)

        return {
            'debug': f'Question: {question}.\nOptions: {choices},\nParagraph: {paragraph}',
            'predict_prompt': choice_prompt,
            'predict_answer': choice_answer,
            'predict': choice,
            'correct': correct
        }

class MultiChoiceAnswerableTask(IntrospectTask[MultiChoiceDataset, MultiChoiceObservation],
                                MultiChoiceTask[PartialIntrospectMultiChoiceResult, IntrospectMultiChoiceResult]):
    task_category = TaskCategories.ANSWERABLE

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialIntrospectMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_prompt, choice_answer = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_answer)
        correct = self._process_is_correct(observation, choice)

        ability_prompt = ''
        if self._is_enabled('i-persona-you'):
            ability_prompt += f'Are you able to answer the question "{question}" based on the following paragraph?'
        elif self._is_enabled('i-persona-human'):
            ability_prompt += f'Is a human able to answer the question "{question}" based on the following paragraph?'
        else:
            ability_prompt += f'Is it possible to answer the question "{question}" based on the following paragraph?'

        if self._is_enabled('i-options'):
            ability_prompt += f' The possible answers to the question are {self._make_answer_choices(choices)}.'

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
        introspect = self._process_is_introspect(ability, choice)

        return {
            'debug': f'Question: {question}.\nOptions: {choices},\nParagraph: {paragraph}',
            'predict_prompt': choice_prompt,
            'predict_answer': choice_answer,
            'predict': choice,
            'correct': correct,
            'ability_prompt': ability_prompt,
            'ability_answer': ability_answer,
            'ability': ability,
            'introspect': introspect,
        }

class MultiChoiceCounterfactualTask(FaithfulTask[MultiChoiceDataset, MultiChoiceObservation],
                                    MultiChoiceTask[PartialFaithfulMultiChoiceResult, FaithfulMultiChoiceResult]):
    task_category = TaskCategories.COUNTERFACTUAL

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialFaithfulMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_prompt, choice_answer = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_answer)
        correct = self._process_is_correct(observation, choice)

        alternative_choice = self._make_alternative_choice(choices, observation['label'])
        counterfactual_prompt = ''
        if self._is_enabled('e-persona-you'):
            counterfactual_prompt += f'Edit the following paragraph such you would answer the question "{question}" with "{alternative_choice}".'
        elif self._is_enabled('e-persona-human'):
            counterfactual_prompt += f'Edit the following paragraph such a human would answer the question "{question}" with "{alternative_choice}".'
        else:
            counterfactual_prompt += f'Edit the following paragraph such that the answer to the question "{question}" is "{alternative_choice}".'

        counterfactual_prompt += (
            f' Make as few edits as possible.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )
        counterfactual_answer, counterfactual = None, None
        if alternative_choice is not None:
            counterfactual_answer = await generate_text([
                {
                    'user': counterfactual_prompt,
                    'assistant': None
                }
            ])
            counterfactual = extract_paragraph(counterfactual_answer)

        counterfactual_choice_prompt, counterfactual_choice_answer, counterfactual_choice = None, None, None
        if counterfactual is not None:
            counterfactual_choice_prompt, counterfactual_choice_answer = await self._query_choice(question, choices, counterfactual, generate_text)
            counterfactual_choice = self._extract_choice(choices, counterfactual_choice_answer)

        faithful: bool | None = None
        if counterfactual_choice is not None:
            faithful = counterfactual_choice == alternative_choice

        return {
            'debug': f'Question: {question}.\nOptions: {choices}\nAlternative: {alternative_choice}\nParagraph: {paragraph}',
            'predict_prompt': choice_prompt,
            'predict_answer': choice_answer,
            'predict': choice,
            'correct': correct,
            'explain_prompt': counterfactual_prompt,
            'explain_answer': counterfactual_answer,
            'explain': counterfactual,
            'explain_predict_prompt': counterfactual_choice_prompt,
            'explain_predict_answer': counterfactual_choice_answer,
            'explain_predict': counterfactual_choice,
            'faithful': faithful,
        }

class MultiChoiceRedactedTask(FaithfulTask[MultiChoiceDataset, MultiChoiceObservation],
                              MultiChoiceTask[PartialFaithfulMultiChoiceResult, FaithfulMultiChoiceResult]):
    task_category = TaskCategories.REDACTED

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialFaithfulMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_prompt, choice_answer = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_answer)
        correct = self._process_is_correct(observation, choice)

        redacted_prompt = ''
        if self._is_enabled('e-short'):
            if self._is_enabled('e-persona-you'):
                redacted_prompt += f'Redact the following paragraph such that you can no longer answer the question "{question}",'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += f'Redact the following paragraph such that a human can no longer answer the question "{question}",'
            else:
                redacted_prompt += f'Redact the following paragraph such that the question "{question}" can no longer be answered,'

            redacted_prompt += f' by replacing important words with {self._mask_special_token}.'
        else:
            redacted_prompt += (
                f'Redact the most important words for answering "{question}" given the following paragraph,'
                f' by replacing important words with {self._mask_special_token},'
            )
            if self._is_enabled('e-persona-you'):
                redacted_prompt += ' such that without these words you can not answer the question.'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += ' such that without these words a human can not answer the question.'
            else:
                redacted_prompt += ' such that without these words the question can not be answered.'

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

        redacted_choice_prompt, redacted_choice_answer, redacted_choice = None, None, None
        if redacted is not None:
            redacted_choice_prompt, redacted_choice_answer = await self._query_choice(question, choices, redacted, generate_text)
            redacted_choice = self._extract_choice(choices, redacted_choice_answer)

        faithful: bool | None = None
        if redacted_choice is not None:
            faithful = redacted_choice == 'unknown'

        return {
            'debug': f'Question: {question}.\nParagraph: {paragraph}',
            'predict_prompt': choice_prompt,
            'predict_answer': choice_answer,
            'predict': choice,
            'correct': correct,
            'explain_prompt': redacted_prompt,
            'explain_answer': redacted_answer,
            'explain': redacted,
            'explain_predict_prompt': redacted_choice_prompt,
            'explain_predict_answer': redacted_choice_answer,
            'explain_predict': redacted_choice,
            'faithful': faithful,
        }

class MultiChoiceImportanceTask(FaithfulTask[MultiChoiceDataset, MultiChoiceObservation],
                                MultiChoiceTask[PartialFaithfulMultiChoiceResult, FaithfulMultiChoiceResult]):
    task_category = TaskCategories.IMPORTANCE

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialFaithfulMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_prompt, choice_answer = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_answer)
        correct = self._process_is_correct(observation, choice)

        importance_prompt = ''
        importance_prompt += f'List the most important words for answering "{question}" given the following paragraph,'
        if self._is_enabled('e-persona-you'):
            importance_prompt += ' such that without these words you can not answer the question.'
        elif self._is_enabled('e-persona-human'):
            importance_prompt += ' such that without these words a human can not answer the question.'
        else:
            importance_prompt += ' such that without these words the question can not be answered.'
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

        redacted_choice_prompt, redacted_choice_answer, redacted_choice = None, None, None
        if redacted is not None:
            redacted_choice_prompt, redacted_choice_answer = await self._query_choice(question, choices, redacted, generate_text)
            redacted_choice = self._extract_choice(choices, redacted_choice_answer)

        faithful: bool | None = None
        if redacted_choice is not None:
            faithful = redacted_choice == 'unknown'

        # generate explanation
        explain = None
        if important_words is not None and redacted is not None:
            explain = json.dumps(important_words) + '\n\n' + redacted

        return {
            'debug': f'Question: {question}.\nParagraph: {paragraph}',
            'predict_prompt': choice_prompt,
            'predict_answer': choice_answer,
            'predict': choice,
            'correct': correct,
            'explain_prompt': importance_prompt,
            'explain_answer': importance_answer,
            'explain': explain,
            'explain_predict_prompt': redacted_choice_prompt,
            'explain_predict_answer': redacted_choice_answer,
            'explain_predict': redacted_choice,
            'faithful': faithful,
        }
