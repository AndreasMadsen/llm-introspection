
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
    ) -> str:
        user_prompt = ''
        if self._is_enabled('c-persona-you'):
            pass
        elif self._is_enabled('c-persona-human'):
            pass
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

        return await generate_text([
            {
                'user': user_prompt,
                'assistant': None
            }
        ])

    def _extract_choice(self, choices: list[str], choice_source: str) -> str|None:
        # check for a matching letter
        # Example: b) The answer is b) washroom.
        if m := re.search(r'(?:^| |\()([a-z])\)', choice_source, flags=re.IGNORECASE | re.MULTILINE):
            answer_letter, = m.groups()
            answer_index = ord(answer_letter) - ord('a')
            if 0 <= answer_index < len(choices):
                return choices[answer_index]
            if answer_index == len(choices):
                return 'unknown'

        # fallback to content matching
        choice_source = choice_source.lower()

        if 'unknown' in choice_source:
            return 'unknown'

        for possible_choice in choices:
            if possible_choice.lower() in choice_source:
                return possible_choice

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

    def _process_redact_words(self, observation: MultiChoiceObservation, important_words: list[str]) -> str:
        return re.sub(
            r'\b(?:' + '|'.join(re.escape(word) for word in important_words) + r')\b',
            self._mask_special_token,
            observation['paragraph'],
            flags=re.IGNORECASE
        )

class MultiChoiceClassifyTask(ClassifyTask[MultiChoiceDataset, MultiChoiceObservation],
                              MultiChoiceTask[PartialClassifyMultiChoiceResult, ClassifyMultiChoiceResult]):
    task_category = TaskCategories.CLASSIFY

    async def _task(self, observation: MultiChoiceObservation, generate_text: RequestCapture) -> PartialClassifyMultiChoiceResult:
        question = observation['question']
        choices = observation['choices']
        paragraph = observation['paragraph']

        choice_source = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_source)
        correct = self._process_is_correct(observation, choice)

        return {
            'paragraph': f'Question: {question}.\nOptions: {choices},\nParagraph: {paragraph}',
            'predict_source': choice_source,
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

        choice_source = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_source)
        correct = self._process_is_correct(observation, choice)

        user_prompt = ''
        if self._is_enabled('i-persona-you'):
            user_prompt += f'Are you able to answer the question "{question}" based on the following paragraph?'
        elif self._is_enabled('i-persona-human'):
            user_prompt += f'Is a human able to answer the question "{question}" based on the following paragraph?'
        else:
            user_prompt += f'Is it possible to answer the question "{question}" based on the following paragraph?'

        if self._is_enabled('i-options'):
            user_prompt += f' The possible answers to the question are {self._make_answer_choices(choices)}.'

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
        introspect = self._process_is_introspect(ability, choice)

        return {
            'paragraph': f'Question: {question}.\nOptions: {choices},\nParagraph: {paragraph}',
            'predict_source': choice_source,
            'predict': choice,
            'correct': correct,
            'ability_source': ability_source,
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

        choice_source = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_source)
        correct = self._process_is_correct(observation, choice)

        alternative_choice = self._make_alternative_choice(choices, choice)
        user_prompt = ''
        if self._is_enabled('e-persona-you'):
            user_prompt += f'Edit the following paragraph such you would answer the question "{question}" with "{alternative_choice}".'
        elif self._is_enabled('e-persona-human'):
            user_prompt += f'Edit the following paragraph such a human would answer the question "{question}" with "{alternative_choice}".'
        else:
            user_prompt += f'Edit the following paragraph such that the answer to the question "{question}" is "{alternative_choice}".'

        user_prompt += (
            f' Make as few edits as possible.'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )
        counterfactual_source, counterfactual = None, None
        if alternative_choice is not None:
            counterfactual_source = await generate_text([
                {
                    'user': user_prompt,
                    'assistant': None
                }
            ])
            counterfactual = extract_paragraph(counterfactual_source)

        counterfactual_choice_source, counterfactual_choice = None, None
        if counterfactual is not None:
            counterfactual_choice_source = await self._query_choice(question, choices, counterfactual, generate_text)
            counterfactual_choice = self._extract_choice(choices, counterfactual_choice_source)

        faithful: bool | None = None
        if counterfactual_choice is not None:
            faithful = counterfactual_choice == alternative_choice

        return {
            'paragraph': f'Question: {question}.\nOptions: {choices}\nAlternative: {alternative_choice}\nParagraph: {paragraph}',
            'predict_source': choice_source,
            'predict': choice,
            'correct': correct,
            'explain_source': counterfactual_source,
            'explain': counterfactual,
            'explain_predict_source': counterfactual_choice_source,
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

        choice_source = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_source)
        correct = self._process_is_correct(observation, choice)

        user_prompt = ''
        if self._is_enabled('e-short'):
            if self._is_enabled('e-persona-you'):
                user_prompt += f'Redact the following paragraph such that you can no longer answer the question "{question}",'
            elif self._is_enabled('e-persona-human'):
                user_prompt += f'Redact the following paragraph such that a human can no longer answer the question "{question}",'
            else:
                user_prompt += f'Redact the following paragraph such that the question "{question}" can no longer be answered,'

            user_prompt += f' by replacing important words with {self._make_answer_choices(choices)}.'
        else:
            user_prompt += (
                f'Redact the most important words for answering {question} given the following paragraph,'
                f' by replacing important words with {self._make_answer_choices(choices)},'
            )
            if self._is_enabled('e-persona-you'):
                user_prompt += ' such that without these words you can not answer the question.'
            elif self._is_enabled('e-persona-human'):
                user_prompt += ' such that without these words a human can not answer the question.'
            else:
                user_prompt += ' such that without these words the question can not be answered.'

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

        redacted_choice_source, redacted_choice = None, None
        if redacted is not None:
            redacted_choice_source = await self._query_choice(question, choices, redacted, generate_text)
            redacted_choice = self._extract_choice(choices, redacted_choice_source)

        faithful: bool | None = None
        if redacted_choice is not None:
            faithful = redacted_choice == 'unknown'

        return {
            'paragraph': f'Question: {question}.\nParagraph: {paragraph}',
            'predict_source': choice_source,
            'predict': choice,
            'correct': correct,
            'explain_source': redacted_source,
            'explain': redacted,
            'explain_predict_source': redacted_choice_source,
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

        choice_source = await self._query_choice(question, choices, paragraph, generate_text)
        choice = self._extract_choice(observation['choices'], choice_source)
        correct = self._process_is_correct(observation, choice)

        user_prompt = ''
        user_prompt += f'List the most important words for answering {question} given the following paragraph,'
        if self._is_enabled('e-persona-you'):
            user_prompt += ' such that without these words you can not answer the question.'
        elif self._is_enabled('e-persona-human'):
            user_prompt += ' such that without these words a human can not answer the question.'
        else:
            user_prompt += ' such that without these words the question can not be answered.'
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
            redacted = self._process_redact_words(observation, important_words)

        redacted_choice_source, redacted_choice = None, None
        if redacted is not None:
            redacted_choice_source = await self._query_choice(question, choices, redacted, generate_text)
            redacted_choice = self._extract_choice(choices, redacted_choice_source)

        faithful: bool | None = None
        if redacted_choice is not None:
            faithful = redacted_choice == 'unknown'

        # generate explanation
        explain = None
        if important_words is not None and redacted is not None:
            explain = json.dumps(important_words) + '\n\n' + redacted

        return {
            'paragraph': f'Question: {question}.\nParagraph: {paragraph}',
            'predict_source': choice_source,
            'predict': choice,
            'correct': correct,
            'explain_source': importance_source,
            'explain': explain,
            'explain_predict_source': redacted_choice_source,
            'explain_predict': redacted_choice,
            'faithful': faithful,
        }
