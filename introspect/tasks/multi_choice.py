
from typing import TypeAlias
import re

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

PartialClassifyMultiChoiceResult: TypeAlias = PartialClassifyResult[str]
ClassifyMultiChoiceResult: TypeAlias = ClassifyResult[str, str]
PartialIntrospectMultiChoiceResult: TypeAlias = PartialIntrospectResult[str]
IntrospectMultiChoiceResult: TypeAlias = IntrospectResult[str, str]
PartialFaithfulMultiChoiceResult: TypeAlias = PartialFaithfulResult[str]
FaithfulMultiChoiceResult: TypeAlias = FaithfulResult[str, str]

class MultiChoiceTask(AbstractTask[MultiChoiceDataset, MultiChoiceObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.MULTI_CHOICE

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
            user_prompt += ' The paragraph can contain redacted words marked with [REDACTED].'

        options_string = ', '.join(f'"{chr(ord("a") + choice_i)}) {choice}"' for choice_i, choice in enumerate(choices))
        user_prompt += (
            f' Answer either {options_string}, or {chr(ord("a") + len(choices))}) "unknown".'
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
                return 'unkonwn'

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
            'paragraph': f'Question: {question}.\n Options: {choices},\n Paragraph: {paragraph}',
            'predict_source': choice_source,
            'predict': choice,
            'correct': correct
        }
