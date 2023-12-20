
import pathlib
from dataclasses import dataclass
from typing import Type

import pytest

from introspect.model import Llama2Model
from introspect.client import TestClient as CreateTestClient
from introspect.dataset import IMDBDataset
from introspect.tasks import AbstractTask, SentimentClassifyTask, SentimentCounterfactualTask
from introspect.types import SentimentObservation, SystemMessage

@pytest.fixture
def sentiment_obs() -> SentimentObservation:
    return {
        'label': 'negative',
        'idx': 1,
        'text': '[PARAGRAPH]'
    }

@dataclass
class RequestExpectations:
    task_type: Type[AbstractTask]
    expected_requests: dict[str, str|None]
    config: list[str]
    name: str|None = None

task_requests = [
    #
    # Classify configs
    #
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=[],
        expected_requests={
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['m-removed'],
        expected_requests={
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain removed words marked with [REMOVED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-persona-you'],
        expected_requests={
            '<s>[INST] What would you classify the sentiment of the following paragraph as?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-persona-human'],
        expected_requests={
            '<s>[INST] What would a human classify the sentiment of the following paragraph as?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted'],
        expected_requests={
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted', 'c-persona-you'],
        expected_requests={
            '<s>[INST] What would you classify the sentiment of the following paragraph as?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted', 'c-persona-human'],
        expected_requests={
            '<s>[INST] What would a human classify the sentiment of the following paragraph as?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted', 'c-persona-human'],
        expected_requests={
            '<s>[INST] What would a human classify the sentiment of the following paragraph as?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': None
        }
    ),

    #
    # Counterfactual, target label is infered
    #
    RequestExpectations(
        task_type=SentimentCounterfactualTask,
        config=[],
        name='positive',
        expected_requests={
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': 'Negative',

            '<s>[INST] Edit the following paragraph such that the sentiment is "positive".'
            ' Make as few edits as possible.'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': '[Counterfactual content ...]',

            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [Counterfactual content ...] [/INST]': None,
        }
    ),
    RequestExpectations(
        task_type=SentimentCounterfactualTask,
        config=[],
        name='negative',
        expected_requests={
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': 'Negative',

            '<s>[INST] Edit the following paragraph such that the sentiment is "positive".'
            ' Make as few edits as possible.'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]': '[Counterfactual content ...]',

            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [Counterfactual content ...] [/INST]': None,
        }
    )
]

@pytest.mark.parametrize("info", task_requests, ids=lambda info: f'{info.task_type.task_category}:{info.name if info.name else ",".join(info.config)}')
@pytest.mark.asyncio
async def test_task_sentiment_config(sentiment_obs: SentimentObservation, info: RequestExpectations):
    client = CreateTestClient(response = info.expected_requests)
    task = info.task_type(
        Llama2Model(client, system_message=SystemMessage.NONE),
        config=info.config
    )
    await task(sentiment_obs)
    assert client.log == list(info.expected_requests.keys())
