
import pathlib
from dataclasses import dataclass
from typing import Type

import pytest

from introspect.model import Llama2Model
from introspect.client import TestClient as CreateTestClient
from introspect.dataset import IMDBDataset
from introspect.tasks import AbstractTask, SentimentClassifyTask
from introspect.types import SentimentObservation, SystemMessage

@pytest.fixture
def sentiment_obs() -> SentimentObservation:
    return {
        'label': 0,
        'idx': 1,
        'text': '[PARAGRAPH]'
    }

@dataclass
class RequestExpectations:
    task_type: Type[AbstractTask]
    config: list[str]
    expected_requests: list[str]

task_requests = [
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=[],
        expected_requests=[
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-persona-you'],
        expected_requests=[
            '<s>[INST] What would you classify the sentiment of the following paragraph as?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-persona-human'],
        expected_requests=[
            '<s>[INST] What would a human classify the sentiment of the following paragraph as?'
            ' The paragraph can contain redacted words marked with [REDACTED].'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted'],
        expected_requests=[
            '<s>[INST] What is the sentiment of the following paragraph?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted', 'c-persona-you'],
        expected_requests=[
            '<s>[INST] What would you classify the sentiment of the following paragraph as?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    ),
    RequestExpectations(
        task_type=SentimentClassifyTask,
        config=['c-no-redacted', 'c-persona-human'],
        expected_requests=[
            '<s>[INST] What would a human classify the sentiment of the following paragraph as?'
            ' Answer only "positive", "negative", "neutral", or "unknown".'
            ' Do not explain the answer.'
            '\n'
            '\n'
            'Paragraph: [PARAGRAPH] [/INST]'
        ]
    )
]

@pytest.mark.parametrize("info", task_requests, ids=lambda info: f'{info.task_type.task_category}:{",".join(info.config)}')
@pytest.mark.asyncio
async def test_task_sentiment_config(sentiment_obs: SentimentObservation, info: RequestExpectations):
    client = CreateTestClient()
    task = info.task_type(
        IMDBDataset(persistent_dir=pathlib.Path('.')),
        Llama2Model(client, system_message=SystemMessage.NONE),
        config=info.config
    )
    await task(sentiment_obs)
    assert client.log == info.expected_requests
