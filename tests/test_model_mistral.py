
import pytest

from introspect.model import MistralModel
from introspect.types import SystemMessage
from introspect.client import OfflineClient

def test_model_mistral_render_prompt_empty():
    model = MistralModel(OfflineClient())

    with pytest.raises(ValueError) as excinfo:
        model.render_prompt([])
    assert str(excinfo.value) == "history must have at least one message pair"

def test_model_mistral_render_docs():
    # test official example from the docs
    model = MistralModel(OfflineClient())

    prompt = model.render_prompt([
        {'user': 'What is your favourite condiment?',
         'assistant': 'Well, I\'m quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I\'m cooking up in the kitchen!'},
        {'user': 'Do you have mayonnaise recipes?',
         'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] What is your favourite condiment? [/INST]"
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        "[INST] Do you have mayonnaise recipes? [/INST]"
    )


def test_model_mistral_render_system_message_default():
    model = MistralModel(OfflineClient())

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST]"
    )

def test_model_mistral_render_system_message_none():
    model = MistralModel(OfflineClient(), system_message=SystemMessage.NONE)

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST]"
    )

def test_model_mistral_render_one_message_with_assistant():
    model = MistralModel(OfflineClient())

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST][assistant message 1]"
    )

def test_model_mistral_render_two_message_no_assistant():
    model = MistralModel(OfflineClient())

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST][assistant message 1]</s>"
        " [INST] [user message 2] [/INST]"
    )

def test_model_mistral_render_two_message_with_assistant():
    model = MistralModel(OfflineClient())

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': '[assistant message 2]'}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST][assistant message 1]</s>"
        " [INST] [user message 2] [/INST][assistant message 2]"
    )
