
import pytest

from introspect.model import Llama2Model
from introspect.types import SystemMessage
from introspect.client import OfflineClient

def test_model_llama2_render_prompt_empty():
    model = Llama2Model(OfflineClient())

    with pytest.raises(ValueError) as excinfo:
        model.render_prompt([])
    assert str(excinfo.value) == "history must have at least one message pair"

def test_model_llama2_render_system_message_default():
    model = Llama2Model(OfflineClient())

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as"
        " helpfully as possible, while being safe. Your answers should not"
        " include any harmful, unethical, racist, sexist, toxic, dangerous,"
        " or illegal content. Please ensure that your responses are socially"
        " unbiased and positive in nature.\n\nIf a question does not make"
        " any sense, or is not factually coherent, explain why instead of"
        " answering something not correct. If you don't know the answer to"
        " a question, please don't share false information.\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST]"
    )

def test_model_llama2_render_system_message_none():
    model = Llama2Model(OfflineClient(), system_message=SystemMessage.NONE)

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] [user message 1] [/INST]"
    )

def test_model_llama2_render_system_message_empty():
    model = Llama2Model(OfflineClient(), system_message='')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST]"
    )

def test_model_llama2_render_one_message_no_assistant():
    model = Llama2Model(OfflineClient(), system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "[system message]\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST]"
    )

def test_model_llama2_render_one_message_with_assistant():
    model = Llama2Model(OfflineClient(), system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "[system message]\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST] [assistant message 1]"
    )

def test_model_llama2_render_two_message_no_assistant():
    model = Llama2Model(OfflineClient(), system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': None}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "[system message]\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST] [assistant message 1] </s>"
        "<s>[INST] [user message 2] [/INST]"
    )

def test_model_llama2_render_two_message_with_assistant():
    model = Llama2Model(OfflineClient(), system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': '[assistant message 2]'}
    ])
    assert prompt == (
        "<s>[INST] <<SYS>>\n"
        "[system message]\n"
        "<</SYS>>\n"
        "\n"
        "[user message 1] [/INST] [assistant message 1] </s>"
        "<s>[INST] [user message 2] [/INST] [assistant message 2]"
    )
