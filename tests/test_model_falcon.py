
import pytest

from introspect.model import FalconModel, SYSTEM_MESSAGE

def test_model_falcon_render_prompt_empty():
    model = FalconModel()

    with pytest.raises(ValueError) as excinfo:
        model.render_prompt([])
    assert str(excinfo.value) == "history must have at least one message pair"

def test_model_falcon_render_system_message_default():
    model = FalconModel()

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "The following is a conversation between a highly knowledgeable and"
        " intelligent AI assistant, called Falcon, and a human user, called"
        " User. In the following interactions, User and Falcon will converse"
        " in natural language, and Falcon will answer User's questions."
        " Falcon was built to be respectful, polite and inclusive. Falcon"
        " was built by the Technology Innovation Institute in Abu Dhabi."
        " Falcon will never decline to answer a question, and always"
        " attempts to give an answer that User would be satisfied with."
        " It knows a lot, and always tells the truth. The conversation"
        " begins.\n"
        "User: [user message 1]\n"
        "Falcon:"
    )

def test_model_falcon_render_system_message_none():
    model = FalconModel(system_message=SYSTEM_MESSAGE.NONE)

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "User: [user message 1]\n"
        "Falcon:"
    )

def test_model_falcon_render_system_message_empty():
    model = FalconModel(system_message='')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "\n"
        "User: [user message 1]\n"
        "Falcon:"
    )

def test_model_falcon_render_one_message_no_assistant():
    model = FalconModel(system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': None}
    ])
    assert prompt == (
        "[system message]\n"
        "User: [user message 1]\n"
        "Falcon:"
    )

def test_model_falcon_render_one_message_with_assistant():
    model = FalconModel(system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'}
    ])
    assert prompt == (
        "[system message]\n"
        "User: [user message 1]\n"
        "Falcon: [assistant message 1]"
    )

def test_model_falcon_render_two_message_no_assistant():
    model = FalconModel(system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': None}
    ])
    assert prompt == (
        "[system message]\n"
        "User: [user message 1]\n"
        "Falcon: [assistant message 1]\n"
        "User: [user message 2]\n"
        "Falcon:"
    )

def test_model_falcon_render_two_message_with_assistant():
    model = FalconModel(system_message='[system message]')

    prompt = model.render_prompt([
        {'user': '[user message 1]', 'assistant': '[assistant message 1]'},
        {'user': '[user message 2]', 'assistant': '[assistant message 2]'}
    ])
    assert prompt == (
        "[system message]\n"
        "User: [user message 1]\n"
        "Falcon: [assistant message 1]\n"
        "User: [user message 2]\n"
        "Falcon: [assistant message 2]"
    )
