
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class FalconModel(AbstractModel):
    _name = 'Falcon'
    _default_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "max_new_tokens": 1024,
        "stop": ["User:", "Falcon:"],
        "seed": 0
    }
    _support_system_message = True
    _default_system_message = (
        "The following is a conversation between a highly knowledgeable and"
        " intelligent AI assistant, called Falcon, and a human user, called"
        " User. In the following interactions, User and Falcon will converse"
        " in natural language, and Falcon will answer User's questions."
        " Falcon was built to be respectful, polite and inclusive. Falcon"
        " was built by the Technology Innovation Institute in Abu Dhabi."
        " Falcon will never decline to answer a question, and always"
        " attempts to give an answer that User would be satisfied with."
        " It knows a lot, and always tells the truth. The conversation"
        " begins."
    )

    def _render_prompt(self, history):
        prompt = ''

        if self._system_message is not None:
            prompt += f'{self._system_message}\n'

        for message_i, message_pair in enumerate(history):
            is_last = message_i == len(history) - 1

            prompt += f'User: {message_pair["user"]}\n'
            if is_last:
                if message_pair['assistant'] is None:
                    prompt += 'Falcon:'
                else:
                    prompt += f'Falcon: {message_pair["assistant"]}'
            else:
                prompt += f'Falcon: {message_pair["assistant"]}\n'

        return prompt
