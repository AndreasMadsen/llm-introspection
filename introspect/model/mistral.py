
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class MistralModel(AbstractModel):
    _name = 'Mistral'
    _default_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "max_new_tokens": 1024,
        "stop": ["[INST]", "[/INST]", "</s>"],
        "seed": 0
    }
    _support_system_message = False

    def _render_prompt(self, history):
        prompt = ''

        for message_i, message_pair in enumerate(history):
            is_first = message_i == 0
            is_last = message_i == len(history) - 1

            if is_first:
                prompt += '<s>'
            else:
                prompt += ' '

            prompt += f'[INST] {message_pair["user"]} [/INST]'
            if is_last:
                if message_pair["assistant"] is not None:
                    prompt += f'{message_pair["assistant"]}'
            else:
                prompt += f'{message_pair["assistant"]}</s>'

        return prompt
