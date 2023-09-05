
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Llama2Model(AbstractModel):
    _name = 'Llama2'
    _default_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "truncate": 1000,
        "max_new_tokens": 1024,
        "stop_sequences": ["[INST]", "[/INST]", "<s>", "</s>"],
        "seed": 0
    }
    _default_system_message = (
        "You are a helpful, respectful and honest assistant. Always answer as"
        " helpfully as possible, while being safe. Your answers should not"
        " include any harmful, unethical, racist, sexist, toxic, dangerous,"
        " or illegal content. Please ensure that your responses are socially"
        " unbiased and positive in nature.\n\nIf a question does not make"
        " any sense, or is not factually coherent, explain why instead of"
        " answering something not correct. If you don't know the answer to"
        " a question, please don't share false information."
    )

    def _render_prompt(self, history):
        prompt = ''

        for message_i, message_pair in enumerate(history):
            is_first = message_i == 0
            is_last = message_i == len(history) - 1

            prompt += '<s>[INST] '
            if is_first and self._system_message is not None:
                prompt += f'<<SYS>>\n{self._system_message}\n<</SYS>>\n\n'
            prompt += f'{message_pair["user"]} [/INST]'
            if is_last:
                if message_pair["assistant"] is not None:
                    prompt += f' {message_pair["assistant"]}'
            else:
                prompt += f' {message_pair["assistant"]} </s>'

        return prompt
