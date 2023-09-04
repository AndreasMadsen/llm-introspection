
from abc import ABCMeta, abstractmethod

from ..types import ChatHistory, GenerateConfig, SystemMessage
from ..client import AbstractClient

class AbstractModel(metaclass=ABCMeta):
    _name: str
    _default_config: GenerateConfig
    _default_system_message: str
    _debug: bool

    _config: GenerateConfig
    _system_message: str|None

    def __init__(self,
                 client: AbstractClient|None = None,
                 system_message: str|SystemMessage = SystemMessage.DEFAULT,
                 config: GenerateConfig|None = None,
                 debug: bool=False) -> None:
        """Model object which manages prompt generation and configuration

        Args:
            client (TGIClient | None, optional): Client to generative model. Defaults to None.
            system_message (str | SYSTEM_MESSAGE, optional): The system message.
                No system message can be chosen using SYSTEM_MESSAGE.NONE.
                A custom message can be set using a string.
                Defaults to SYSTEM_MESSAGE.DEFAULT.
            config (GenerateConfig | None, optional): Configuation controling the LLM model. Defaults to None.
            debug (bool, optional): If True, generative model exchances are printed to stdout. Defaults to False.
        """
        self._client = client

        match system_message:
            case SystemMessage.DEFAULT:
                self._system_message = self._default_system_message
            case SystemMessage.NONE:
                self._system_message = None
            case _:
                self._system_message = system_message

        self._config = self._default_config if config is None else {**self._default_config, **config}
        self._debug = debug

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> GenerateConfig:
        return self._config

    @abstractmethod
    def render_prompt(self, history: ChatHistory) -> str:
        ...

    async def generate_text(self, history: ChatHistory) -> str:
        if self._client is None:
            raise RuntimeError('no client was specified in the model constructor, can not generate')

        prompt = self.render_prompt(history)
        answer = await self._client.generate(prompt, self.config)

        if self._debug:
            print(f'PROMPT: 「{prompt}」')
            print(f'ANSWER: 「{answer.generated_text}」')
            print(f'PERPLEXITY: {sum(token.logprob for token in answer.details.tokens)}')

        return answer.generated_text.strip()
