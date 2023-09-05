
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

    def render_prompt(self, history: ChatHistory) -> str:
        """Converts a chat-history to a prompt string

        Example:
        render_prompt([
            { 'user': 'What is 1 + 1?', 'assistant': None }
        ])

        Args:
            history (ChatHistory): A structured history. For all message-pairs
                `user` and `assistant` must be provided. For the final message
                `assistant` can be either None, or a partial assistant message.

        Returns:
            str: The prompt
        """
        if len(history) < 1:
            raise ValueError('history must have at least one message pair')

        return self._render_prompt(history)

    @abstractmethod
    def _render_prompt(self, history: ChatHistory) -> str:
        ...

    async def generate_text(self, history: ChatHistory) -> str:
        """Run inference, using the prompt generated from the message history.

        Args:
            history (ChatHistory): A structured history. See `help(self.render_prompt)`
                for details.

        Raises:
            RuntimeError: If a `client` was not provided to the constructor,
                an error is raied.

        Returns:
            str: Model response. Leading and trailing space is removed.
        """
        if self._client is None:
            raise RuntimeError('no client was specified in the model constructor, can not generate')

        prompt = self.render_prompt(history)
        answer = await self._client.generate(prompt, self.config)

        if self._debug:
            print(f'PROMPT: 「{prompt}」')
            print(f'ANSWER: 「{answer.generated_text}」')
            print(f'PERPLEXITY: {sum(token.logprob for token in answer.details.tokens)}')

        return answer.generated_text.strip()
