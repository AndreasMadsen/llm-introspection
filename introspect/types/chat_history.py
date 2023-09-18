
from typing import TypedDict, TypeAlias, Required

class ChatMessagePair(TypedDict):
    user: Required[str]
    assistant: Required[str | None]

ChatHistory: TypeAlias = list[ChatMessagePair]
