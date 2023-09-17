
from tblib import pickling_support
from typing import TypedDict, NotRequired, Required

class GenerateConfig(TypedDict):
    # Maximum number of generated tokens
    max_new_tokens: NotRequired[int]

    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: NotRequired[int]

    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: NotRequired[list[str]]

    # The value used to module the logits distribution.
    temperature: NotRequired[float]

    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: NotRequired[int]

    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: NotRequired[float]

    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: NotRequired[float]

    # Random sampling seed
    seed: NotRequired[int]

class GenerateResponse(TypedDict):
    response: Required[str]
    duration: Required[float]

class GenerateError(Exception):
    pass

class OfflineError(GenerateError):
    pass

pickling_support.install(GenerateError, OfflineError)
