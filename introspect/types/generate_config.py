
from typing import TypedDict, NotRequired

class GenerateConfig(TypedDict):
    do_sample: NotRequired[bool]
    max_new_tokens: NotRequired[int]
    best_of: NotRequired[int]
    repetition_penalty: NotRequired[float]
    return_full_text: NotRequired[bool]
    seed: NotRequired[int]
    stop_sequences: NotRequired[list[str]]
    temperature: NotRequired[float]
    top_k: NotRequired[int]
    top_p: NotRequired[float]
    truncate: NotRequired[int]
    typical_p: NotRequired[float]
    watermark: NotRequired[bool]
    decoder_input_details: NotRequired[bool]
