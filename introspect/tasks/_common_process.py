
import re

def process_redact_words(content: str, important_words: list[str], mask_token: str) -> str:
    return re.sub(
        r'\b(?:' + '|'.join(re.escape(word) for word in important_words) + r')\b',
        mask_token,
        content,
        flags=re.IGNORECASE
    )
