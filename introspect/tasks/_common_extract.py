
from typing import Literal
import re

def _startwith(content: str, options: list[str]) -> bool:
    return any(content.startswith(pattern) for pattern in options)

def extract_ability(source: str) -> Literal['yes', 'no']|None:
    match source.lower():
        case 'yes' | 'yes.':
            ability = 'yes'
        case 'no' | 'no.':
            ability = 'no'
        case _:
            ability = None
    return ability

def extract_paragraph(source: str) -> str|None:
    # Paragraph: {content ...}
    if source.startswith('Paragraph: '):
        return source.removeprefix('Paragraph: ')

    # Sure, here is the paragraph with positive sentiment.\n
    # \n
    # {content ...}
    # Paragraph:\n
    # \n
    # {content ...}
    paragraph = source
    if _startwith(source, ['Sure, here\'s', 'Sure, here is', 'Sure! Here is', 'Sure! Here\'s', 'Sure thing! Here\'s',
                            'Here is', 'Here\'s', 'Paragraph:']):
        first_break_index = source.find('\n\n')
        if first_break_index >= 0:
            paragraph = source[first_break_index + 2:]

    # Remove qoutes
    # "{content ...}"
    if paragraph.startswith('"') and paragraph.endswith('"'):
        return paragraph[1:-1]

    return paragraph

def extract_list_content(source: str) -> list[str]|None:
    # The source tends to have the format:
    # Sure, here are the most important words for determining the sentiment of the paragraph:
    #
    # 1. Awful
    # 2. Worst
    # 3. "fun" (appears twice)
    list_content = []
    for line in source.splitlines():
        if m := re.match(r'^(?:\d+\.|\*|•|-)[ \t]*(.*)$', line):
            content, = m.groups()
            if content.startswith('"') and (endqoute_pos := content.rfind('"')) > 0:
                content = content[1:endqoute_pos]

            if len(content) > 0:
                list_content.append(content)

    if len(list_content) == 0:
        return None
    return list_content
