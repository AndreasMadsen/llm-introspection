
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
    list_content = []

    # Sure, here are the most important words for determining the sentiment of the paragraph:
    #
    # 1. Awful
    # 2. Worst
    # 3. "fun" (appears twice)
    if '\n' in source:
        for line in source.splitlines():
            if m := re.match(r'^(?:\d+\.|\*|•|-)[ \t]*(.*)$', line):
                content, = m.groups()
                if content.startswith('"') and (endqoute_pos := content.rfind('"')) > 0:
                    content = content[1:endqoute_pos]

                if len(content) > 0:
                    list_content.append(content)

    if len(list_content) == 0:
        for line in source.splitlines():
            # check for colon:
            # These are the most important words: "abc,"
            strip_idx = line.find(':') + 1
            # check for qoute words:
            # These are the most important words "abc,"
            if m := re.search(r"[\"*]([^\",\.]+)[,\.]?[\"*]", line, flags=re.IGNORECASE):
                strip_idx = min(m.start(), strip_idx)
            # check for words followed by comma:
            # These are the most important words abc,
            elif strip_idx == 0:
                if m := re.search(r"(\w+)(?:,|\.|$)", line, flags=re.IGNORECASE):
                    strip_idx = m.start()

            line = line[strip_idx:].lstrip()

            if m := re.findall(r"(?:[\"*]([^\",*\.]+)[,\.]?[\"*]|(\w[\w \-]*)(?:,|\.|$))(?:\. |, | and | or | |$|)", line, flags=re.IGNORECASE):
                for content_match_0, content_match_1 in m:
                    if len(content_match_0) > 0:
                        list_content.append(content_match_0)
                    elif len(content_match_1) > 0:
                        list_content.append(content_match_1)

    if len(list_content) == 0:
        return None
    return list_content
