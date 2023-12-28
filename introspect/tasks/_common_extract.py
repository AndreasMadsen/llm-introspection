
from typing import Literal
import re
import regex
from ._common_match import match_startwith, search_contains, match_contains, count_contains, replace_contains

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
    paragraph = source

    # Multiple suggestions
    # Example:
    # Here are some suggestions based on your request:
    # <ul>
    # ...
    if match_contains(('here are some suggestions', ))(paragraph) and count_contains(('<li>', '<p>', '- '))(paragraph) > 1:
        return None

    # Example:
    # Here are some suggestions based on your request:
    # <p>I am excited to watch this movie! After watching the trailer...
    if match_contains(('<p>', '<ul>', '<ol>', '<blockqoute>'))(paragraph):
        paragraph = paragraph[paragraph.find('<'):]

    # Example:
    # We could do make the follow changes:
    # - ABC
    # - DEF
    # - GHI
    #
    # Here's one edited version:'
    # {content ...}
    #
    # Paragraph: {content ...}
    #
    # Paragraph:\n
    # \n
    # {content ...}
    elif m := search_contains((
        'I edited the paragraph as follows:',
        'to:',
        'Paragraph:',
        'sentiments:',
        'Sentiment:',
        'Analysis:',
        'version of it:',
        'the paragraph could be:',
        'it can be rewritten as follows:',
        'you can change the first sentence to something like this:',
        'I have edited the given paragraph as follows:',
        'we can modify the last sentence to read:',
        'The edited paragraph reads:',
        'The revised paragraph reads:',
        'the given paragraph could be:',
        'We could say something like:',
        'while removing unnecessary details:',
        'we can change the last sentence to something like this:',
        'I have edited the paragraph as follows:',
        'paragraph could be:',
        'without changing its meaning:',
        'the paragraph could be something like this:',
        'without explaining why:'
    ), find_last=True)(paragraph):
        paragraph = paragraph[m.end():]

    elif m := regex.search(r"\b(Here's|Here are|Here is)\b[\w ]*:", paragraph, flags=regex.REVERSE | regex.VERSION1):
        paragraph = paragraph[m.end():]

    # Example:
    # Sure, here is the paragraph with positive sentiment.\n
    # \n
    # {content ...}
    elif match_startwith((
        'Sure, here\'s',
        'Sure, here is',
        'Sure thing! Here\'s',
        'Sure! Here\'s',
        'Sure! Here is',
        'Here is',
        'Here\'s',
        'Thank you for sharing your thoughts',
        'Thank you for sharing your opinion',
        'I am sorry,'
        'Thank you for your feedback.',
        'This paragraph has been edited',
        'What specific changes',
        'What specific aspects of the original text did you change',
        'This paragraph',
        'Thank you for sharing this information with me.'
    ))(paragraph):
        first_break_index = paragraph.find('\n', 0)
        if first_break_index < 0:
            return None
        paragraph = paragraph[first_break_index + 1:]

    # Remove HTML, primarily by Falcon
    paragraph = replace_contains((
        '<p>',
        '</p>',
        '<ul>',
        '</ul>',
        '<ol>',
        '</ol>',
        '<blockquote>',
        '</blockquote>',
    ), '\n\n')(paragraph)
    paragraph = replace_contains(('<br>', '<br />'), '\n')(paragraph)
    paragraph = paragraph.replace('<li>', '* ').replace('</li>', '')

    # Reduce newlines
    paragraph = re.sub(r'\n\n+', '\n\n', paragraph)
    # Strip space on each line
    paragraph = '\n'.join(line.strip() for line in paragraph.splitlines())
    # Strip surronding newlines
    paragraph = paragraph.strip()

    # Remove qoutes
    # "{content ...}"
    if paragraph.startswith('"') and paragraph.endswith('"'):
        paragraph = paragraph[1:-1]

    if len(paragraph) == 0:
        return None
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
