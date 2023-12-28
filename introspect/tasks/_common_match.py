
from functools import cache
from typing import Callable
import re
import regex

@cache
def match_startwith(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(prefix_re, flags=re.IGNORECASE)
    return lambda content: r.match(content) is not None

@cache
def match_pair_match(prefixes: tuple[str, ...], postfixes: tuple[str, ...]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'
    postfix_re = '(?:' + '|'.join(re.escape(postfix) for postfix in postfixes) + ')'

    r = re.compile(r'\b' + prefix_re + ' ' + postfix_re + r'(\b|\W)', flags=re.IGNORECASE)
    return lambda content: r.search(content) is not None

@cache
def match_contains(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(r'\b' + prefix_re + r'(\b|\W)', flags=re.IGNORECASE)
    return lambda content: r.search(content) is not None

@cache
def search_contains(prefixes: tuple[str, ...], find_last=False) -> Callable[[str], regex.Match[str]|None]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'
    pattern = r'\b' + prefix_re + r'(\b|\W)'

    if find_last:
        r = regex.compile(pattern, flags=regex.IGNORECASE | regex.REVERSE | regex.VERSION1)
    else:
        r = regex.compile(pattern, flags=regex.IGNORECASE | regex.VERSION1)

    return lambda content: r.search(content)

@cache
def count_contains(prefixes: tuple[str, ...]) -> Callable[[str], int]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(prefix_re, flags=re.IGNORECASE)
    return lambda content: len(r.findall(content))

@cache
def replace_contains(prefixes: tuple[str, ...], replace_content) -> Callable[[str], str]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(prefix_re, flags=re.IGNORECASE)
    return lambda content: r.sub(replace_content, content)
