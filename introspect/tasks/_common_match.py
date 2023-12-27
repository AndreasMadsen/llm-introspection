
from functools import cache
from typing import Iterable, Callable
import re

@cache
def match_startwith(prefixes: Iterable[str]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(prefix_re, flags=re.IGNORECASE)
    return lambda content: r.match(content) is not None

@cache
def match_contains(prefixes: Iterable[str]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'

    r = re.compile(r'\b' + prefix_re + r'(\b|\W)', flags=re.IGNORECASE)
    return lambda content: r.search(content) is not None

@cache
def match_pair_match(prefixes: Iterable[str], postfixes: Iterable[str]) -> Callable[[str], bool]:
    prefix_re = '(?:' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'
    postfix_re = '(?:' + '|'.join(re.escape(postfix) for postfix in postfixes) + ')'

    r = re.compile(r'\b' + prefix_re + ' ' + postfix_re + r'(\b|\W)', flags=re.IGNORECASE)
    return lambda content: r.search(content) is not None
