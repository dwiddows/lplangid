import re
import string
from typing import List


def tokenize_fast(input_text: str) -> List[str]:
    """Returns a very naive whitespace and punctuation based tokenization.

    This helps for most but not all languages, should only be used if you don't know the language yet,
    or if you have a lot of data and can sacrifice a lot of output quality for the sake of speed.
    """
    return strip_most_punctuation(remove_html_tags(input_text)).split()


def remove_html_tags(input_text: str) -> str:
    """Removes all text enclosed by angle brackets."""
    html_regex = re.compile("<.*?>")
    return re.sub(html_regex, "", input_text)


def strip_most_punctuation(input_text: str) -> str:
    """Removes most punctuation except for particular characters inside a word.

    E.g., "The dog." becomes "The dog" but "U.S.A." becomes "U.S.A".
    """
    chars = [c for c in input_text]
    for i in range(len(chars)):
        if chars[i] in string.punctuation:
            if ((chars[i] in "'./?&=:")
                    and 0 < i < len(chars) - 1 and not chars[i-1].isspace() and not chars[i+1].isspace()):
                continue
            chars[i] = ' '
    return ''.join(chars)
