import unicodedata
import re

from scalingqa.common.drqa_tokenizers.regexp_tokenizer import RegexpTokenizer
from scalingqa.common.drqa_tokenizers.simple_tokenizer import SimpleTokenizer


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def has_answer_decorator(tokenizer):
    """
    Decorator inject tokenizer into global function scope.
    """

    def inner(foo):
        def wrapper(*args, **kwargs):
            return foo(*args, **kwargs, tokenizer=tokenizer)

        return wrapper

    return inner


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException as e:
        return False
    return pattern.search(text) is not None


def has_answer(answers, text, tokenizer, match_type='string') -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = normalize(text)

    if match_type == 'string':
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == 'regex':
        # Answer is a regex
        for single_answer in answers:
            single_answer = normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


has_answer_dpr = has_answer_decorator(SimpleTokenizer())(has_answer)
has_answer_drqa = has_answer_decorator(RegexpTokenizer())(has_answer)
