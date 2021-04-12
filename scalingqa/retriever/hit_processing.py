import re

from ..common.drqa_tokenizers.simple_tokenizer import SimpleTokenizer
from ..common.utility.metrics import normalize

dpr_tokenizer = None


def process_hit_token_dpr(e, db, match_type="string"):
    global dpr_tokenizer
    if dpr_tokenizer is None:
        dpr_tokenizer = SimpleTokenizer()

    def regex_match(text, pattern):
        """Test if a regex pattern is contained within a text."""
        try:
            pattern = re.compile(
                pattern,
                flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
            )
        except BaseException:
            return False
        return pattern.search(text) is not None

    def has_answer(answers, text, tokenizer, match_type) -> bool:
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

    top, answers, raw_question = e
    if type(top) != list:
        top = top.tolist()

    for rank, t in enumerate(top):
        text = db.get_doc_text(t)[0]
        if has_answer(answers, text, dpr_tokenizer, match_type):
            return {"hit": True, "hit_rank": rank}
    return {"hit": False, "hit_rank": -1}
