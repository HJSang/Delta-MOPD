"""Exact branch-and-bound version of EvalPlus 0.3.1's code_extract.

Retains its longest syntactically valid contiguous line window, counting
nonempty lines and resolving ties by earliest start then earliest end.
The upstream loop parses every window even when it cannot beat the winner.
"""

from functools import lru_cache
import io
import re
import tokenize


# Exhaustive EvalPlus parity is valuable for ordinary answers, but its reference
# algorithm is quadratic in physical lines. Generated code can contain 100K of
# prose or word salad; cap the exact path and use syntax-bounded candidates.
_EXHAUSTIVE_CHAR_LIMIT = 32_768
_CODE_START = re.compile(
    r"^(?:async\s+def|def|class|from\s+\S+\s+import|import\s+\S+|@\w|"
    r"(?:[A-Za-z_]\w*\s*=|if\s+__name__\s*==|try\s*:|with\s+|for\s+|while\s+))"
)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


@lru_cache(maxsize=4096)
def invalid_first_line(line):
    # A lexical error on the first physical line cannot be repaired by later
    # lines. Explicit continuations are excluded; unfinished triple strings
    # raise TokenError (not ERRORTOKEN) and remain candidates.
    if "\\" in line:
        return False
    try:
        for token in tokenize.generate_tokens(io.StringIO(line + "\n").readline):
            if token.type == tokenize.INDENT:
                # A module cannot begin with an indented statement. Blank and
                # comment-only lines do not emit INDENT tokens.
                return True
            if token.type == tokenize.ERRORTOKEN and token.string.strip():
                return True
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return False


def lexical_barrier(text):
    """First line containing a lexical error outside a string or comment.

    No valid Python window starting here can include that line. Tokenize the
    whole suffix so backticks inside multiline strings remain eligible. On an
    earlier tokenization error, conservatively retain the exhaustive search.
    """
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if ((token.type == tokenize.ERRORTOKEN and token.string.strip())
                    or (token.type == tokenize.OP and token.string == "`")):
                return token.start[0] - 1
    except tokenize.TokenError as exc:
        # Python 3.12 reports an unclosed single/double-quoted physical line
        # as TokenError instead of ERRORTOKEN. Unlike an unfinished triple
        # string, later lines cannot repair this lexical error.
        if exc.args[0].startswith("unterminated string literal"):
            return exc.args[1][0] - 1
    except (IndentationError, SyntaxError):
        pass
    return None


def _looks_like_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if line[:1].isspace():
        return True
    return bool(_CODE_START.match(stripped))


def _bounded_code_extract(text: str, syntax_check) -> str:
    """Extract valid code candidates without quadratic prose scanning.

    EvalPlus later parses top-level definitions and keeps the requested
    entrypoint. For long model answers, a valid contiguous function/class block
    is the useful unit; including unrelated prose only increases extraction cost.
    This is deliberately conservative: candidates must parse as Python, and an
    invalid candidate is discarded rather than repaired or scored as code.
    """
    candidates = [match.group(1).strip() for match in _FENCE.finditer(text)]
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if not _CODE_START.match(line.lstrip()):
            continue
        end = start + 1
        while end < len(lines) and _looks_like_code_line(lines[end]):
            end += 1
        candidate = "\n".join(lines[start:end]).rstrip()
        if candidate:
            candidates.append(candidate)

    best = ""
    best_count = 0
    for candidate in candidates:
        if len(candidate) > _EXHAUSTIVE_CHAR_LIMIT * 2:
            continue
        if syntax_check(candidate):
            count = sum(bool(line.strip()) for line in candidate.splitlines())
            if count > best_count:
                best, best_count = candidate, count
    return best


@lru_cache(maxsize=1)
def code_extract(text: str) -> str:
    from evalplus.syncheck import syntax_check

    if len(text) > _EXHAUSTIVE_CHAR_LIMIT:
        return _bounded_code_extract(text, syntax_check)

    lines = text.split("\n")
    prefix = [0]
    offsets = [0]
    for line in lines:
        prefix.append(prefix[-1] + bool(line.strip()))
        offsets.append(offsets[-1] + len(line) + 1)
    first_statement = [len(lines)] * (len(lines) + 1)
    for index in range(len(lines) - 1, -1, -1):
        first_statement[index] = (first_statement[index + 1]
            if not lines[index].strip() or lines[index].lstrip().startswith("#") else index)

    def window(start, end):
        return text[offsets[start]:offsets[end + 1] - 1]

    best = 0
    winner = (0, 0)
    for start in range(len(lines)):
        if prefix[-1] - prefix[start] <= best:
            break
        maximum_end = len(lines) - 1
        statement = first_statement[start]
        if statement < len(lines) and invalid_first_line(lines[statement]):
            # Leading blank/comment lines cannot turn a later lexical error into
            # valid Python. Windows ending before that line are still eligible.
            maximum_end = statement - 1
        else:
            barrier = lexical_barrier(text[offsets[start]:])
            if barrier is not None:
                maximum_end = start + barrier - 1
        for end in range(maximum_end, start, -1):
            count = prefix[end + 1] - prefix[start]
            if count <= best:
                break
            if syntax_check(window(start, end)):
                # Descending search found this start's longest valid window.
                # Preserve the upstream earliest-end tie break for blank tails.
                earliest = end
                while earliest > start + 1 and not lines[earliest].strip():
                    if not syntax_check(window(start, earliest - 1)):
                        break
                    earliest -= 1
                best, winner = count, (start, earliest)
                break
    return window(*winner)
