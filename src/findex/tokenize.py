import re
from collections.abc import Iterable, Iterator

TOKEN_RE: re.Pattern[str] = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Ліниво розбиває рядок на нормалізовані (casefold) токени."""
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).casefold()
        if token:
            yield token


def tokenize_all(texts: Iterable[str]) -> Iterator[str]:
    """Ліниво токенізує потік текстів."""
    for text in texts:
        yield from tokenize(text)
