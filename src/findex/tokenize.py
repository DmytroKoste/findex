import re
from collections.abc import Iterator

TOKEN_RE = re.compile(r"[a-zа-яіїєґ0-9]+(?:['’][a-zа-яіїєґ0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> Iterator[str]:
    """Генератор токенів у нижньому регістрі."""
    for match in TOKEN_RE.finditer(text):
        yield match.group(0).lower()