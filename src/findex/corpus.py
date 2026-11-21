from collections.abc import Iterator
from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})


def iter_files(root: Path, limit: int | None = None) -> Iterator[Path]:
    """Ліниво обходить підтримувані текстові файли у директорії корпусу."""
    if not root.exists():
        raise FileNotFoundError(f"Директорію корпусу не знайдено: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Очікувалась директорія корпусу: {root}")

    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
            count += 1
            if limit is not None and count >= limit:
                return
