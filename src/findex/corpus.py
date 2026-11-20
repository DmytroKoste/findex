from collections.abc import Iterator
from pathlib import Path


def iter_files(root: Path, limit: int | None = None) -> Iterator[Path]:
    """Ліниво обходить усі .txt файли в директорії root (включно з підпапками)."""
    count = 0
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def read_documents(root: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Ліниво читає документи та віддає пари (doc_id, text)."""
    for path in iter_files(root, limit=limit):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield doc_id, text