import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md"})


@dataclass(frozen=True, slots=True)
class Document:
    source: str
    title: str
    text: str


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return fallback


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


def iter_documents(root: Path, limit: int | None = None) -> Iterator[Document]:
    """Ліниво читає документи з директорії (.txt/.md) або з потокового .jsonl файлу."""
    if not root.exists():
        raise FileNotFoundError(f"Джерело корпусу не знайдено: {root}")

    if root.is_file():
        if root.suffix.lower() != ".jsonl":
            raise ValueError(f"Непідтримуваний формат файлу корпусу: {root.suffix}")
        count = 0
        with root.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                source = str(row.get("url") or row.get("path") or f"{root}:{line_no}")
                text = str(row.get("text") or "")
                title = str(row.get("title") or _extract_title(text, source))[:120]
                yield Document(source=source, title=title, text=text)
                count += 1
                if limit is not None and count >= limit:
                    return
        return

    for path in iter_files(root, limit=limit):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = _extract_title(text, path.stem)
        yield Document(source=str(path), title=title, text=text)
