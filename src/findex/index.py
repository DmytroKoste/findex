import argparse
import functools
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_files
from findex.tokenize import tokenize


def timed(fn):
    """Декоратор вимірювання часу виконання функції у мілісекундах."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[timed] {fn.__name__}: {elapsed_ms:.2f} ms")

    return wrapper


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс як повноцінний Python-контейнер (Mapping)."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        with_positions: bool = False,
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self.with_positions = with_positions
        self._closed = False

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        plist = self.postings.get(term)
        return len(plist) if plist is not None else 0

    def close(self) -> None:
        self._closed = True


InvertedIndex = Index


@dataclass(slots=True)
class ArrayInvertedIndex:
    postings: dict[str, tuple[array, array]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


@contextmanager
def open_index(path: Path, fmt: str | None = None) -> Iterator[Index]:
    """Контекстний менеджер для завантаження індексу та гарантованого звільнення ресурсів."""
    from findex.store import load_index

    ix = load_index(path, fmt=fmt)
    try:
        yield ix
    finally:
        ix.close()


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return fallback


@timed
def build_index(
    root: Path,
    with_positions: bool = False,
    variant: str = "slots",
    limit: int | None = None,
) -> Index | ArrayInvertedIndex | dict[str, list[PlainPosting]]:
    """Будує інвертований індекс одним проходом по лінивому потоку документів."""
    if variant == "array":
        raw_ids: dict[str, array] = defaultdict(lambda: array("I"))
        raw_tfs: dict[str, array] = defaultdict(lambda: array("I"))
    else:
        raw_postings: dict[str, list] = defaultdict(list)

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, path in enumerate(iter_files(root, limit=limit)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = _extract_title(text, path.stem)
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(path), title=title)

        tf_counter: Counter[str] = Counter()
        pos_map: dict[str, list[int]] = defaultdict(list)
        total_tokens = 0

        for pos, token in enumerate(tokenize(text)):
            total_tokens += 1
            tf_counter[token] += 1
            if with_positions and variant == "slots":
                pos_map[token].append(pos)

        doc_lengths[doc_id] = total_tokens

        for term, tf in tf_counter.items():
            if variant == "array":
                raw_ids[term].append(doc_id)
                raw_tfs[term].append(tf)
            elif variant == "plain":
                raw_postings[term].append(PlainPosting(doc_id=doc_id, tf=tf))
            else:
                positions = tuple(pos_map[term]) if with_positions else ()
                raw_postings[term].append(Posting(doc_id=doc_id, tf=tf, positions=positions))

    if variant == "array":
        packed = {term: (raw_ids[term], raw_tfs[term]) for term in raw_ids}
        return ArrayInvertedIndex(postings=packed, doc_lengths=doc_lengths, doc_meta=doc_meta)

    if variant == "plain":
        return dict(raw_postings)

    return Index(
        postings=dict(raw_postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу findex")
    parser.add_argument("corpus_dir", type=Path, help="Шлях до директорії корпусу")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Файл збереження індексу")
    parser.add_argument("--format", choices=["pickle", "binary"], default="pickle", help="Формат серіалізації")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів у постінгах")
    args = parser.parse_args()

    from findex.store import save_index

    tracemalloc.start()
    index = build_index(args.corpus_dir, with_positions=args.positions, variant="slots")
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert isinstance(index, Index)
    save_index(index, args.out, fmt=args.format)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Побудовано {index!r} | Пік RAM: {peak_bytes / (1024 * 1024):.2f} МБ")
    print(f"Збережено у '{args.out}' ({args.format}): {size_mb:.2f} МБ")


if __name__ == "__main__":
    main()