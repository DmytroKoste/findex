import functools
import logging
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from findex.corpus import iter_files
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

IndexVariant = Literal["slots", "plain", "array"]
IndexFormat = Literal["pickle", "binary"]


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Декоратор вимірювання часу виконання функції у мілісекундах через logging."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.debug("%s виконано за %.2f ms", fn.__name__, elapsed_ms)

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
        self.postings: dict[str, list[Posting]] = postings
        self.doc_lengths: dict[int, int] = doc_lengths
        self.doc_meta: dict[int, DocMeta] = doc_meta
        self.with_positions: bool = with_positions
        self._closed: bool = False

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, key: object) -> bool:
        return key in self.postings

    def __getitem__(self, key: str) -> list[Posting]:
        return self.postings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def is_closed(self) -> bool:
        return self._closed

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
    postings: dict[str, tuple[array[int], array[int]]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


@contextmanager
def open_index(path: Path, fmt: IndexFormat | None = None) -> Generator[Index]:
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


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    *,
    variant: Literal["array"],
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
) -> ArrayInvertedIndex: ...


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    *,
    variant: Literal["plain"],
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
) -> dict[str, list[PlainPosting]]: ...


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    variant: Literal["slots"] = ...,
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
) -> Index: ...


@timed
def build_index(
    root: Path,
    with_positions: bool = False,
    variant: IndexVariant = "slots",
    limit: int | None = None,
    on_doc: Callable[[int, Path], None] | None = None,
) -> Index | ArrayInvertedIndex | dict[str, list[PlainPosting]]:
    """Будує інвертований індекс одним проходом по лінивому потоку документів."""
    raw_ids: dict[str, array[int]] = defaultdict(lambda: array("I"))
    raw_tfs: dict[str, array[int]] = defaultdict(lambda: array("I"))
    raw_slots: dict[str, list[Posting]] = defaultdict(list)
    raw_plain: dict[str, list[PlainPosting]] = defaultdict(list)

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
                raw_plain[term].append(PlainPosting(doc_id=doc_id, tf=tf))
            else:
                positions = tuple(pos_map[term]) if with_positions else ()
                raw_slots[term].append(Posting(doc_id=doc_id, tf=tf, positions=positions))

        if on_doc is not None:
            on_doc(doc_id + 1, path)

    if variant == "array":
        packed = {term: (raw_ids[term], raw_tfs[term]) for term in raw_ids}
        return ArrayInvertedIndex(postings=packed, doc_lengths=doc_lengths, doc_meta=doc_meta)

    if variant == "plain":
        return dict(raw_plain)

    ix = Index(
        postings=dict(raw_slots),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )
    log.info("Побудовано %r", ix)
    return ix
