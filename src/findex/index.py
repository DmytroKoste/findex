import functools
import logging
import math
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Literal, overload

from findex.corpus import Document, iter_documents, iter_files
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

IndexVariant = Literal["slots", "plain", "array"]
IndexFormat = Literal["pickle", "binary"]
ExecutorKind = Literal["serial", "threads", "processes"]


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


@dataclass(slots=True)
class PartialIndex:
    """Частковий індекс для одного чанку документів (придатний для pickle)."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    with_positions: bool = False


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


def _chunk_indexed_items[T](
    indexed_items: Sequence[tuple[int, T]],
    num_chunks: int,
) -> list[list[tuple[int, T]]]:
    """Розбиває нумеровані елементи на неперетинні чанки з унікальними doc_id."""
    if not indexed_items:
        return []
    n = max(1, min(num_chunks, len(indexed_items)))
    chunk_size = math.ceil(len(indexed_items) / n)
    return [list(indexed_items[i : i + chunk_size]) for i in range(0, len(indexed_items), chunk_size)]


def build_partial(
    chunk: Sequence[tuple[int, Path | Document]],
    with_positions: bool = False,
    on_doc: Callable[[int, Path], None] | None = None,
) -> PartialIndex:
    """Будує частковий індекс для заданого списку пар (doc_id, Path | Document).

    Функція верхнього рівня без замикань — безпечна для серіалізації через pickle
    та запуску у ProcessPoolExecutor (включно з методом spawn).
    """
    raw_slots: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, item in chunk:
        if isinstance(item, Document):
            text = item.text
            title = item.title
            source_str = item.source
            callback_path = Path(item.source)
        else:
            text = item.read_text(encoding="utf-8", errors="ignore")
            title = _extract_title(text, item.stem)
            source_str = str(item)
            callback_path = item

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=source_str, title=title)
        tf_counter: Counter[str] = Counter()
        pos_map: dict[str, list[int]] = defaultdict(list)
        total_tokens = 0
        for pos, token in enumerate(tokenize(text)):
            total_tokens += 1
            tf_counter[token] += 1
            if with_positions:
                pos_map[token].append(pos)
        doc_lengths[doc_id] = total_tokens
        for term, tf in tf_counter.items():
            positions = tuple(pos_map[term]) if with_positions else ()
            raw_slots[term].append(Posting(doc_id=doc_id, tf=tf, positions=positions))
        if on_doc is not None:
            on_doc(doc_id + 1, callback_path)

    return PartialIndex(
        postings=dict(raw_slots),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Об'єднує часткові індекси у фінальний Index зі збереженням порядку doc_id."""
    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    with_positions = False

    for part in partials:
        with_positions = with_positions or part.with_positions
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
        for term, plist in part.postings.items():
            merged_postings[term].extend(plist)

    by_doc_id = attrgetter("doc_id")
    for plist in merged_postings.values():
        if len(plist) > 1:
            plist.sort(key=by_doc_id)

    ordered_lengths = dict(sorted(doc_lengths.items()))
    ordered_meta = dict(sorted(doc_meta.items()))
    return Index(
        postings=dict(merged_postings),
        doc_lengths=ordered_lengths,
        doc_meta=ordered_meta,
        with_positions=with_positions,
    )


def _build_legacy_variant(
    root: Path,
    with_positions: bool,
    variant: Literal["array", "plain"],
    limit: int | None,
    on_doc: Callable[[int, Path], None] | None,
) -> ArrayInvertedIndex | dict[str, list[PlainPosting]]:
    raw_ids: dict[str, array[int]] = defaultdict(lambda: array("I"))
    raw_tfs: dict[str, array[int]] = defaultdict(lambda: array("I"))
    raw_plain: dict[str, list[PlainPosting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, path in enumerate(iter_files(root, limit=limit)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = _extract_title(text, path.stem)
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(path), title=title)
        tf_counter: Counter[str] = Counter()
        total_tokens = 0
        for token in tokenize(text):
            total_tokens += 1
            tf_counter[token] += 1
        doc_lengths[doc_id] = total_tokens
        for term, tf in tf_counter.items():
            if variant == "array":
                raw_ids[term].append(doc_id)
                raw_tfs[term].append(tf)
            else:
                raw_plain[term].append(PlainPosting(doc_id=doc_id, tf=tf))
        if on_doc is not None:
            on_doc(doc_id + 1, path)

    if variant == "array":
        packed = {term: (raw_ids[term], raw_tfs[term]) for term in raw_ids}
        return ArrayInvertedIndex(postings=packed, doc_lengths=doc_lengths, doc_meta=doc_meta)
    return dict(raw_plain)


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    *,
    variant: Literal["array"],
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
    workers: int = ...,
    executor: ExecutorKind = ...,
    merge_time_out: list[float] | None = ...,
) -> ArrayInvertedIndex: ...


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    *,
    variant: Literal["plain"],
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
    workers: int = ...,
    executor: ExecutorKind = ...,
    merge_time_out: list[float] | None = ...,
) -> dict[str, list[PlainPosting]]: ...


@overload
def build_index(
    root: Path,
    with_positions: bool = ...,
    variant: Literal["slots"] = ...,
    limit: int | None = ...,
    on_doc: Callable[[int, Path], None] | None = ...,
    workers: int = ...,
    executor: ExecutorKind = ...,
    merge_time_out: list[float] | None = ...,
) -> Index: ...


@timed
def build_index(
    root: Path,
    with_positions: bool = False,
    variant: IndexVariant = "slots",
    limit: int | None = None,
    on_doc: Callable[[int, Path], None] | None = None,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    merge_time_out: list[float] | None = None,
) -> Index | ArrayInvertedIndex | dict[str, list[PlainPosting]]:
    """Будує інвертований індекс у серійному, потоковому або процесному режимі."""
    if workers < 1:
        raise ValueError(f"Кількість воркерів має бути >= 1, отримано: {workers}")

    if variant in ("array", "plain"):
        return _build_legacy_variant(root, with_positions, variant, limit, on_doc)

    indexed_items: list[tuple[int, Path | Document]]
    if root.is_file():
        indexed_items = list(enumerate(iter_documents(root, limit=limit)))
    else:
        indexed_items = list(enumerate(iter_files(root, limit=limit)))

    partials: list[PartialIndex]
    if executor == "serial":
        num_chunks = max(1, workers)
        chunks = _chunk_indexed_items(indexed_items, num_chunks)
        partials = [build_partial(chunk, with_positions=with_positions, on_doc=on_doc) for chunk in chunks]
    elif executor == "threads":
        chunks = _chunk_indexed_items(indexed_items, workers)
        partials = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_partial, chunk, with_positions, None): chunk for chunk in chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                partials.append(fut.result())
                if on_doc is not None:
                    for doc_id, item in chunk:
                        cb_path = Path(item.source) if isinstance(item, Document) else item
                        on_doc(doc_id + 1, cb_path)
    elif executor == "processes":
        chunks = _chunk_indexed_items(indexed_items, workers)
        partials = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_partial, chunk, with_positions, None): chunk for chunk in chunks}
            try:
                for fut in as_completed(futures):
                    chunk = futures[fut]
                    partials.append(fut.result())
                    if on_doc is not None:
                        for doc_id, item in chunk:
                            cb_path = Path(item.source) if isinstance(item, Document) else item
                            on_doc(doc_id + 1, cb_path)
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
    else:
        raise ValueError(f"Невідомий тип executor: {executor}")

    t_merge0 = time.perf_counter()
    ix = merge(partials)
    merge_elapsed = time.perf_counter() - t_merge0
    if merge_time_out is not None:
        merge_time_out.append(merge_elapsed)

    log.info(
        "Побудовано %r (executor=%s, workers=%d, merge=%.3fs)",
        ix,
        executor,
        workers,
        merge_elapsed,
    )
    return ix
