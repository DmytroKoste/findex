import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_files
from findex.tokenize import tokenize


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
class InvertedIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    with_positions: bool = False


@dataclass(slots=True)
class ArrayInvertedIndex:
    postings: dict[str, tuple[array, array]]  # term -> (doc_ids: array('I'), tfs: array('I'))
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return fallback


def build_index(
    root: Path,
    with_positions: bool = False,
    variant: str = "slots",
    limit: int | None = None,
) -> InvertedIndex | ArrayInvertedIndex | dict[str, list[PlainPosting]]:
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

    return InvertedIndex(
        postings=dict(raw_postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def run_memory_study(root: Path, limit: int | None = None) -> None:
    """Порівнює 3 варіанти зберігання постінгів під tracemalloc (Етап M4)."""
    from findex.store import load_index, save_index

    variants = [
        ("plain @dataclass", "plain", False),
        ("slots=True @dataclass", "slots", False),
        ("slots=True + positions", "slots", True),
        ("array('I') pairs (no objects)", "array", False),
    ]

    print(f"{'Варіант':<32} | {'Пік RAM (build)':<16} | {'Розмір файлу':<14} | {'Час load':<10}")
    print("-" * 80)

    tmp_path = Path("_bench_tmp.bin")
    try:
        for label, var, pos_flag in variants:
            tracemalloc.start()
            idx = build_index(root, with_positions=pos_flag, variant=var, limit=limit)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            save_index(idx, tmp_path, fmt="pickle")
            file_mb = tmp_path.stat().st_size / (1024 * 1024)

            t0 = time.perf_counter()
            _ = load_index(tmp_path, fmt="pickle")
            load_ms = (time.perf_counter() - t0) * 1000

            peak_mb = peak_bytes / (1024 * 1024)
            print(f"{label:<32} | {peak_mb:>12.2f} MB | {file_mb:>10.2f} MB | {load_ms:>7.1f} ms")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу findex")
    parser.add_argument("corpus_dir", type=Path, help="Шлях до директорії корпусу")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Файл збереження індексу")
    parser.add_argument("--format", choices=["pickle", "binary"], default="pickle", help="Формат серіалізації")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів у постінгах")
    parser.add_argument("--memory-study", action="store_true", help="Запустити порівняння 3 варіантів пам'яті")
    args = parser.parse_args()

    if args.memory_study:
        run_memory_study(args.corpus_dir)
        return

    from findex.store import save_index

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(args.corpus_dir, with_positions=args.positions, variant="slots")
    build_time = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    save_index(index, args.out, fmt=args.format)
    save_time = time.perf_counter() - t1

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Документів проіндексовано: {len(index.doc_meta)}")
    print(f"Унікальних термінів: {len(index.postings)}")
    print(f"Час побудови: {build_time:.3f} с | Пік пам'яті: {peak_bytes / (1024 * 1024):.2f} МБ")
    print(f"Збережено у '{args.out}' ({args.format}): {size_mb:.2f} МБ за {save_time:.3f} с")


if __name__ == "__main__":
    main()