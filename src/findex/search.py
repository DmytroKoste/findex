import argparse
import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex
from findex.store import load_index
from findex.tokenize import tokenize


def intersect_merge(a: list[int], b: list[int]) -> list[int]:
    """Двовказівниковий перетин (AND) двох відсортованих списків doc_id."""
    res: list[int] = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return res


def union_merge(a: list[int], b: list[int]) -> list[int]:
    """Двовказівникове об'єднання (OR) двох відсортованих списків doc_id."""
    res: list[int] = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    if i < len_a:
        res.extend(a[i:])
    if j < len_b:
        res.extend(b[j:])
    return res


def diff_merge(a: list[int], b: list[int]) -> list[int]:
    """Двовказівникова різниця (a AND NOT b) для відсортованих списків doc_id."""
    res: list[int] = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            j += 1
    if i < len_a:
        res.extend(a[i:])
    return res


def _get_doc_ids(index: InvertedIndex, term: str) -> list[int]:
    normalized = next(tokenize(term), "")
    if not normalized or normalized not in index.postings:
        return []
    return [p.doc_id for p in index.postings[normalized]]


def boolean_search(index: InvertedIndex, query: str, engine: str = "merge") -> list[int]:
    """Виконує булевий запит (неявний AND між словами, підтримка OR та NOT)."""
    raw_tokens = query.strip().split()
    if not raw_tokens:
        return []

    all_docs = sorted(index.doc_meta.keys())
    current: list[int] | None = None
    op = "AND"
    negate = False

    for tok in raw_tokens:
        upper = tok.upper()
        if upper in ("AND", "OR"):
            op = upper
            continue
        if upper == "NOT":
            negate = True
            continue

        term_docs = _get_doc_ids(index, tok)
        if negate:
            if engine == "set":
                term_docs = sorted(set(all_docs) - set(term_docs))
            else:
                term_docs = diff_merge(all_docs, term_docs)
            negate = False

        if current is None:
            current = term_docs
        elif engine == "set":
            if op == "AND":
                current = sorted(set(current) & set(term_docs))
            else:
                current = sorted(set(current) | set(term_docs))
        else:
            if op == "AND":
                current = intersect_merge(current, term_docs)
            else:
                current = union_merge(current, term_docs)

        op = "AND"

    return current or []


def benchmark_engines(index: InvertedIndex, iterations: int = 5000) -> None:
    """Порівнює merge vs set на 2 найчастіших і 2 найрідкісніших термінах."""
    sorted_terms = sorted(index.postings.items(), key=lambda item: len(item[1]), reverse=True)
    common_t1, common_t2 = sorted_terms[0][0], sorted_terms[1][0]
    rare_t1, rare_t2 = sorted_terms[-1][0], sorted_terms[-2][0]

    pairs = [
        (f"Найчастіші ({common_t1} AND {common_t2})", common_t1, common_t2),
        (f"Найрідкісніші ({rare_t1} AND {rare_t2})", rare_t1, rare_t2),
    ]

    print(f"\nБенчмарк пошуку ({iterations} ітерацій):")
    print(f"{'Запит':<42} | {'df1 / df2':<13} | {'merge (мс)':<12} | {'set (мс)':<12}")
    print("-" * 88)

    for label, t1, t2 in pairs:
        l1 = [p.doc_id for p in index.postings[t1]]
        l2 = [p.doc_id for p in index.postings[t2]]

        t0 = time.perf_counter()
        for _ in range(iterations):
            intersect_merge(l1, l2)
        merge_ms = (time.perf_counter() - t0) * 1000

        t1_start = time.perf_counter()
        for _ in range(iterations):
            sorted(set(l1) & set(l2))
        set_ms = (time.perf_counter() - t1_start) * 1000

        df_info = f"{len(l1)} / {len(l2)}"
        print(f"{label:<42} | {df_info:<13} | {merge_ms:>9.2f} ms | {set_ms:>9.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Булевий пошук по інвертованому індексу findex")
    parser.add_argument("index_path", type=Path, help="Шлях до файлу індексу")
    parser.add_argument("query", nargs="?", default="", help="Пошуковий запит (напр. 'україна київ')")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Алгоритм об'єднання")
    parser.add_argument("--top", type=int, default=10, help="Скільки результатів показати")
    parser.add_argument("--benchmark", action="store_true", help="Запустити бенчмарк merge vs set і форматів")
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load_index(args.index_path)
    load_time = time.perf_counter() - t0

    if args.benchmark:
        from findex.store import benchmark_formats

        benchmark_formats(index)
        benchmark_engines(index)
        tracemalloc.stop()
        return

    t1 = time.perf_counter()
    doc_ids = boolean_search(index, args.query, engine=args.engine)
    search_time = time.perf_counter() - t1
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Знайдено документів: {len(doc_ids)} (рушій: {args.engine})")
    print(f"Час load: {load_time * 1000:.2f} мс | Час пошуку: {search_time * 1000:.3f} мс | Пік RAM: {peak_bytes / (1024 * 1024):.2f} МБ")
    print("-" * 60)
    for doc_id in doc_ids[: args.top]:
        meta = index.doc_meta[doc_id]
        print(f"[{doc_id:>4}] {meta.title} ({meta.path})")


if __name__ == "__main__":
    main()