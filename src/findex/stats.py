import argparse
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import read_documents
from findex.tokenize import tokenize


def compute_stats_lazy(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    """Потоковий (лінивий) підрахунок статистики без завантаження всього корпусу в RAM."""
    doc_count = 0
    total_tokens = 0
    vocab: Counter[str] = Counter()

    for _doc_id, text in read_documents(root, limit=limit):
        doc_count += 1
        for token in tokenize(text):
            total_tokens += 1
            vocab[token] += 1

    return doc_count, total_tokens, vocab


def compute_stats_eager(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    """Жадібний (eager) підрахунок: спершу вантажить усі тексти й токени у списки."""
    docs = list(read_documents(root, limit=limit))
    all_tokens = [token for _doc_id, text in docs for token in tokenize(text)]
    vocab = Counter(all_tokens)
    return len(docs), len(all_tokens), vocab


def run_benchmark(label: str, fn, root: Path, limit: int | None) -> None:
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count, total_tokens, vocab = fn(root, limit=limit)

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"=== {label} ===")
    print(f"Документів:        {doc_count}")
    print(f"Усього токенів:    {total_tokens}")
    print(f"Унікальних слів:   {len(vocab)}")
    print(f"Топ-10 слів:       {vocab.most_common(10)}")
    print(f"Час виконання:     {elapsed:.4f} с")
    print(f"Пікова пам'ять:    {peak_mem / (1024 * 1024):.2f} МБ\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Статистика текстового корпусу findex")
    parser.add_argument("corpus_dir", type=Path, help="Шлях до папки з текстами")
    parser.add_argument("--limit", type=int, default=None, help="Обмеження кількості файлів")
    parser.add_argument(
        "--mode",
        choices=["lazy", "eager", "compare"],
        default="compare",
        help="Режим обчислення: lazy, eager або compare",
    )
    args = parser.parse_args()

    if args.mode in ("lazy", "compare"):
        run_benchmark("LAZY (генератори)", compute_stats_lazy, args.corpus_dir, args.limit)
    if args.mode in ("eager", "compare"):
        run_benchmark("EAGER (списки в RAM)", compute_stats_eager, args.corpus_dir, args.limit)


if __name__ == "__main__":
    main()