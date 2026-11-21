import logging
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_files
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusStats:
    num_docs: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]
    elapsed_sec: float
    peak_mem_mb: float


def compute_stats(files: Iterable[Path], top_n: int = 20) -> CorpusStats:
    """Обчислює статистику корпусу за один прохід із вимірюванням часу та пам'яті."""
    tracemalloc.start()
    t0 = time.perf_counter()

    num_docs = 0
    total_tokens = 0
    term_counts: Counter[str] = Counter()

    for path in files:
        num_docs += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in tokenize(text):
            total_tokens += 1
            term_counts[token] += 1

    elapsed = time.perf_counter() - t0
    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log.info(
        "Оброблено %d документів (%d токенів, словник %d) за %.3f с",
        num_docs,
        total_tokens,
        len(term_counts),
        elapsed,
    )
    return CorpusStats(
        num_docs=num_docs,
        total_tokens=total_tokens,
        vocab_size=len(term_counts),
        top_terms=term_counts.most_common(top_n),
        elapsed_sec=elapsed,
        peak_mem_mb=peak_mem / (1024 * 1024),
    )


def analyze_corpus(root: Path, limit: int | None = None, top_n: int = 20) -> CorpusStats:
    """Збирає статистику для директорії корпусу."""
    return compute_stats(iter_files(root, limit=limit), top_n=top_n)
