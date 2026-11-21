import argparse
import functools
import heapq
import math
import re
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Protocol

from findex.index import Index, Posting, open_index, timed
from findex.query import parse
from findex.tokenize import TOKEN_RE


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log(index.num_docs / df)
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0:
            return 0.0
        n = index.num_docs
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        dl = index.doc_length(posting.doc_id)
        avgdl = index.avg_doc_length or 1.0
        tf = posting.tf
        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
        return idf * ((tf * (self.k1 + 1.0)) / denom)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    path: str
    snippet: str = ""


def build_snippet(text: str, terms: set[str], radius: int = 80) -> str:
    """Витягує вікно ±radius символів навколо першого знайденого терміна і підсвічує збіги."""
    if not text:
        return ""
    lowered = text.lower()
    best_pos = -1
    for term in terms:
        pos = lowered.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        return text[: radius * 2].replace("\n", " ").strip()

    start = max(0, best_pos - radius)
    end = min(len(text), best_pos + radius)
    window = text[start:end].replace("\n", " ").strip()

    def _highlight(match: re.Match[str]) -> str:
        word = match.group(0)
        return f"[{word}]" if word.lower() in terms else word

    highlighted = TOKEN_RE.sub(_highlight, window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"


_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=256)
def _cached_ranked_ids(
    index_id: int,
    query: str,
    scorer: Scorer,
    k: int,
) -> tuple[tuple[int, float], ...]:
    index_ref = _INDEX_REGISTRY[index_id]
    tree = parse(query)
    matched_docs = tree.evaluate(index_ref)
    if not matched_docs:
        return ()

    query_terms = tree.terms()
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched_docs}

    for term in query_terms:
        if term not in index_ref:
            continue
        for posting in index_ref[term]:
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer.score(term, posting, index_ref)

    top_items = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
    return tuple(top_items)


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    """Ранжований пошук із кешуванням та генерацією сніпетів."""
    _INDEX_REGISTRY[id(index)] = index
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    top_pairs = _cached_ranked_ids(id(index), query, active_scorer, k)
    query_terms = parse(query).terms()

    results: list[SearchResult] = []
    for doc_id, score in top_pairs:
        meta = index.doc_meta[doc_id]
        snippet = ""
        if with_snippets:
            p = Path(meta.path)
            if p.exists():
                raw_text = p.read_text(encoding="utf-8", errors="ignore")
                snippet = build_snippet(raw_text, query_terms)
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                path=meta.path,
                snippet=snippet,
            )
        )
    return results


def run_sanity_checks(index: Index) -> None:
    """Три sanity-check перевірки рейтингу BM25 на корпусі (Етап M2)."""
    bm25 = BM25()
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]))
    rare_term, rare_postings = next((t, pl) for t, pl in sorted_terms if len(pl) == 1)
    common_term, common_postings = sorted_terms[-1]

    print("=== Sanity Check 1: Рідкісний термін проти частого ===")
    rare_score = bm25.score(rare_term, rare_postings[0], index)
    common_score = bm25.score(common_term, common_postings[0], index)
    print(
        f"Рідкісний '{rare_term}' (df=1): score={rare_score:.4f} > "
        f"Частий '{common_term}' (df={len(common_postings)}): score={common_score:.4f}"
    )

    print("\n=== Sanity Check 2: Насичення TF у BM25 (k1=1.5) ===")
    sample_doc = rare_postings[0].doc_id
    s1 = bm25.score(rare_term, Posting(doc_id=sample_doc, tf=1), index)
    s2 = bm25.score(rare_term, Posting(doc_id=sample_doc, tf=2), index)
    s20 = bm25.score(rare_term, Posting(doc_id=sample_doc, tf=20), index)
    s21 = bm25.score(rare_term, Posting(doc_id=sample_doc, tf=21), index)
    print(f"tf=1 -> {s1:.4f} | tf=2 -> {s2:.4f} (+{s2 - s1:.4f})")
    print(f"tf=20 -> {s20:.4f} | tf=21 -> {s21:.4f} (+{s21 - s20:.6f} — насичення)")

    print("\n=== Sanity Check 3: Короткий документ проти довгого (tf=1, b=0.75) ===")
    by_len = sorted(index.doc_lengths.items(), key=lambda kv: kv[1])
    short_doc, short_len = next((d, ln) for d, ln in by_len if ln > 5)
    long_doc, long_len = by_len[-1]
    s_short = bm25.score(common_term, Posting(doc_id=short_doc, tf=1), index)
    s_long = bm25.score(common_term, Posting(doc_id=long_doc, tf=1), index)
    print(
        f"Короткий doc={short_doc} (len={short_len}): score={s_short:.4f} > "
        f"Довгий doc={long_doc} (len={long_len}): score={s_long:.4f}"
    )


EVAL_BENCHMARK: list[tuple[str, set[int]]] = [
    ('"посольство україни"', {70}),
    ('"ротарі інтернешнл"', {39}),
    ('"вулиця вернадського"', {194}),
    ("голубець орест", {723}),
    ("веркалець михайло", {405}),
    ("полоз валерія", {382}),
    ("київ україна", {70, 194, 405, 672}),
    ("львів україна", {723}),
    ("художник україна", {672, 723}),
    ("педагог журналіст", {405}),
]


def run_eval(index: Index) -> None:
    """Обчислює precision@5 для TF-IDF та BM25 на 10 розмічених запитах."""
    tfidf = TfIdf()
    bm25 = BM25()
    p5_tfidf_list: list[float] = []
    p5_bm25_list: list[float] = []

    print(f"{'Запит':<26} | {'Релевантні':<16} | {'P@5 (TF-IDF)':<12} | {'P@5 (BM25)':<12}")
    print("-" * 74)
    for q, relevant in EVAL_BENCHMARK:
        res_tfidf = [r.doc_id for r in search(index, q, scorer=tfidf, k=5, with_snippets=False)]
        res_bm25 = [r.doc_id for r in search(index, q, scorer=bm25, k=5, with_snippets=False)]

        p5_t = sum(1 for d in res_tfidf if d in relevant) / 5.0
        p5_b = sum(1 for d in res_bm25 if d in relevant) / 5.0
        p5_tfidf_list.append(p5_t)
        p5_bm25_list.append(p5_b)

        rel_str = ",".join(str(d) for d in sorted(relevant))
        print(f"{q:<26} | {rel_str:<16} | {p5_t:<12.2f} | {p5_b:<12.2f}")

    mean_t = sum(p5_tfidf_list) / len(p5_tfidf_list)
    mean_b = sum(p5_bm25_list) / len(p5_bm25_list)
    print("-" * 74)
    print(f"{'Середній P@5':<26} | {'':<16} | {mean_t:<12.3f} | {mean_b:<12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ранжований пошук findex (TF-IDF / BM25)")
    parser.add_argument("index_path", type=Path, help="Шлях до файлу індексу")
    parser.add_argument("query", nargs="?", default="", help="Пошуковий запит")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Функція ранжування")
    parser.add_argument("--top", type=int, default=5, help="Кількість результатів (top-k)")
    parser.add_argument("--sanity", action="store_true", help="Запустити 3 sanity-check перевірки рейтингу")
    parser.add_argument("--repeat", action="store_true", help="Виконати запит двічі для демонстрації lru_cache")
    parser.add_argument("--eval", action="store_true", help="Порахувати precision@5 на 10 запитах")
    args = parser.parse_args()

    scorer: Scorer = BM25() if args.scorer == "bm25" else TfIdf()

    with open_index(args.index_path) as ix:
        if args.sanity:
            run_sanity_checks(ix)
            return

        if args.eval:
            run_eval(ix)
            return

        results = search(ix, args.query, scorer=scorer, k=args.top)
        if args.repeat:
            results = search(ix, args.query, scorer=scorer, k=args.top)
            print(f"[cache] {_cached_ranked_ids.cache_info()}")

        print(f"Топ-{len(results)} результатів ({args.scorer.upper()}):")
        for res in sorted(results, key=attrgetter("score"), reverse=True):
            print(f"[{res.doc_id:>4}] score={res.score:.4f} | {res.title}")
            if res.snippet:
                print(f"       {res.snippet}")


if __name__ == "__main__":
    main()