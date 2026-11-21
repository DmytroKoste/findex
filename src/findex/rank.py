import functools
import heapq
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from findex.index import Index, Posting, timed
from findex.query import parse
from findex.tokenize import TOKEN_RE

log = logging.getLogger(__name__)

ScorerName = Literal["bm25", "tfidf"]


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


def make_scorer(name: ScorerName) -> Scorer:
    """Повертає екземпляр функції ранжування за ім'ям."""
    return BM25() if name == "bm25" else TfIdf()


def build_snippet(text: str, terms: set[str], radius: int = 80) -> str:
    """Витягує вікно ±radius символів навколо першого знайденого терміна і підсвічує збіги."""
    if not text:
        return ""
    lowered = text.casefold()
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
        return f"[{word}]" if word.casefold() in terms else word

    highlighted = TOKEN_RE.sub(_highlight, window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"


_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=256)
def _cached_ranked_ids(
    index_id: int,
    query: str,
    scorer_name: ScorerName,
    k: int,
) -> tuple[tuple[int, float], ...]:
    index_ref = _INDEX_REGISTRY[index_id]
    scorer = make_scorer(scorer_name)
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
    if k <= 0:
        return []
    _INDEX_REGISTRY[id(index)] = index
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    scorer_name: ScorerName = "tfidf" if isinstance(active_scorer, TfIdf) else "bm25"

    top_pairs = _cached_ranked_ids(id(index), query, scorer_name, k)
    log.debug("Стан кешу запитів: %s", _cached_ranked_ids.cache_info())
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
