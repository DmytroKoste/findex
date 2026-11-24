import json
import math
import re
from pathlib import Path

from markupsafe import Markup, escape

from findex.index import Index
from findex.query import parse
from findex.rank import ScorerName, build_snippet, make_scorer
from findex.rank import search as core_search

_DOC_TEXT_CACHE: dict[tuple[int, int], str] = {}
_SNIPPET_MARK_RE = re.compile(r"\[([^\[\]]+)\]")


def load_doc_text(index: Index, doc_id: int) -> str:
    key = (id(index), doc_id)
    if key in _DOC_TEXT_CACHE:
        return _DOC_TEXT_CACHE[key]

    meta = index.doc_meta[doc_id]
    p = Path(meta.path)
    text = ""
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="ignore")
    else:
        for candidate in (Path("data/py_docs.jsonl"), Path("data/bench_c5.jsonl")):
            if candidate.is_file():
                with candidate.open(encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("url") == meta.path or rec.get("title") == meta.title:
                            text = str(rec.get("text", ""))
                            break
            if text:
                break

    if not text:
        terms = [term for term, plist in index.postings.items() if any(post.doc_id == doc_id for post in plist)]
        text = f"{meta.title}. Ключові терміни документа: " + ", ".join(terms[:60])

    _DOC_TEXT_CACHE[key] = text
    return text


def highlight_snippet_html(snippet: str) -> Markup:
    escaped = str(escape(snippet))
    highlighted = _SNIPPET_MARK_RE.sub(r"<mark>\1</mark>", escaped)
    return Markup(highlighted)


def run_paginated_search(
    index: Index,
    query: str,
    scorer_name: ScorerName,
    k: int,
    page: int,
) -> tuple[int, list[tuple[int, str, float, str]]]:
    try:
        tree = parse(query)
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    matched_docs = tree.evaluate(index)
    total = len(matched_docs)
    if total == 0:
        return 0, []

    needed = page * k
    scorer = make_scorer(scorer_name)
    raw_results = core_search(index, query, scorer=scorer, k=needed, with_snippets=False)
    offset = (page - 1) * k
    page_slice = raw_results[offset : offset + k]
    query_terms = tree.terms()

    items: list[tuple[int, str, float, str]] = []
    for res in page_slice:
        doc_text = load_doc_text(index, res.doc_id)
        snippet = build_snippet(doc_text, query_terms, radius=90)
        items.append((res.doc_id, res.title, round(res.score, 4), snippet))

    return total, items


def total_pages(total: int, k: int) -> int:
    return max(1, math.ceil(total / k))
