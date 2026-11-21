from pathlib import Path

from findex.index import Index, build_index, open_index
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.rank import BM25, TfIdf, search
from findex.store import save_index


def test_index_dunder_and_context_manager(tmp_path: Path) -> None:
    (tmp_path / "0.txt").write_text("Київ столиця України", encoding="utf-8")
    (tmp_path / "1.txt").write_text("Львів культурна столиця України", encoding="utf-8")

    ix = build_index(tmp_path, with_positions=True)
    assert isinstance(ix, Index)
    assert len(ix) == 5
    assert "київ" in ix
    assert "відсутнє" not in ix
    assert ix.num_docs == 2
    assert ix.avg_doc_length == 3.5
    assert ix.df("столиця") == 2
    assert list(ix) == list(ix.postings.keys())
    assert repr(ix) == "Index(terms=5, docs=2)"

    out = tmp_path / "ix.pkl"
    save_index(ix, out)
    with open_index(out) as opened:
        assert opened.num_docs == 2
        assert not opened._closed
    assert opened._closed


def test_query_parser_trees_and_operators() -> None:
    tree = parse("a OR b c")
    assert tree == Or(Term("a"), And(Term("b"), Term("c")))

    complex_tree = parse('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert complex_tree == expected

    manual = (Term("a") | (Term("b") & Term("c"))) & ~Term("d")
    assert parse("(a OR b c) NOT d") == manual


def test_ranking_phrase_and_snippets(tmp_path: Path) -> None:
    (tmp_path / "0_short.txt").write_text("Рідкісний термін квантування у тексті.", encoding="utf-8")
    (tmp_path / "1_long.txt").write_text(
        "Дуже довгий документ про різні загальні речі " + ("слово " * 40) + "термін.",
        encoding="utf-8",
    )
    (tmp_path / "2_phrase.txt").write_text("Тут є точна фраза квантування у тексті.", encoding="utf-8")

    ix = build_index(tmp_path, with_positions=True)
    assert search.__name__ == "search"

    phrase_res = search(ix, '"квантування у"', scorer=BM25(), k=5)
    assert [r.doc_id for r in phrase_res] == [0, 2]
    assert "[квантування]" in phrase_res[0].snippet

    tfidf_res = search(ix, "термін", scorer=TfIdf(), k=5)
    bm25_res = search(ix, "термін", scorer=BM25(), k=5)
    assert bm25_res[0].doc_id == 0
    assert len(tfidf_res) == 2