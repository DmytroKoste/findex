import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index, Posting
from findex.query import And, Not, Or, Phrase, Term, parse
from findex.rank import BM25, TfIdf, search

runner = CliRunner()


@pytest.mark.parametrize(
    ("query_str", "expected"),
    [
        ("python", Term("python")),
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("(a OR b) NOT c", And(Or(Term("a"), Term("b")), Not(Term("c")))),
        ('"пошуковий рушій"', Phrase(("пошуковий", "рушій"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(
                    And(Term("python"), Or(Term("async"), Term("await"))),
                    Not(Term("java")),
                ),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_parser_tree_equality(query_str: str, expected: object) -> None:
    assert parse(query_str) == expected


def test_parser_operator_overloading() -> None:
    manual = (Term("a") | (Term("b") & Term("c"))) & ~Term("d")
    assert parse("(a OR b c) NOT d") == manual


@pytest.mark.parametrize("bad_query", ["", "   ", "(a OR b", "a ) b", '""'])
def test_parser_invalid_queries(bad_query: str) -> None:
    with pytest.raises(ValueError):
        parse(bad_query)


def test_bm25_three_sanity_checks(sample_index: Index) -> None:
    bm25 = BM25()

    # 1. Рідкісний термін вище за частий
    rare_score = bm25.score("квантування", sample_index["квантування"][0], sample_index)
    common_score = bm25.score("столиця", sample_index["столиця"][0], sample_index)
    assert rare_score > common_score

    # 2. Насичення TF: 20-те повторення додає майже нічого порівняно з 1 -> 2
    s1 = bm25.score("квантування", Posting(doc_id=3, tf=1), sample_index)
    s2 = bm25.score("квантування", Posting(doc_id=3, tf=2), sample_index)
    s20 = bm25.score("квантування", Posting(doc_id=3, tf=20), sample_index)
    s21 = bm25.score("квантування", Posting(doc_id=3, tf=21), sample_index)
    assert (s2 - s1) > 10 * (s21 - s20)

    # 3. Короткий документ з одним входженням вище за довгий з одним входженням
    short_score = bm25.score("столиця", Posting(doc_id=1, tf=1), sample_index)
    long_score = bm25.score("столиця", Posting(doc_id=4, tf=1), sample_index)
    assert short_score > long_score


def test_tfidf_and_phrase_snippets(sample_index: Index) -> None:
    assert search.__name__ == "search"

    phrase_res = search(sample_index, '"столиця україни"', scorer=BM25(), k=5)
    assert {r.doc_id for r in phrase_res} == {1, 2, 4}
    assert "[столиця]" in phrase_res[0].snippet

    tfidf_res = search(sample_index, "столиця", scorer=TfIdf(), k=2)
    assert len(tfidf_res) == 2
    assert tfidf_res[0].score >= tfidf_res[1].score


def test_cli_index_search_stats_and_json(sample_corpus_dir: Path, tmp_path: Path) -> None:
    ix_file = tmp_path / "cli_ix.bin"

    res_idx = runner.invoke(
        app,
        ["-v", "index", str(sample_corpus_dir), "--out", str(ix_file), "--positions", "--format", "binary"],
    )
    assert res_idx.exit_code == 0
    assert ix_file.exists()

    res_search = runner.invoke(app, ["search", str(ix_file), '"столиця україни"', "--k", "2", "--json"])
    assert res_search.exit_code == 0
    lines = [json.loads(line) for line in res_search.stdout.strip().splitlines()]
    assert len(lines) == 2
    assert {"doc_id", "score", "title", "path", "snippet"} <= set(lines[0].keys())

    res_table = runner.invoke(app, ["search", str(ix_file), "квантування"])
    assert res_table.exit_code == 0
    assert "квантування" in res_table.stdout

    res_stats_ix = runner.invoke(app, ["stats", str(ix_file)])
    assert res_stats_ix.exit_code == 0

    res_stats_dir = runner.invoke(app, ["stats", str(sample_corpus_dir), "--top", "5"])
    assert res_stats_dir.exit_code == 0


def test_cli_expected_errors_exit_nonzero(saved_index_path: Path, tmp_path: Path) -> None:
    missing = runner.invoke(app, ["search", str(tmp_path / "no_such.bin"), "python"])
    assert missing.exit_code == 1

    bad_query = runner.invoke(app, ["search", str(saved_index_path), "(unclosed"])
    assert bad_query.exit_code == 1
