from pathlib import Path

from findex.index import DocMeta, Posting, build_index
from findex.search import boolean_search
from findex.store import load_index, save_index


def test_posting_and_docmeta_hashable() -> None:
    p1 = Posting(doc_id=1, tf=3)
    p2 = Posting(doc_id=1, tf=3)
    assert p1 == p2
    assert len({p1, p2}) == 1

    m1 = DocMeta(doc_id=0, path="a.txt", title="Doc A")
    assert m1 in {m1}


def test_build_and_boolean_search(tmp_path: Path) -> None:
    (tmp_path / "0.txt").write_text("Київ столиця України", encoding="utf-8")
    (tmp_path / "1.txt").write_text("Львів місто в Україні", encoding="utf-8")
    (tmp_path / "2.txt").write_text("Лондон столиця Британії", encoding="utf-8")

    idx = build_index(tmp_path, with_positions=True)
    assert len(idx.doc_meta) == 3
    assert idx.postings["столиця"][0].positions == (1,)

    for engine in ("merge", "set"):
        assert boolean_search(idx, "столиця україни", engine=engine) == [0]
        assert boolean_search(idx, "київ OR львів", engine=engine) == [0, 1]
        assert boolean_search(idx, "столиця NOT київ", engine=engine) == [2]


def test_store_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("тест індексу", encoding="utf-8")
    idx = build_index(tmp_path)

    for fmt, name in (("pickle", "idx.pkl"), ("binary", "idx.bin")):
        out = tmp_path / name
        save_index(idx, out, fmt=fmt)
        loaded = load_index(out)
        assert loaded.doc_lengths == idx.doc_lengths
        assert boolean_search(loaded, "тест") == [0]