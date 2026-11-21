import tempfile
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.corpus import iter_files
from findex.index import DocMeta, Index, Posting, build_index, open_index
from findex.search import boolean_and, merge_and, merge_not, merge_or
from findex.stats import analyze_corpus
from findex.store import load_index, save_index


def test_iter_files_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_files(tmp_path / "missing"))

    f = tmp_path / "single.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        list(iter_files(f))


def test_index_mapping_protocol(sample_index: Index) -> None:
    assert len(sample_index) > 0
    assert "київ" in sample_index
    assert "неіснуюче_слово" not in sample_index
    assert sample_index.num_docs == 5
    assert sample_index.df("столиця") == 3
    assert sample_index.df("відсутнє") == 0
    assert sample_index.avg_doc_length > 0
    assert list(sample_index) == list(sample_index.postings.keys())
    assert repr(sample_index) == f"Index(terms={len(sample_index):_}, docs=5)"


@pytest.mark.parametrize("fmt", ["pickle", "binary"])
def test_save_load_roundtrip_and_context_manager(
    sample_index: Index,
    tmp_path: Path,
    fmt: Literal["pickle", "binary"],
) -> None:
    out = tmp_path / f"index_{fmt}.bin"
    save_index(sample_index, out, fmt=fmt)

    with open_index(out, fmt=fmt) as loaded:
        assert not loaded.is_closed
        assert loaded.num_docs == sample_index.num_docs
        assert loaded.postings == sample_index.postings
        assert loaded.doc_lengths == sample_index.doc_lengths
        assert loaded.doc_meta == sample_index.doc_meta
    assert loaded.is_closed


def test_load_index_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "absent.bin")


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([], [1, 2], []),
        ([1, 3, 5], [2, 3, 5, 7], [3, 5]),
        ([0, 1, 2], [0, 1, 2], [0, 1, 2]),
    ],
)
def test_merge_and_cases(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_and(a, b) == expected


def test_merge_or_and_not() -> None:
    assert merge_or([1, 3], [2, 3, 4]) == [1, 2, 3, 4]
    assert merge_not([0, 1, 2, 3, 4], [1, 3]) == [0, 2, 4]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_and_engines(sample_index: Index, engine: Literal["merge", "set"]) -> None:
    assert boolean_and(sample_index, ["столиця", "україни"], engine=engine) == [1, 2, 4]
    assert boolean_and(sample_index, ["київ", "львів"], engine=engine) == []
    assert boolean_and(sample_index, [], engine=engine) == []


def test_analyze_corpus(sample_corpus_dir: Path) -> None:
    st = analyze_corpus(sample_corpus_dir, top_n=3)
    assert st.num_docs == 5
    assert st.vocab_size == len(build_index(sample_corpus_dir))
    assert len(st.top_terms) == 3


@given(
    st.lists(st.integers(min_value=0, max_value=200), unique=True),
    st.lists(st.integers(min_value=0, max_value=200), unique=True),
)
def test_hypothesis_merge_matches_set_operations(raw_a: list[int], raw_b: list[int]) -> None:
    a = sorted(raw_a)
    b = sorted(raw_b)
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(
    st.dictionaries(
        keys=st.text(alphabet="абвгдежзиклмнопрстуфхцчшщьюя", min_size=1, max_size=8),
        values=st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=6, unique=True),
        min_size=1,
        max_size=10,
    )
)
def test_hypothesis_binary_roundtrip_and_sorted_postings(raw_map: dict[str, list[int]]) -> None:
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    for term, ids in raw_map.items():
        sorted_ids = sorted(ids)
        postings[term] = [Posting(doc_id=d, tf=1, positions=(0,)) for d in sorted_ids]
        for d in sorted_ids:
            doc_lengths[d] = doc_lengths.get(d, 0) + 1

    doc_meta = {d: DocMeta(doc_id=d, path=f"{d}.txt", title=f"Doc {d}") for d in doc_lengths}
    ix = Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta, with_positions=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "roundtrip.bin"
        save_index(ix, file_path, fmt="binary")
        restored = load_index(file_path, fmt="binary")

    assert restored.postings == ix.postings
    for plist in restored.postings.values():
        ids_list = [p.doc_id for p in plist]
        assert ids_list == sorted(ids_list)
