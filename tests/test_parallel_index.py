from pathlib import Path
from unittest.mock import patch

import pytest

from findex.corpus import iter_files
from findex.index import build_index, build_partial, merge


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    docs = [
        "Київ — столиця та найбільше місто України. Київ розташований на річці Дніпро.",
        "Львів — культурна столиця України, заснована королем Данилом Галицьким.",
        "Харків — великий науковий, промисловий та студентський центр на сході України.",
        "Одеса — головний морський порт України на узбережжі Чорного моря.",
        "Дніпро — важливий індустріальний та космічний центр у центральній Україні.",
        "Полтава славиться своєю історією, літературними традиціями та галушками.",
    ]
    for i, text in enumerate(docs):
        (tmp_path / f"doc_{i:02d}.txt").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("with_positions", [False, True])
def test_partial_merge_matches_serial(sample_corpus: Path, with_positions: bool) -> None:
    indexed_paths = list(enumerate(iter_files(sample_corpus)))
    serial_ix = build_index(sample_corpus, with_positions=with_positions, executor="serial", workers=1)
    manual_ix = merge([build_partial(indexed_paths, with_positions=with_positions)])

    assert manual_ix.postings == serial_ix.postings
    assert manual_ix.doc_lengths == serial_ix.doc_lengths
    assert manual_ix.doc_meta == serial_ix.doc_meta
    assert manual_ix.with_positions == serial_ix.with_positions


@pytest.mark.parametrize("with_positions", [False, True])
@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_produce_identical_index(
    sample_corpus: Path,
    with_positions: bool,
    executor: str,
    workers: int,
) -> None:
    baseline = build_index(sample_corpus, with_positions=with_positions, executor="serial", workers=1)
    actual = build_index(
        sample_corpus,
        with_positions=with_positions,
        executor=executor,  # type: ignore[arg-type]
        workers=workers,
    )

    assert actual.postings == baseline.postings
    assert actual.doc_lengths == baseline.doc_lengths
    assert actual.doc_meta == baseline.doc_meta
    assert actual.with_positions == baseline.with_positions


def test_worker_exception_propagates(sample_corpus: Path) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Помилка всередині воркера")

    with patch("findex.index.build_partial", side_effect=_boom):
        with pytest.raises(RuntimeError, match="Помилка всередині воркера"):
            build_index(sample_corpus, executor="threads", workers=2)