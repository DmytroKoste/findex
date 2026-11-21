from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import save_index


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    docs: dict[str, str] = {
        "00_python.txt": "Python мова програмування для аналізу даних та пошукових рушіїв.",
        "01_kyiv.txt": "Київ столиця України та великий науковий центр.",
        "02_lviv.txt": "Львів культурна столиця України з давньою архітектурою.",
        "03_rare.txt": "Унікальний термін квантування зустрічається лише тут один раз.",
        "04_long.txt": (
            "Довгий оглядовий документ про різні технології та загальні поняття " + ("текст " * 35) + "столиця України."
        ),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    return build_index(sample_corpus_dir, with_positions=True, variant="slots")


@pytest.fixture
def saved_index_path(sample_index: Index, tmp_path: Path) -> Path:
    out = tmp_path / "test_index.bin"
    save_index(sample_index, out, fmt="binary")
    return out
