import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.tokenize import tokenize, tokenize_all


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("   \n\t  ", []),
        ("Hello, World!", ["hello", "world"]),
        ("Київ — столиця України!", ["київ", "столиця", "україни"]),
        ("м'ясо та ім'я, роз'їзд", ["м'ясо", "та", "ім'я", "роз'їзд"]),
        ("науково-дослідний інститут", ["науково-дослідний", "інститут"]),
        ("Python 3.12 & Rust 2026", ["python", "3", "12", "rust", "2026"]),
        ("İSTANBUL café naïve", ["i̇stanbul", "café", "naïve"]),
        ("...,,,---!!!???", []),
    ],
)
def test_tokenize_parametrized(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected


def test_tokenize_all_stream() -> None:
    stream = ["Перший рядок", "Другий рядок!"]
    assert list(tokenize_all(stream)) == ["перший", "рядок", "другий", "рядок"]


@given(st.text())
def test_hypothesis_no_empty_tokens(text: str) -> None:
    tokens = list(tokenize(text))
    assert all(tok != "" for tok in tokens)


@given(st.text())
def test_hypothesis_tokenize_idempotent(text: str) -> None:
    first = list(tokenize(text))
    second = list(tokenize(" ".join(first)))
    assert second == first
