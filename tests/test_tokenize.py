from findex.tokenize import tokenize


def test_tokenize_basic() -> None:
    text = "Привіт, Світ! Python 3.12 — це круто."
    assert list(tokenize(text)) == ["привіт", "світ", "python", "3", "12", "це", "круто"]


def test_tokenize_apostrophe() -> None:
    text = "П'ять комп'ютерів та м'яч."
    assert list(tokenize(text)) == ["п'ять", "комп'ютерів", "та", "м'яч"]


def test_tokenize_empty() -> None:
    assert list(tokenize("   ... --- !!!   ")) == []