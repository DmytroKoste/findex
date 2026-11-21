import re
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize

_QUERY_TOKEN_RE = re.compile(r'"[^"]+"|\(|\)|[^\s()]+')


class QueryNode:
    """Базовий вузол дерева запиту з перевантаженням &, |, ~."""

    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> set[str]:
        return set()


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}

    def terms(self) -> set[str]:
        return {self.value}


def _merge_adjacent_positions(pos_a: tuple[int, ...], pos_b: tuple[int, ...]) -> tuple[int, ...]:
    """Повертає позиції pos_b, для яких у pos_a існує pos_b - 1 (двовказівниковий merge)."""
    res: list[int] = []
    i = j = 0
    len_a, len_b = len(pos_a), len(pos_b)
    while i < len_a and j < len_b:
        target = pos_a[i] + 1
        if pos_b[j] == target:
            res.append(pos_b[j])
            i += 1
            j += 1
        elif pos_b[j] < target:
            j += 1
        else:
            i += 1
    return tuple(res)


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.words:
            return set()
        if any(w not in index for w in self.words):
            return set()

        first_postings = {p.doc_id: p.positions for p in index[self.words[0]]}
        candidate_docs = dict(first_postings)

        for word in self.words[1:]:
            next_postings = {p.doc_id: p.positions for p in index[word]}
            updated: dict[int, tuple[int, ...]] = {}
            for doc_id, prev_pos in candidate_docs.items():
                if doc_id in next_postings:
                    matched_pos = _merge_adjacent_positions(prev_pos, next_postings[doc_id])
                    if matched_pos:
                        updated[doc_id] = matched_pos
            candidate_docs = updated
            if not candidate_docs:
                break

        return set(candidate_docs.keys())

    def terms(self) -> set[str]:
        return set(self.words)


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_meta.keys()) - self.operand.evaluate(index)

    def terms(self) -> set[str]:
        return set()


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while (tok := self.peek()) is not None and tok not in ("OR", ")"):
            if tok == "AND":
                self.consume()
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> QueryNode:
        if self.peek() == "NOT":
            self.consume()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise ValueError("Несподіваний кінець запиту")
        if tok == "(":
            self.consume()
            node = self.parse_or()
            if self.peek() != ")":
                raise ValueError("Очікувалась закриваюча дужка ')'")
            self.consume()
            return node
        self.consume()
        if tok.startswith('"') and tok.endswith('"'):
            words = tuple(tokenize(tok[1:-1]))
            return Phrase(words)
        norm = tuple(tokenize(tok))
        if len(norm) == 1:
            return Term(norm[0])
        if len(norm) > 1:
            node: QueryNode = Term(norm[0])
            for w in norm[1:]:
                node = And(node, Term(w))
            return node
        return Term(tok.lower())


def parse(query: str) -> QueryNode:
    """Розбирає рядок запиту методом рекурсивного спуску у дерево вузлів."""
    raw_tokens = _QUERY_TOKEN_RE.findall(query.strip())
    if not raw_tokens:
        raise ValueError("Порожній пошуковий запит")
    parser = _Parser(raw_tokens)
    tree = parser.parse_or()
    if parser.peek() is not None:
        raise ValueError(f"Зайвий токен у запиті: {parser.peek()}")
    return tree