from collections.abc import Iterable, Sequence
from typing import Literal

from findex.index import Index, Posting

SearchEngine = Literal["merge", "set"]


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Двовказівникове об'єднання (перетин) двох відсортованих списків doc_id за O(|a| + |b|)."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        va = a[i]
        vb = b[j]
        if va == vb:
            result.append(va)
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Двовказівникове об'єднання (сума) двох відсортованих списків doc_id за O(|a| + |b|)."""
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        va = a[i]
        vb = b[j]
        if va == vb:
            result.append(va)
            i += 1
            j += 1
        elif va < vb:
            result.append(va)
            i += 1
        else:
            result.append(vb)
            j += 1
    if i < len_a:
        result.extend(a[i:])
    if j < len_b:
        result.extend(b[j:])
    return result


def merge_not(universe: Sequence[int], excluded: Sequence[int]) -> list[int]:
    """Різниця відсортованих списків doc_id за O(|universe| + |excluded|)."""
    result: list[int] = []
    i = 0
    j = 0
    len_u = len(universe)
    len_e = len(excluded)
    while i < len_u and j < len_e:
        vu = universe[i]
        ve = excluded[j]
        if vu == ve:
            i += 1
            j += 1
        elif vu < ve:
            result.append(vu)
            i += 1
        else:
            j += 1
    if i < len_u:
        result.extend(universe[i:])
    return result


def _doc_ids(postings: Sequence[Posting]) -> list[int]:
    return [p.doc_id for p in postings]


def boolean_and(index: Index, terms: Iterable[str], engine: SearchEngine = "merge") -> list[int]:
    """Булевий AND-пошук для довільної кількості термінів."""
    term_list = [t.casefold() for t in terms]
    if not term_list:
        return []

    posting_lists: list[list[int]] = []
    for term in term_list:
        if term not in index:
            return []
        posting_lists.append(_doc_ids(index[term]))

    posting_lists.sort(key=len)

    if engine == "set":
        acc = set(posting_lists[0])
        for plist in posting_lists[1:]:
            acc &= set(plist)
        return sorted(acc)

    res = posting_lists[0]
    for plist in posting_lists[1:]:
        res = merge_and(res, plist)
        if not res:
            break
    return res
