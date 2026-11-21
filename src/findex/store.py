import logging
import pickle
import struct
import sys
from pathlib import Path

from findex.index import DocMeta, Index, IndexFormat, Posting, timed

log = logging.getLogger(__name__)

MAGIC = b"FIDX0002"
HEADER_STRUCT = struct.Struct("<8sIII")
DOC_LEN_STRUCT = struct.Struct("<II")
META_STRUCT = struct.Struct("<IHH")
TERM_STRUCT = struct.Struct("<HI")
POSTING_STRUCT = struct.Struct("<II")
POS_COUNT_STRUCT = struct.Struct("<H")


def save_index(index: Index, path: Path, fmt: IndexFormat = "pickle") -> None:
    """Серіалізує інвертований індекс на диск у форматі pickle або власному бінарному форматі."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Збережено індекс у '%s' (pickle)", path)
        return

    with path.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                MAGIC,
                len(index.postings),
                len(index.doc_lengths),
                int(index.with_positions),
            )
        )
        for doc_id, length in index.doc_lengths.items():
            f.write(DOC_LEN_STRUCT.pack(doc_id, length))

        for doc_id, meta in index.doc_meta.items():
            path_bytes = meta.path.encode("utf-8")
            title_bytes = meta.title.encode("utf-8")
            f.write(META_STRUCT.pack(doc_id, len(path_bytes), len(title_bytes)))
            f.write(path_bytes)
            f.write(title_bytes)

        for term, plist in index.postings.items():
            term_bytes = term.encode("utf-8")
            f.write(TERM_STRUCT.pack(len(term_bytes), len(plist)))
            f.write(term_bytes)
            for p in plist:
                f.write(POSTING_STRUCT.pack(p.doc_id, p.tf))
                if index.with_positions:
                    f.write(POS_COUNT_STRUCT.pack(len(p.positions)))
                    if p.positions:
                        f.write(struct.pack(f"<{len(p.positions)}I", *p.positions))

    log.info("Збережено індекс у '%s' (binary)", path)


@timed
def load_index(path: Path, fmt: IndexFormat | None = None) -> Index:
    """Завантажує індекс із диска (автовизначення формату за сигнатурою MAGIC)."""
    if not path.exists():
        raise FileNotFoundError(f"Файл індексу не знайдено: {path}")

    with path.open("rb") as f:
        head = f.read(len(MAGIC))

    detected_fmt: IndexFormat = fmt if fmt is not None else ("binary" if head == MAGIC else "pickle")

    if detected_fmt == "pickle":
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            main_mod.Index = Index  # type: ignore[attr-defined]  # сумісність із pickle із __main__
            main_mod.InvertedIndex = Index  # type: ignore[attr-defined]  # псевдонім для старих індексів
            main_mod.Posting = Posting  # type: ignore[attr-defined]  # сумісність із pickle із __main__
            main_mod.DocMeta = DocMeta  # type: ignore[attr-defined]  # сумісність із pickle із __main__
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, Index):
            raise TypeError(f"Очікувався об'єкт Index у '{path}', отримано {type(loaded).__name__}")
        return loaded

    with path.open("rb") as f:
        raw_header = f.read(HEADER_STRUCT.size)
        if len(raw_header) < HEADER_STRUCT.size:
            raise ValueError(f"Пошкоджений заголовок індексу: {path}")
        magic, num_terms, num_docs, pos_flag = HEADER_STRUCT.unpack(raw_header)
        if magic != MAGIC:
            raise ValueError(f"Невірна сигнатура файлу індексу: {magic!r}")
        with_positions = bool(pos_flag)

        doc_lengths: dict[int, int] = {}
        for _ in range(num_docs):
            doc_id, length = DOC_LEN_STRUCT.unpack(f.read(DOC_LEN_STRUCT.size))
            doc_lengths[doc_id] = length

        doc_meta: dict[int, DocMeta] = {}
        for _ in range(num_docs):
            doc_id, path_len, title_len = META_STRUCT.unpack(f.read(META_STRUCT.size))
            doc_path = f.read(path_len).decode("utf-8")
            doc_title = f.read(title_len).decode("utf-8")
            doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=doc_path, title=doc_title)

        postings: dict[str, list[Posting]] = {}
        for _ in range(num_terms):
            term_len, plist_len = TERM_STRUCT.unpack(f.read(TERM_STRUCT.size))
            term = f.read(term_len).decode("utf-8")
            plist: list[Posting] = []
            for _ in range(plist_len):
                doc_id, tf = POSTING_STRUCT.unpack(f.read(POSTING_STRUCT.size))
                positions: tuple[int, ...] = ()
                if with_positions:
                    (pos_count,) = POS_COUNT_STRUCT.unpack(f.read(POS_COUNT_STRUCT.size))
                    if pos_count > 0:
                        positions = struct.unpack(f"<{pos_count}I", f.read(4 * pos_count))
                plist.append(Posting(doc_id=doc_id, tf=tf, positions=positions))
            postings[term] = plist

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )
