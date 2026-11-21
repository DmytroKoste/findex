import json
import pickle
import struct
import sys
import time
from array import array
from pathlib import Path

from findex.index import (
    ArrayInvertedIndex,
    DocMeta,
    InvertedIndex,
    PlainPosting,
    Posting,
)

# БЕЗПЕКА (pickle):
# pickle.load() виконує довільний байткод Python під час десеріалізації об'єктів
# (через спеціальний метод __reduce__). Якщо завантажити чужий або підмінений
# файл індексу з мережі, зловмисник отримає віддалене виконання коду (RCE).
# Тому pickle допустимий ЛИШЕ для власних локально згенерованих файлів.

MAGIC = b"FIDX0002"


def save_index(index: object, path: Path, fmt: str = "pickle") -> None:
    """Зберігає індекс на диск у форматі 'pickle' або 'binary' (array('I') + JSON)."""
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fmt == "binary":
        if not isinstance(index, InvertedIndex):
            raise TypeError("Бінарний формат підтримує лише InvertedIndex")

        terms_meta: dict[str, tuple[int, int]] = {}
        payload = bytearray()

        for term, plist in index.postings.items():
            doc_ids = array("I", (p.doc_id for p in plist))
            tfs = array("I", (p.tf for p in plist))
            offset = len(payload)
            payload.extend(doc_ids.tobytes())
            payload.extend(tfs.tobytes())
            terms_meta[term] = (offset, len(plist))

        header_obj = {
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): [m.doc_id, m.path, m.title] for k, m in index.doc_meta.items()
            },
            "terms": terms_meta,
        }
        header_bytes = json.dumps(header_obj, ensure_ascii=False).encode("utf-8")

        with path.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
            f.write(payload)
        return

    raise ValueError(f"Невідомий формат: {fmt}")


def load_index(path: Path, fmt: str | None = None) -> InvertedIndex:
    """Завантажує індекс із диска (автовизначення формату за сигнатурою MAGIC)."""
    with path.open("rb") as f:
        head = f.read(len(MAGIC))

    detected_fmt = fmt or ("binary" if head == MAGIC else "pickle")

    if detected_fmt == "pickle":
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            for cls in (Posting, PlainPosting, DocMeta, InvertedIndex, ArrayInvertedIndex):
                setattr(main_mod, cls.__name__, cls)
        with path.open("rb") as f:
            return pickle.load(f)

    if detected_fmt == "binary":
        with path.open("rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError("Невірна сигнатура бінарного індексу")
            (header_len,) = struct.unpack("<I", f.read(4))
            header_obj = json.loads(f.read(header_len).decode("utf-8"))
            payload = f.read()

        doc_lengths = {int(k): int(v) for k, v in header_obj["doc_lengths"].items()}
        doc_meta = {
            int(k): DocMeta(doc_id=v[0], path=v[1], title=v[2])
            for k, v in header_obj["doc_meta"].items()
        }

        postings: dict[str, list[Posting]] = {}
        mv = memoryview(payload)
        for term, (offset, count) in header_obj["terms"].items():
            byte_len = count * 4
            doc_ids = array("I")
            doc_ids.frombytes(mv[offset : offset + byte_len])
            tfs = array("I")
            tfs.frombytes(mv[offset + byte_len : offset + 2 * byte_len])
            postings[term] = [Posting(doc_id=d, tf=t) for d, t in zip(doc_ids, tfs)]

        return InvertedIndex(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)

    raise ValueError(f"Невідомий формат: {detected_fmt}")


def benchmark_formats(index: InvertedIndex) -> None:
    """Міряє розмір файлу, час save та load для pickle і binary."""
    print(f"{'Формат':<12} | {'Розмір файлу':<14} | {'Час save':<12} | {'Час load':<12}")
    print("-" * 60)

    for fmt in ("pickle", "binary"):
        tmp = Path(f"_tmp_index_{fmt}.bin")
        try:
            t0 = time.perf_counter()
            save_index(index, tmp, fmt=fmt)
            save_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            _ = load_index(tmp, fmt=fmt)
            load_ms = (time.perf_counter() - t1) * 1000

            size_mb = tmp.stat().st_size / (1024 * 1024)
            print(f"{fmt:<12} | {size_mb:>10.2f} MB | {save_ms:>9.1f} ms | {load_ms:>9.1f} ms")
        finally:
            if tmp.exists():
                tmp.unlink()