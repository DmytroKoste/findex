import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from findex.corpus import iter_files
from findex.index import ExecutorKind, build_index


def get_cpu_info() -> tuple[str, int, int]:
    """Повертає (модель CPU, фізичні ядра, логічні ядра)."""
    logical = os.cpu_count() or 1
    physical = logical
    cpu_name = platform.processor() or "Unknown CPU"
    try:
        import psutil  # type: ignore[import-untyped]

        phys = psutil.cpu_count(logical=False)
        if phys:
            physical = phys
    except Exception:
        pass

    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                cpu_name = out
        except Exception:
            pass
    return cpu_name, physical, logical


def get_peak_rss_mb() -> float:
    """Повертає пікове споживання пам'яті (RSS) поточного процесу у МБ."""
    try:
        import psutil  # type: ignore[import-untyped]

        mem = psutil.Process().memory_info()
        peak = getattr(mem, "peak_wset", mem.rss)
        return peak / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def measure_single_run(
    corpus_dir: Path,
    executor: ExecutorKind,
    workers: int,
) -> dict[str, float]:
    """Виконує один прогін build_index у чистому підпроцесі для точного Peak RSS і CPU time."""
    code = (
        "import gc, json, time\n"
        "from pathlib import Path\n"
        "from findex.index import build_index\n"
        "try:\n"
        "    import psutil\n"
        "    proc = psutil.Process()\n"
        "except Exception:\n"
        "    proc = None\n"
        "gc.collect()\n"
        "merge_times = []\n"
        "t_wall0 = time.perf_counter()\n"
        "t_cpu0 = time.process_time()\n"
        f"ix = build_index(Path({str(corpus_dir)!r}), executor={executor!r}, "
        f"workers={workers}, merge_time_out=merge_times)\n"
        "wall = time.perf_counter() - t_wall0\n"
        "cpu = time.process_time() - t_cpu0\n"
        "rss_mb = 0.0\n"
        "if proc is not None:\n"
        "    mi = proc.memory_info()\n"
        "    rss_mb = getattr(mi, 'peak_wset', mi.rss) / (1024.0 * 1024.0)\n"
        "res = {'wall': wall, 'cpu': cpu, 'rss_mb': rss_mb, "
        "'merge': merge_times[0], 'docs': ix.num_docs, 'terms': len(ix)}\n"
        "print(json.dumps(res))\n"
    )
    res = subprocess.check_output([sys.executable, "-c", code], text=True)
    return json.loads(res.strip().splitlines()[-1])


def run_io_bound_benchmark(
    files: list[Path],
    worker_counts: list[int],
    repeats: int = 3,
) -> dict[int, float]:
    """Контр-бенчмарк: чисте I/O читання файлів корпусу з диска у потоках."""

    def _read_chunk(paths: list[Path]) -> int:
        total_bytes = 0
        for p in paths:
            total_bytes += len(p.read_bytes())
        return total_bytes

    results: dict[int, float] = {}
    for w in worker_counts:
        step = max(1, (len(files) + w - 1) // w)
        chunks = [files[i : i + step] for i in range(0, len(files), step)]
        samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            if w == 1:
                _read_chunk(files)
            else:
                with ThreadPoolExecutor(max_workers=w) as pool:
                    list(pool.map(_read_chunk, chunks))
            samples.append(time.perf_counter() - t0)
        results[w] = statistics.median(samples)
    return results


def run_race_demo() -> dict[str, int | bool]:
    """Демонстрація гонки оновлень спільного стану та виправлення через Lock."""
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    iterations = 200_000
    threads_count = 4
    expected = iterations * threads_count

    shared_counter: Counter[str] = Counter()

    def unsafe_worker() -> None:
        for _ in range(iterations):
            shared_counter["hits"] = shared_counter["hits"] + 1

    workers = [threading.Thread(target=unsafe_worker) for _ in range(threads_count)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    unsafe_actual = shared_counter["hits"]

    safe_counter: Counter[str] = Counter()
    lock = threading.Lock()

    def safe_worker() -> None:
        for _ in range(iterations):
            with lock:
                safe_counter["hits"] = safe_counter["hits"] + 1

    workers_safe = [threading.Thread(target=safe_worker) for _ in range(threads_count)]
    for t in workers_safe:
        t.start()
    for t in workers_safe:
        t.join()
    safe_actual = safe_counter["hits"]

    return {
        "gil_enabled": bool(gil_enabled),
        "expected": expected,
        "unsafe_actual": unsafe_actual,
        "lost_updates": expected - unsafe_actual,
        "safe_actual": safe_actual,
    }


def main() -> None:
    if "--race-only" in sys.argv:
        print(json.dumps(run_race_demo()))
        return

    corpus_dir = Path("data")
    if not corpus_dir.exists():
        raise FileNotFoundError("Директорію 'data' з корпусом не знайдено.")

    files = list(iter_files(corpus_dir))
    cpu_name, physical_cores, logical_cores = get_cpu_info()
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()

    raw_workers = [1, 2, 4, 8, logical_cores]
    worker_counts = sorted({w for w in raw_workers if 1 <= w <= logical_cores})

    print(f"Python: {sys.version.split()[0]} (GIL={gil_enabled})")
    print(f"CPU: {cpu_name} | Фізичних ядер: {physical_cores} | Логічних ядер: {logical_cores}")
    print(f"Файлів у корпусі: {len(files)}")
    print(f"Конфігурації воркерів: {worker_counts}\n")

    build_index(corpus_dir, executor="serial", workers=1)
    gc.collect()

    repeats = 3
    rows: list[dict[str, object]] = []

    serial_runs = [measure_single_run(corpus_dir, "serial", 1) for _ in range(repeats)]
    baseline_wall = statistics.median(r["wall"] for r in serial_runs)
    rows.append(
        {
            "runtime": "CPython 3.12 (GIL)",
            "executor": "serial",
            "workers": 1,
            "wall": baseline_wall,
            "cpu": statistics.median(r["cpu"] for r in serial_runs),
            "rss_mb": statistics.median(r["rss_mb"] for r in serial_runs),
            "merge": statistics.median(r["merge"] for r in serial_runs),
            "speedup": 1.0,
        }
    )

    for executor in ("threads", "processes"):
        for w in worker_counts:
            if w == 1:
                continue
            runs = [measure_single_run(corpus_dir, executor, w) for _ in range(repeats)]  # type: ignore[arg-type]
            med_wall = statistics.median(r["wall"] for r in runs)
            rows.append(
                {
                    "runtime": "CPython 3.12 (GIL)",
                    "executor": executor,
                    "workers": w,
                    "wall": med_wall,
                    "cpu": statistics.median(r["cpu"] for r in runs),
                    "rss_mb": statistics.median(r["rss_mb"] for r in runs),
                    "merge": statistics.median(r["merge"] for r in runs),
                    "speedup": baseline_wall / med_wall,
                }
            )

    py313t = Path(".venv-313t/Scripts/python.exe")
    if py313t.exists():
        for w in worker_counts:
            exe_kind = "serial" if w == 1 else "threads"
            runs_313t: list[dict[str, float]] = []
            for _ in range(repeats):
                code = (
                    "import gc, json, time\n"
                    "from pathlib import Path\n"
                    "from findex.index import build_index\n"
                    "gc.collect()\n"
                    "merge_times = []\n"
                    "t_wall0 = time.perf_counter()\n"
                    "t_cpu0 = time.process_time()\n"
                    f"ix = build_index(Path({str(corpus_dir)!r}), executor={exe_kind!r}, "
                    f"workers={w}, merge_time_out=merge_times)\n"
                    "wall = time.perf_counter() - t_wall0\n"
                    "cpu = time.process_time() - t_cpu0\n"
                    "print(json.dumps({'wall': wall, 'cpu': cpu, 'rss_mb': 0.0, 'merge': merge_times[0]}))\n"
                )
                out = subprocess.check_output([str(py313t), "-c", code], text=True)
                runs_313t.append(json.loads(out.strip().splitlines()[-1]))
            med_wall = statistics.median(r["wall"] for r in runs_313t)
            rows.append(
                {
                    "runtime": "Python 3.13t (no-GIL)",
                    "executor": f"threads ({exe_kind})" if w == 1 else "threads",
                    "workers": w,
                    "wall": med_wall,
                    "cpu": statistics.median(r["cpu"] for r in runs_313t),
                    "rss_mb": 0.0,
                    "merge": statistics.median(r["merge"] for r in runs_313t),
                    "speedup": baseline_wall / med_wall,
                }
            )

    print("| Runtime | Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS (MB) | Merge (s) | Speedup |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rss_str = f"{r['rss_mb']:.1f}" if float(r["rss_mb"]) > 0 else "—"
        print(
            f"| {r['runtime']} | {r['executor']} | {r['workers']} | "
            f"{r['wall']:.3f} | {r['cpu']:.3f} | {rss_str} | {r['merge']:.3f} | **{r['speedup']:.2f}×** |"
        )

    Path("docs").mkdir(exist_ok=True)
    io_res = run_io_bound_benchmark(files, worker_counts, repeats=repeats)
    race_312 = run_race_demo()
    race_313t = None
    if py313t.exists():
        race_out = subprocess.check_output([str(py313t), "bench_lab5.py", "--race-only"], text=True)
        race_313t = json.loads(race_out.strip().splitlines()[-1])

    summary = {
        "cpu_name": cpu_name,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "num_files": len(files),
        "rows": rows,
        "io_bound_wall_s": io_res,
        "race_312": race_312,
        "race_313t": race_313t,
    }
    Path("docs/bench_lab5_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- I/O-bound контр-бенчмарк (читання файлів корпусу у потоках) ---")
    io_base = io_res[1]
    for w, t_val in io_res.items():
        print(f"workers={w}: {t_val:.4f} s (speedup {io_base / t_val:.2f}×)")

    print("\n--- Експеримент із гонкою даних (Counter += 1 у 4 потоках × 200 000) ---")
    print("Python 3.12 (GIL=True):", race_312)
    if race_313t:
        print("Python 3.13t (GIL=False):", race_313t)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 5.2), dpi=150)
    max_w = max(worker_counts)
    ideal_x = list(range(1, max_w + 1))
    plt.plot(ideal_x, ideal_x, "--", color="gray", linewidth=1.5, label="Ідеальне лінійне прискорення (y = x)")

    th_rows = [r for r in rows if r["runtime"] == "CPython 3.12 (GIL)" and r["executor"] == "threads"]
    th_x = [1] + [int(r["workers"]) for r in th_rows]
    th_y = [1.0] + [float(r["speedup"]) for r in th_rows]
    plt.plot(th_x, th_y, "o-", color="#d9534f", linewidth=2, label="threads (CPython 3.12, GIL=True)")

    pr_rows = [r for r in rows if r["runtime"] == "CPython 3.12 (GIL)" and r["executor"] == "processes"]
    pr_x = [1] + [int(r["workers"]) for r in pr_rows]
    pr_y = [1.0] + [float(r["speedup"]) for r in pr_rows]
    plt.plot(pr_x, pr_y, "s-", color="#0275d8", linewidth=2, label="processes (CPython 3.12, ProcessPoolExecutor)")

    ft_rows = [r for r in rows if r["runtime"] == "Python 3.13t (no-GIL)"]
    if ft_rows:
        ft_x = [int(r["workers"]) for r in ft_rows]
        ft_y = [float(r["speedup"]) for r in ft_rows]
        plt.plot(ft_x, ft_y, "^-", color="#5cb85c", linewidth=2, label="threads (Python 3.13t, free-threaded no-GIL)")

    title = (
        f"Прискорення паралельної індексації findex ({len(files)} документів)\n"
        f"{cpu_name} ({physical_cores}P / {logical_cores}L)"
    )
    plt.title(title)
    plt.xlabel("Кількість воркерів (workers)")
    plt.ylabel("Прискорення (Speedup відносно serial 1 worker)")
    plt.xticks(worker_counts)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/speedup_lab5.png")
    print("\nГрафік збережено у docs/speedup_lab5.png")


if __name__ == "__main__":
    main()