import asyncio
import json
import logging
import os
import re
from enum import StrEnum
from operator import attrgetter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.corpus import iter_documents, iter_files
from findex.crawler import CrawlStats, crawl
from findex.index import ExecutorKind, IndexFormat, build_index, open_index
from findex.rank import make_scorer, search
from findex.stats import compute_stats
from findex.store import save_index

app = typer.Typer(
    name="findex",
    help="findex — інвертований індекс та ранжований пошуковий рушій.",
    no_args_is_help=True,
)

stdout_console = Console()
stderr_console = Console(stderr=True)
_HIGHLIGHT_RE = re.compile(r"\[([^\[\]]+)\]")


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class FormatChoice(StrEnum):
    pickle = "pickle"
    binary = "binary"


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


def _format_rich_snippet(snippet: str) -> str:
    """Екранує квадратні дужки тексту й підсвічує лише знайдені терміни."""
    parts: list[str] = []
    last = 0
    for match in _HIGHLIGHT_RE.finditer(snippet):
        parts.append(escape(snippet[last : match.start()]))
        parts.append(f"[bold yellow]{escape(match.group(1))}[/bold yellow]")
        last = match.end()
    parts.append(escape(snippet[last:]))
    return "".join(parts)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handler = RichHandler(console=stderr_console, show_path=False, rich_tracebacks=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )


def _fail(message: str) -> None:
    stderr_console.print(f"[bold red]Помилка:[/bold red] {escape(message)}")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Рівень деталізації логів (-v для INFO, -vv для DEBUG).",
        ),
    ] = 0,
) -> None:
    """Глобальні параметри CLI findex."""
    _configure_logging(verbose)


@app.command("crawl")
def crawl_cmd(
    seed_url: Annotated[str, typer.Argument(help="Початковий URL для обходу.")],
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, help="Максимальна кількість сторінок.")] = 500,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1, help="Загальна кількість воркерів.")] = 10,
    per_host: Annotated[int, typer.Option("--per-host", min=1, help="Ліміт одночасних запитів на один хост.")] = 2,
    delay: Annotated[float, typer.Option("--delay", min=0.0, help="Мінімальна пауза (с) на хост.")] = 0.2,
    out: Annotated[Path, typer.Option("--out", "-o", help="Вихідний файл JSON Lines.")] = Path("data/crawl.jsonl"),
    log_file: Annotated[Path, typer.Option("--log", help="Файл журналу обходу URL.")] = Path("crawl.log"),
    debug: Annotated[bool, typer.Option("--debug", help="Увімкнути asyncio debug mode.")] = False,
) -> None:
    """Асинхронно обходить веб-сайт та потоково зберігає сторінки у JSON Lines."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    stats = CrawlStats()

    def _render_dashboard() -> Table:
        tbl = Table(title="findex crawl — живий монітор")
        tbl.add_column("Метрика", style="bold cyan")
        tbl.add_column("Значення", justify="right", style="green")
        tbl.add_row("Зібрано сторінок", f"{stats.fetched} / {max_pages}")
        tbl.add_row("Швидкість (pages/s)", f"{stats.pages_per_sec:.2f}")
        tbl.add_row("Активні запити (in-flight)", str(stats.in_flight))
        tbl.add_row("Помилки", str(stats.errors))
        tbl.add_row("Глибина черги", str(stats.queue_depth))
        return tbl

    async def _run() -> int:
        written = 0
        with (
            out.open("w", encoding="utf-8") as fp,
            Live(_render_dashboard(), console=stderr_console, refresh_per_second=8) as live,
        ):
            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                per_host_delay=delay,
                log_path=log_file,
                stats=stats,
            ):
                record = {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp.flush()
                written += 1
                live.update(_render_dashboard())
            live.update(_render_dashboard())
        return written

    total = asyncio.run(_run(), debug=debug)
    stderr_console.print(
        f"[green]Готово:[/green] зібрано {total} сторінок у [bold]{escape(str(out))}[/bold] "
        f"за {stats.elapsed:.2f} с ({stats.pages_per_sec:.2f} pages/s)."
    )


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Директорія з документами корпусу або файл .jsonl.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Шлях для збереження індексу.")] = Path("index.bin"),
    fmt: Annotated[FormatChoice, typer.Option("--format", "-f", help="Формат серіалізації.")] = FormatChoice.pickle,
    positions: Annotated[bool, typer.Option("--positions", help="Зберігати позиції токенів.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Обмежити кількість документів.")] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Кількість паралельних воркерів."),
    ] = os.cpu_count() or 1,
    executor: Annotated[
        ExecutorChoice,
        typer.Option("--executor", "-e", help="Стратегія виконання: serial, threads або processes."),
    ] = ExecutorChoice.processes,
) -> None:
    """Будує інвертований індекс корпусу та зберігає його на диск."""
    try:
        docs = list(iter_documents(corpus, limit=limit))
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        _fail(str(exc))
        return

    chosen_executor: ExecutorKind = (
        "serial"
        if executor == ExecutorChoice.serial
        else ("threads" if executor == ExecutorChoice.threads else "processes")
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=stderr_console,
    ) as progress:
        task = progress.add_task(
            f"Індексація ({chosen_executor}, w={workers})...",
            total=len(docs),
        )

        def _advance(_processed: int, _path: Path) -> None:
            progress.advance(task, 1)

        ix = build_index(
            corpus,
            with_positions=positions,
            variant="slots",
            limit=limit,
            on_doc=_advance,
            workers=workers,
            executor=chosen_executor,
        )

    chosen_fmt: IndexFormat = "binary" if fmt == FormatChoice.binary else "pickle"
    save_index(ix, out, fmt=chosen_fmt)
    size_kb = out.stat().st_size / 1024.0
    stderr_console.print(
        f"[green]Готово:[/green] {ix!r} збережено у [bold]{escape(str(out))}[/bold] ({size_kb:.1f} КБ)."
    )


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу.")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит (AND/OR/NOT, дужки, фрази).")],
    k: Annotated[int, typer.Option("--k", "-k", help="Кількість топ-результатів.")] = 10,
    scorer: Annotated[ScorerChoice, typer.Option("--scorer", "-s", help="Функція ранжування.")] = ScorerChoice.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="Вивести результати у форматі JSON Lines у stdout.")] = False,
) -> None:
    """Виконує ранжований пошук за індексом."""
    active_scorer = make_scorer("tfidf" if scorer == ScorerChoice.tfidf else "bm25")
    try:
        with open_index(index_path) as ix:
            results = search(ix, query, scorer=active_scorer, k=k, with_snippets=True)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _fail(str(exc))
        return

    ordered = sorted(results, key=attrgetter("score"), reverse=True)
    if as_json:
        for res in ordered:
            payload = {
                "doc_id": res.doc_id,
                "score": round(res.score, 6),
                "title": res.title,
                "path": res.path,
                "snippet": res.snippet,
            }
            stdout_console.file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout_console.file.flush()
        return

    if not ordered:
        stderr_console.print("[yellow]Нічого не знайдено.[/yellow]")
        return

    table = Table(title=f"Результати пошуку ({scorer.value.upper()}) — знайдено {len(ordered)}")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Заголовок", style="bold")
    table.add_column("Сніпет")

    for res in ordered:
        table.add_row(str(res.doc_id), f"{res.score:.4f}", escape(res.title), _format_rich_snippet(res.snippet))
    stdout_console.print(table)


@app.command("stats")
def stats_cmd(
    target: Annotated[Path, typer.Argument(help="Шлях до директорії корпусу або збереженого індексу.")],
    limit: Annotated[int | None, typer.Option("--limit", help="Обмежити кількість файлів корпусу.")] = None,
    top: Annotated[int, typer.Option("--top", help="Кількість найчастіших термінів.")] = 10,
) -> None:
    """Виводить зведену статистику корпусу або збереженого індексу."""
    if not target.exists():
        _fail(f"Шлях не знайдено: {target}")
        return

    table = Table(title=f"Статистика: {escape(str(target))}")
    table.add_column("Метрика", style="bold cyan")
    table.add_column("Значення", justify="right")

    if target.is_file():
        try:
            with open_index(target) as ix:
                total_tokens = sum(ix.doc_lengths.values())
                table.add_row("Документів у індексі", f"{ix.num_docs:,}")
                table.add_row("Унікальних термінів", f"{len(ix):,}")
                table.add_row("Загалом токенів", f"{total_tokens:,}")
                table.add_row("Середня довжина документа", f"{ix.avg_doc_length:.1f}")
        except (ValueError, TypeError, OSError) as exc:
            _fail(str(exc))
            return
        stdout_console.print(table)
        return

    st = compute_stats(iter_files(target, limit=limit), top_n=top)
    table.add_row("Документів", f"{st.num_docs:,}")
    table.add_row("Загалом токенів", f"{st.total_tokens:,}")
    table.add_row("Розмір словника", f"{st.vocab_size:,}")
    table.add_row("Час обробки", f"{st.elapsed_sec:.3f} с")
    table.add_row("Пік пам'яті", f"{st.peak_mem_mb:.2f} МБ")
    stdout_console.print(table)

    top_table = Table(title=f"Топ-{len(st.top_terms)} найчастіших термінів")
    top_table.add_column("Термін", style="bold")
    top_table.add_column("Частота", justify="right", style="green")
    for term, freq in st.top_terms:
        top_table.add_row(escape(term), f"{freq:,}")
    stdout_console.print(top_table)


if __name__ == "__main__":
    app()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Адреса прослуховування"),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Порт HTTP-сервера"),
    workers: int = typer.Option(1, "--workers", min=1, max=32, help="Кількість процесів Uvicorn"),
) -> None:
    """Запускає веб-сервер FastAPI (API + UI)."""
    import uvicorn

    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
    )
