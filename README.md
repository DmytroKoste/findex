# findex — Асинхронний веб-краулер та пошуковий рушій (Lab 6)

Ввічливий асинхронний веб-краулер (`asyncio` + `httpx` + `selectolax`) із потоковою індексацією в JSON Lines та інтеграцією в CLI пошукового рушія `findex`.

## Архітектура краулера (`src/findex/crawler/`)

- **`fetch.py`** — асинхронний HTTP-клієнт на базі `httpx.AsyncClient` із таймаутами на підключення та читання, явним `User-Agent` (`findex-bot/0.6`), повторними спробами з експоненційною затримкою + jitter для `429` (з урахуванням заголовка `Retry-After`) та `5xx`, і негайним поверненням помилок `4xx`.
- **`robots.py`** — потокобезпечний кеш правил `robots.txt` (`urllib.robotparser.RobotFileParser`) на рівні хосту (`asyncio.Lock`), щоб не завантажувати `robots.txt` повторно на кожен URL.
- **`crawl.py`** — асинхронний генератор `async def crawl(...) -> AsyncIterator[Page]`, побудований на `asyncio.Queue` (frontier), `asyncio.TaskGroup`, глобальному семафорі та семафорах/паузах на кожен хост, канонізації URL (`canonicalize_url`), дедуплікації через `seen` та парсингу HTML через `selectolax.lexbor.LexborHTMLParser`.
- **`corpus.py` / `index.py`** — потокове читання `.jsonl` (`iter_documents`) без завантаження всього корпусу в пам'ять та паралельна побудова інвертованого індексу.
- **`cli.py`** — команда `findex crawl` із живою консольною панеллю `rich.live.Live` (лічильники сторінок, `pages/s`, `in-flight`, помилок та глибини черги) і підтримкою `--debug`.

## Приклади використання CLI

```bash
# 1. Асинхронний обхід сайту з живим монітором та записом у JSON Lines
uv run findex crawl [https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/) --max-pages 50 --concurrency 10 --per-host 2 --delay 0.1 --out data/py_docs.jsonl --log data/py_crawl.log

# 2. Побудова інвертованого індексу напряму з потокового .jsonl
uv run findex index data/py_docs.jsonl --out data/py.idx

# 3. Булевий та ранжований пошук (BM25) по зібраному корпусу
uv run findex search data/py.idx "asyncio AND coroutine" -k 5