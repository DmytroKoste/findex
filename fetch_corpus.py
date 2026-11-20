import json
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT_DIR = Path("data/wiki")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "FindexEducationalBot/0.1 (https://github.com/student/findex; student@example.com) Python-urllib/3.12"
}
API_URL = (
    "https://uk.wikipedia.org/w/api.php"
    "?action=query&generator=random&grnnamespace=0&grnlimit=50"
    "&prop=extracts&explaintext=1&format=json"
)


def download_articles(target_count: int = 1000) -> None:
    saved = len(list(OUT_DIR.glob("*.txt")))
    print(f"Вже є {saved} статей. Докачуємо до {target_count}...")

    while saved < target_count:
        req = urllib.request.Request(API_URL, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After", 10))
                print(f"Отримано 429 Too Many Requests, чекаємо {wait} с...")
                time.sleep(wait)
                continue
            raise

        pages = payload.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            text = page.get("extract", "").strip()
            title = page.get("title", f"doc_{page_id}")
            if len(text) < 300:
                continue
            file_path = OUT_DIR / f"{page_id}.txt"
            if not file_path.exists():
                file_path.write_text(f"{title}\n\n{text}", encoding="utf-8")
                saved += 1
                if saved % 100 == 0:
                    print(f"Збережено {saved}/{target_count} статей...")
                if saved >= target_count:
                    break
        time.sleep(1.5)


if __name__ == "__main__":
    download_articles(1000)
    print("Готово!")