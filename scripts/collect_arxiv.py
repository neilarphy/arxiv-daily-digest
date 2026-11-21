"""Сбор данных arXiv для EDA.

Подкоманды:
    meta      метаданные статей cs.LG и cs.AI за последние N дней -> data/raw/arxiv_meta.parquet
    monthly   число статей по месяцам за 2018-2026 -> data/processed/monthly_counts.csv
    fulltext  выборка полных текстов (HTML, LaTeX, PDF) -> data/processed/fulltext_sample.csv
    profile   заголовки и аннотации любимых статей -> data/gold/profile_<annotator>.json

Между запросами пауза 3 секунды по условиям arXiv API. Скачанное кэшируется в data/raw/,
поэтому повторный запуск не ходит в сеть за тем, что уже есть.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import feedparser
import httpx
import pandas as pd
import pymupdf
import tiktoken
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ("cs.LG", "cs.AI")
PAUSE_S = 3.0
USER_AGENT = "arxiv-daily-digest-eda (https://github.com/neilarphy/arxiv-daily-digest)"


class Client:
    """HTTP-клиент с паузой между запросами и повтором при временных ошибках."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
        )
        self._last = 0.0

    def get(self, url: str, params: dict | None = None, retries: int = 5) -> httpx.Response:
        for attempt in range(retries):
            wait = PAUSE_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError:
                if attempt == retries - 1:
                    raise
                time.sleep(PAUSE_S * (attempt + 2))
                continue
            if response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                limited = response.status_code == 429
                time.sleep(30 * 2**attempt if limited else PAUSE_S * (attempt + 2))
                continue
            return response
        raise RuntimeError("unreachable")


def _query(categories: tuple[str, ...], start: datetime, end: datetime) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    span = f"[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    return f"({cats}) AND submittedDate:{span}"


def _parse_entry(entry: feedparser.FeedParserDict) -> dict:
    arxiv_id, _, version = entry.id.rsplit("/", 1)[-1].partition("v")
    return {
        "id": arxiv_id,
        "version": int(version or 1),
        "title": " ".join(entry.title.split()),
        "abstract": " ".join(entry.summary.split()),
        "authors": [a.name for a in entry.get("authors", [])],
        "primary_category": entry.arxiv_primary_category["term"],
        "categories": [t["term"] for t in entry.get("tags", [])],
        "published": entry.published,
        "updated": entry.updated,
        "comment": entry.get("arxiv_comment"),
        "journal_ref": entry.get("arxiv_journal_ref"),
        "doi": entry.get("arxiv_doi"),
    }


def fetch_day(client: Client, day: date, page_size: int = 1000) -> list[dict]:
    """Все статьи cs.LG/cs.AI, поданные за сутки (UTC), с пагинацией."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    query = _query(CATEGORIES, start, start + timedelta(days=1) - timedelta(minutes=1))
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "search_query": query,
            "start": offset,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        feed = feedparser.parse(client.get(API_URL, params).text)
        total = int(feed.feed.get("opensearch_totalresults", 0))
        if total and not feed.entries and offset < total:
            feed = feedparser.parse(client.get(API_URL, params).text)
        rows.extend(_parse_entry(e) for e in feed.entries)
        offset += page_size
        if offset >= total or not feed.entries:
            return rows


def cmd_meta(days: int, last: date | None = None) -> None:
    cache = RAW / "meta_days"
    cache.mkdir(parents=True, exist_ok=True)
    client = Client()
    last = last or datetime.now(UTC).date() - timedelta(days=4)
    frames = []
    for shift in range(days - 1, -1, -1):
        day = last - timedelta(days=shift)
        path = cache / f"{day:%Y-%m-%d}.parquet"
        if not path.exists():
            rows = fetch_day(client, day)
            print(f"{day}: {len(rows)} статей")
            if not rows:
                continue
            pd.DataFrame(rows).to_parquet(path)
        frames.append(pd.read_parquet(path))
    meta = pd.concat(frames, ignore_index=True)
    meta.to_parquet(RAW / "arxiv_meta.parquet")
    print(f"Итого {len(meta)} строк -> {RAW / 'arxiv_meta.parquet'}")


def _total(client: Client, query: str) -> int:
    params = {"search_query": query, "max_results": 1}
    response = client.get(API_URL, params)
    feed = feedparser.parse(response.text)
    if response.status_code != 200 or "opensearch_totalresults" not in feed.feed:
        raise RuntimeError(f"arXiv API: HTTP {response.status_code} для {query}")
    return int(feed.feed["opensearch_totalresults"])


def cmd_monthly(first_year: int) -> None:
    out = PROCESSED / "monthly_counts.csv"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["month"])
    client = Client()
    rows = done.to_dict("records")
    month = date(first_year, 1, 1)
    last = datetime.now(UTC).date().replace(day=1)
    while month < last:
        nxt = (month + timedelta(days=32)).replace(day=1)
        key = f"{month:%Y-%m}"
        if key not in set(done["month"]):
            start = datetime(month.year, month.month, 1, tzinfo=UTC)
            end = datetime(nxt.year, nxt.month, 1, tzinfo=UTC) - timedelta(minutes=1)
            row = {"month": key}
            for name, cats in (("cs_lg", ("cs.LG",)), ("cs_ai", ("cs.AI",))):
                row[name] = _total(client, _query(cats, start, end))
            row["union"] = _total(client, _query(CATEGORIES, start, end))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
            print(row)
        month = nxt
    print(f"-> {out}")


ENCODING = tiktoken.get_encoding("o200k_base")


def _tokens(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


def _cached(client: Client, url: str, path: Path) -> tuple[int, bytes]:
    """Скачать url в path один раз; вернуть (HTTP-статус, тело)."""
    status_path = path.with_suffix(path.suffix + ".status")
    if status_path.exists():
        status = int(status_path.read_text())
        return status, path.read_bytes() if path.exists() else b""
    response = client.get(url)
    status_path.write_text(str(response.status_code))
    if response.status_code == 200:
        path.write_bytes(response.content)
    return response.status_code, response.content if response.status_code == 200 else b""


def _html_stats(html: bytes) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.body or soup
    errors = len(article.select(".ltx_ERROR"))
    full = article.get_text(" ", strip=True)
    for selector in (".ltx_bibliography", ".ltx_appendix"):
        for node in article.select(selector):
            node.decompose()
    body = article.get_text(" ", strip=True)
    return {
        "html_tokens": _tokens(full),
        "html_body_tokens": _tokens(body),
        "html_ltx_errors": errors,
    }


def _pdf_stats(pdf: bytes) -> dict:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
        pages = doc.page_count
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    return {
        "pdf_pages": pages,
        "pdf_tokens": _tokens(text),
        "pdf_cid_glyphs": len(re.findall(r"\(cid:\d+\)", text)),
        "pdf_nonprintable_share": round(1 - printable / max(len(text), 1), 4),
    }


def _source_kind(blob: bytes) -> str:
    """Тип e-print: tex (архив или одиночный .tex), pdf (исходников нет) или other."""
    if blob.startswith(b"%PDF"):
        return "pdf"
    data = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            names = tar.getnames()
        return "tex" if any(n.endswith(".tex") for n in names) else "other"
    except tarfile.TarError:
        return "tex" if b"\\documentclass" in data[:20000] else "other"


def cmd_fulltext(size: int, seed: int) -> None:
    meta = pd.read_parquet(RAW / "arxiv_meta.parquet")
    primary = meta[meta["primary_category"].isin(CATEGORIES)].drop_duplicates("id")
    sample = primary.sample(n=min(size, len(primary)), random_state=seed)
    cache = RAW / "fulltext"
    cache.mkdir(parents=True, exist_ok=True)
    client = Client()
    rows = []
    for n, paper in enumerate(sample.itertuples(), 1):
        pid = f"{paper.id}v{paper.version}"
        row: dict = {"id": paper.id, "primary_category": paper.primary_category}
        status, html = _cached(client, f"https://arxiv.org/html/{pid}", cache / f"{pid}.html")
        row["html_status"] = status
        if status == 200 and b"ltx_document" in html:
            row.update(_html_stats(html))
        status, src = _cached(client, f"https://arxiv.org/e-print/{pid}", cache / f"{pid}.src")
        row["source_kind"] = _source_kind(src) if status == 200 else f"http_{status}"
        status, pdf = _cached(client, f"https://arxiv.org/pdf/{pid}", cache / f"{pid}.pdf")
        row["pdf_status"] = status
        if status == 200:
            try:
                row.update(_pdf_stats(pdf))
            except (pymupdf.FileDataError, RuntimeError) as exc:
                row["pdf_error"] = str(exc)[:200]
        rows.append(row)
        print(f"[{n}/{len(sample)}] {pid}: html={row['html_status']} src={row['source_kind']}")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "fulltext_sample.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"-> {out}")


def cmd_profile(annotator: str) -> None:
    """Дописать в профиль заголовки и аннотации любимых статей по их id."""
    path = ROOT / "data" / "gold" / f"profile_{annotator}.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    ids = [re.sub(r"v\d+$", "", i.rstrip("/").rsplit("/", 1)[-1]) for i in profile["favorites"]]
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    feed = feedparser.parse(Client().get(API_URL, params).text)
    papers = [_parse_entry(e) for e in feed.entries]
    profile["favorite_papers"] = [
        {k: p[k] for k in ("id", "title", "abstract", "primary_category")} for p in papers
    ]
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for p in profile["favorite_papers"]:
        print(f"{p['id']}: {p['title']}")
    missing = set(ids) - {p["id"] for p in papers}
    if missing:
        raise SystemExit(f"Не найдены в arXiv: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    meta = sub.add_parser("meta")
    meta.add_argument("--days", type=int, default=42)
    meta.add_argument("--last", type=date.fromisoformat, help="последний день YYYY-MM-DD")
    monthly = sub.add_parser("monthly")
    monthly.add_argument("--first-year", type=int, default=2018)
    fulltext = sub.add_parser("fulltext")
    fulltext.add_argument("--size", type=int, default=100)
    fulltext.add_argument("--seed", type=int, default=42)
    profile = sub.add_parser("profile")
    profile.add_argument("--annotator", default="author")
    args = parser.parse_args()
    if args.command == "meta":
        cmd_meta(args.days, args.last)
    elif args.command == "monthly":
        cmd_monthly(args.first_year)
    elif args.command == "profile":
        cmd_profile(args.annotator)
    else:
        cmd_fulltext(args.size, args.seed)


if __name__ == "__main__":
    main()
