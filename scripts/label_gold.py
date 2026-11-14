"""Разметка gold-набора в терминале.

Из двух дней подачи берется случайная выборка статей с основной категорией cs.LG или cs.AI.
Выборка фиксируется в data/gold/gold_sample.csv, метки дописываются в
data/gold/gold_<annotator>.csv после каждого ответа, так что разметку можно прервать и продолжить.

Ответы: y – релевантна, n – нет, ? – не уверен, b – исправить предыдущую, q – выйти.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "raw" / "arxiv_meta.parquet"
GOLD = ROOT / "data" / "gold"
SAMPLE = GOLD / "gold_sample.csv"
CATEGORIES = ("cs.LG", "cs.AI")
LABELS = {"y": 1, "n": 0, "?": -1}


def build_sample(days: list[str] | None, per_day: int, seed: int) -> pd.DataFrame:
    meta = pd.read_parquet(META)
    meta = meta[meta["primary_category"].isin(CATEGORIES)].drop_duplicates("id")
    meta["day"] = meta["published"].str[:10]
    weekdays = meta[pd.to_datetime(meta["day"]).dt.dayofweek < 5]
    if not days:
        # Два последних будних дня, по которым данные заведомо полные.
        days = sorted(weekdays["day"].unique())[-3:-1]
    parts = [
        meta[meta["day"] == d].sample(n=min(per_day, (meta["day"] == d).sum()), random_state=seed)
        for d in days
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    return sample[["id", "day", "primary_category", "title", "abstract"]].reset_index(drop=True)


def show(paper: pd.Series, n: int, total: int) -> None:
    print("\033[2J\033[H", end="")
    print(f"[{n}/{total}] {paper['id']}  {paper['primary_category']}  {paper['day']}\n")
    print(textwrap.fill(paper["title"], 100), "\n")
    print(textwrap.fill(paper["abstract"], 100), "\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotator", default="author")
    parser.add_argument("--days", nargs="*", help="дни подачи YYYY-MM-DD (по умолчанию последние)")
    parser.add_argument("--per-day", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    GOLD.mkdir(parents=True, exist_ok=True)
    if not SAMPLE.exists():
        sample = build_sample(args.days, args.per_day, args.seed)
        sample.to_csv(SAMPLE, index=False, lineterminator="\n")
    sample = pd.read_csv(SAMPLE, dtype={"id": str})
    out = GOLD / f"gold_{args.annotator}.csv"
    done: dict[str, int] = {}
    if out.exists():
        labels = pd.read_csv(out, dtype={"id": str})
        done = dict(zip(labels["id"], labels["label"], strict=True))

    todo = [i for i, pid in enumerate(sample["id"].astype(str)) if pid not in done]
    history: list[str] = []
    pos = 0
    while pos < len(todo):
        paper = sample.iloc[todo[pos]]
        show(paper, len(done) + 1, len(sample))
        answer = input("y / n / ? / b – назад / q – выход: ").strip().lower()
        if answer == "q":
            break
        if answer == "b" and history:
            done.pop(history.pop())
            pos -= 1
        elif answer in LABELS:
            done[str(paper["id"])] = LABELS[answer]
            history.append(str(paper["id"]))
            pos += 1
        else:
            continue
        rows = [{"id": pid, "label": label} for pid, label in done.items()]
        labels = pd.DataFrame(rows).assign(annotator=args.annotator)
        labels.to_csv(out, index=False, lineterminator="\n")

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nРазмечено {len(done)} из {len(sample)} ({stamp}) -> {out}")


if __name__ == "__main__":
    main()
