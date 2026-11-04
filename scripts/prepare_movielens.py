"""Convert local MovieLens ratings into user_id,item_id,timestamp CSV.

No data is downloaded. Both MovieLens-1M ratings.dat and newer ratings.csv
files are supported. With --min-rating, only ratings at or above that value
are interpreted as positive interactions; the choice changes the task.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple


def rows(path: Path, format_name: str) -> Iterator[Tuple[str, str, float, float]]:
    if format_name == "ml1m":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                parts = line.strip().split("::")
                if len(parts) != 4:
                    raise ValueError(f"expected user::item::rating::timestamp on line {line_number}")
                user, item, rating, timestamp = parts
                yield user, item, float(rating), float(timestamp)
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if {"userId", "movieId", "rating", "timestamp"}.issubset(fields):
                user_key, item_key = "userId", "movieId"
            elif {"user_id", "item_id", "timestamp"}.issubset(fields):
                user_key, item_key = "user_id", "item_id"
            else:
                raise ValueError("CSV needs MovieLens userId/movieId or user_id/item_id columns, plus timestamp")
            for row in reader:
                yield row[user_key], row[item_key], float(row.get("rating", 5)), float(row["timestamp"])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=["auto", "ml1m", "csv"], default="auto")
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.output.exists() and not args.overwrite:
        parser.error("output already exists; choose another path or pass --overwrite")
    if not math.isfinite(args.min_rating):
        parser.error("min-rating must be finite")
    format_name = args.format
    if format_name == "auto":
        format_name = "ml1m" if args.input.suffix.lower() == ".dat" else "csv"
    converted = []
    for user, item, rating, timestamp in rows(args.input, format_name):
        if not user or not item or not math.isfinite(rating) or not math.isfinite(timestamp):
            raise ValueError("IDs must be nonempty and ratings/timestamps finite")
        if rating >= args.min_rating:
            converted.append((user, item, timestamp))
    if not converted:
        raise ValueError("no interactions remain after the rating threshold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp"])
        writer.writerows(converted)
    print(f"Wrote {len(converted)} interactions to {args.output}; min_rating={args.min_rating:g}")


if __name__ == "__main__":
    main()
