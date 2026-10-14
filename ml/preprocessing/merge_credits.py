"""
merge_credits.py
----------------
Merges cached TMDB cast and director information into movies.parquet,
saving the resulting dataset to data/processed/movies_enriched.parquet.
Reports % filled per language.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_PARQUET = PROJECT_ROOT / "data" / "processed" / "movies.parquet"
CREDITS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "credits_cache.jsonl"
MOVIES_ENRICHED_PARQUET = PROJECT_ROOT / "data" / "processed" / "movies_enriched.parquet"


def load_credits_cache(cache_path: Path) -> dict[int, tuple[list[str], str | None]]:
    """Load credits cache into dict tmdb_id -> (cast_list, director_str)."""
    credits_map: dict[int, tuple[list[str], str | None]] = {}
    if not cache_path.exists():
        print(f"Warning: Credits cache file does not exist at {cache_path}")
        return credits_map

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                tmdb_id = int(rec["tmdb_id"])
                cast = rec.get("cast") or []
                director = rec.get("director") or None
                credits_map[tmdb_id] = (cast, director)
            except Exception:
                continue
    return credits_map


def merge_credits(
    movies_path: Path = MOVIES_PARQUET,
    cache_path: Path = CREDITS_CACHE_PATH,
    output_path: Path = MOVIES_ENRICHED_PARQUET,
) -> pd.DataFrame:
    """Merge cached credits into movies.parquet and save movies_enriched.parquet."""
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies parquet file not found at {movies_path}")

    df = pd.read_parquet(movies_path)
    credits_map = load_credits_cache(cache_path)
    print(f"Loaded {len(credits_map):,} cached credit records.")

    def get_cast(tmdb_id: int) -> list[str] | None:
        if tmdb_id in credits_map:
            cast, _ = credits_map[tmdb_id]
            return cast if cast else None
        return None

    def get_director(tmdb_id: int) -> str | None:
        if tmdb_id in credits_map:
            _, director = credits_map[tmdb_id]
            return director
        return None

    # Update cast and director columns
    df["cast"] = df["tmdb_id"].apply(get_cast)
    df["director"] = df["tmdb_id"].apply(get_director)

    # Save enriched dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved enriched dataset ({len(df):,} rows) to: {output_path}")

    # Calculate and print % filled per language
    print("\n" + "=" * 60)
    print("CREDITS ENRICHMENT REPORT (% Filled per Language)")
    print("=" * 60)

    languages = ["en", "hi", "mr"]
    for lang in languages:
        sub = df[df["language"] == lang]
        total_lang = len(sub)
        if total_lang == 0:
            continue

        cast_filled = sub["cast"].apply(lambda c: isinstance(c, (list, tuple, range)) and len(c) > 0).sum()
        director_filled = sub["director"].apply(lambda d: isinstance(d, str) and len(d.strip()) > 0).sum()

        cast_pct = (cast_filled / total_lang) * 100
        director_pct = (director_filled / total_lang) * 100

        print(
            f"Language `{lang}` (total {total_lang:,} movies):\n"
            f"  - Cast filled    : {cast_filled:,} / {total_lang:,} ({cast_pct:.2f}%)\n"
            f"  - Director filled: {director_filled:,} / {total_lang:,} ({director_pct:.2f}%)\n"
        )

    return df


if __name__ == "__main__":
    merge_credits()
