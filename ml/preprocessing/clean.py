"""
clean.py
--------
Reusable cleaning functions and processing pipeline for TMDB movies dataset.
"""

from __future__ import annotations

import os
import re
import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.preprocessing.config import (
    MIN_VOTE_COUNT,
    TARGET_LANGUAGES,
    REQUIRED_STATUS,
    ALLOW_ADULT,
    REQUIRE_OVERVIEW,
    REQUIRE_RELEASE_DATE,
    REQUIRE_POSTER_PATH,
    MIN_RUNTIME,
    MIN_YEAR,
    MAX_YEAR,
    MIN_VOTE_AVERAGE,
    MAX_VOTE_AVERAGE,
    USE_COLS,
    CHUNK_SIZE,
)
from ml.preprocessing.schema import Movie


# ---------------------------------------------------------------------------
# Reusable Cleaning Functions
# ---------------------------------------------------------------------------

def clean_text(val: Any) -> str:
    """Return stripped string or empty string if None/NaN/empty."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return s


def parse_comma_list(val: Any) -> list[str]:
    """Parse comma-separated string into a list of trimmed, non-empty strings."""
    if val is None or pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def normalize_genres(val: Any) -> list[str]:
    """Parse and normalize genre strings into standard title-cased list."""
    items = parse_comma_list(val)
    return [item.title() for item in items]


def parse_year(val: Any) -> Optional[int]:
    """Extract 4-digit year as int from date string or value (e.g. YYYY-MM-DD)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    match = re.search(r"\b(1\d{3}|20\d{2})\b", s)
    if match:
        return int(match.group(1))
    return None


def deduplicate_by_id(
    df: pd.DataFrame, id_col: str = "tmdb_id", vote_col: str = "vote_count"
) -> pd.DataFrame:
    """
    Deduplicate DataFrame by id_col, keeping the row with the highest vote_col.
    """
    if df.empty or id_col not in df.columns:
        return df

    sort_cols = [id_col]
    ascending = [True]
    if vote_col in df.columns:
        sort_cols.append(vote_col)
        ascending.append(False)

    df_sorted = df.sort_values(by=sort_cols, ascending=ascending)
    df_dedup = df_sorted.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    return df_dedup


# ---------------------------------------------------------------------------
# Main Pipeline Logic
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_csv_path: Path,
    links_csv_path: Path,
    output_movies_path: Path,
    output_links_path: Path,
) -> pd.DataFrame:
    """
    Executes the schema + cleaning pipeline, prints funnel statistics,
    saves movies.parquet and ml_links.parquet, and prints final report.
    """
    print(f"Reading raw TMDB dataset from: {raw_csv_path}")

    # Track funnel counts for each language across steps
    # Steps:
    # 0: Raw in target languages
    # 1: Status == Released & Adult == False
    # 2: Required text fields (overview, release_date, poster_path)
    # 3: Valid year (1900..MAX_YEAR)
    # 4: Runtime > 0 & vote_average 0..10
    # 5: Language vote count threshold (en>=50, hi>=5, mr>=1)
    # 6: Deduplicated by tmdb_id (highest vote_count)

    step_names = [
        "0. Target language (en/hi/mr)",
        "1. Status==Released & Adult==False",
        "2. Non-empty overview, release_date, poster_path",
        "3. Valid year (1900..2026)",
        "4. Runtime > 0 & vote_average 0..10",
        "5. Vote count threshold (en>=50, hi>=5, mr>=1)",
        "6. Deduplicate by tmdb_id (highest vote_count)",
    ]

    funnel_counts = {step: {lang: 0 for lang in TARGET_LANGUAGES} for step in range(7)}

    filtered_chunks: list[pd.DataFrame] = []

    chunk_iter = pd.read_csv(
        raw_csv_path,
        usecols=USE_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in chunk_iter:
        # Normalize column names & clean string types
        df = chunk.copy()

        # Step 0: Language filter
        df["language"] = df["original_language"].astype(str).str.strip()
        df = df[df["language"].isin(TARGET_LANGUAGES)].copy()
        for lang in TARGET_LANGUAGES:
            funnel_counts[0][lang] += int((df["language"] == lang).sum())

        if df.empty:
            continue

        # Step 1: Status == Released & Adult == False
        # adult can be bool or string/int
        df["adult_bool"] = df["adult"].fillna(False).astype(bool)
        df["status_str"] = df["status"].astype(str).str.strip()

        mask_step1 = (df["status_str"] == REQUIRED_STATUS) & (df["adult_bool"] == ALLOW_ADULT)
        df = df[mask_step1].copy()
        for lang in TARGET_LANGUAGES:
            funnel_counts[1][lang] += int((df["language"] == lang).sum())

        if df.empty:
            continue

        # Step 2: Require non-empty overview, release_date, poster_path
        df["overview_clean"] = df["overview"].apply(clean_text)
        df["release_date_clean"] = df["release_date"].apply(clean_text)
        df["poster_path_clean"] = df["poster_path"].apply(clean_text)

        mask_step2 = (
            (df["overview_clean"] != "")
            & (df["release_date_clean"] != "")
            & (df["poster_path_clean"] != "")
        )
        df = df[mask_step2].copy()
        for lang in TARGET_LANGUAGES:
            funnel_counts[2][lang] += int((df["language"] == lang).sum())

        if df.empty:
            continue

        # Step 3: Valid year parsing (1900..MAX_YEAR)
        df["release_year"] = df["release_date_clean"].apply(parse_year)
        mask_step3 = (
            df["release_year"].notna()
            & (df["release_year"] >= MIN_YEAR)
            & (df["release_year"] <= MAX_YEAR)
        )
        df = df[mask_step3].copy()
        df["release_year"] = df["release_year"].astype(int)
        for lang in TARGET_LANGUAGES:
            funnel_counts[3][lang] += int((df["language"] == lang).sum())

        if df.empty:
            continue

        # Step 4: Runtime > 0 and 0 <= vote_average <= 10
        df["runtime_num"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0.0)
        df["vote_avg_num"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)

        mask_step4 = (
            (df["runtime_num"] > MIN_RUNTIME)
            & (df["vote_avg_num"] >= MIN_VOTE_AVERAGE)
            & (df["vote_avg_num"] <= MAX_VOTE_AVERAGE)
        )
        df = df[mask_step4].copy()
        for lang in TARGET_LANGUAGES:
            funnel_counts[4][lang] += int((df["language"] == lang).sum())

        if df.empty:
            continue

        # Step 5: Vote count threshold per language
        df["vote_count_num"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
        
        # Build language-specific vote count mask
        vote_mask = pd.Series(False, index=df.index)
        for lang, min_votes in MIN_VOTE_COUNT.items():
            vote_mask |= (df["language"] == lang) & (df["vote_count_num"] >= min_votes)
        
        df = df[vote_mask].copy()
        for lang in TARGET_LANGUAGES:
            funnel_counts[5][lang] += int((df["language"] == lang).sum())

        if not df.empty:
            filtered_chunks.append(df)

    # Combine all filtered chunks
    if filtered_chunks:
        df_combined = pd.concat(filtered_chunks, ignore_index=True)
    else:
        df_combined = pd.DataFrame()

    # Step 6: Deduplicate by tmdb_id (keeping highest vote_count)
    df_combined["tmdb_id"] = pd.to_numeric(df_combined["id"], errors="coerce").astype(int)
    df_clean = deduplicate_by_id(df_combined, id_col="tmdb_id", vote_col="vote_count_num")

    for lang in TARGET_LANGUAGES:
        funnel_counts[6][lang] = int((df_clean["language"] == lang).sum())

    # Map cleaned columns to final Schema
    df_clean["title"] = df_clean["title"].apply(clean_text)
    df_clean["original_title"] = df_clean["original_title"].apply(clean_text)
    df_clean["overview"] = df_clean["overview_clean"]
    df_clean["tagline"] = df_clean["tagline"].apply(lambda v: clean_text(v) or None)
    df_clean["genres"] = df_clean["genres"].apply(normalize_genres)
    df_clean["keywords"] = df_clean["keywords"].apply(parse_comma_list)
    df_clean["runtime"] = df_clean["runtime_num"]
    df_clean["vote_average"] = df_clean["vote_avg_num"]
    df_clean["vote_count"] = df_clean["vote_count_num"]
    df_clean["popularity"] = pd.to_numeric(df_clean["popularity"], errors="coerce").fillna(0.0)
    df_clean["poster_path"] = df_clean["poster_path_clean"]
    df_clean["backdrop_path"] = df_clean["backdrop_path"].apply(lambda v: clean_text(v) or None)
    df_clean["imdb_id"] = df_clean["imdb_id"].apply(lambda v: clean_text(v) or None)
    df_clean["cast"] = None
    df_clean["director"] = None

    final_cols = [
        "tmdb_id",
        "title",
        "original_title",
        "language",
        "overview",
        "tagline",
        "genres",
        "keywords",
        "release_year",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "poster_path",
        "backdrop_path",
        "imdb_id",
        "cast",
        "director",
    ]
    df_final = df_clean[final_cols].copy()

    # Validate against Pydantic schema for sample verification
    if not df_final.empty:
        sample_dict = df_final.iloc[0].to_dict()
        m = Movie(**sample_dict)
        assert m.tmdb_id == df_final.iloc[0]["tmdb_id"]

    # Save movies.parquet
    output_movies_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_movies_path, index=False)
    print(f"\nSaved clean movies dataset ({len(df_final):,} rows) to: {output_movies_path}")

    # Process MovieLens join
    print(f"\nProcessing MovieLens links join from: {links_csv_path}")
    links_df = pd.read_csv(links_csv_path)

    # Facts: drop 8 missing tmdbId, resolve 1 duplicate tmdbId
    initial_links_cnt = len(links_df)
    links_df = links_df.dropna(subset=["tmdbId"]).copy()
    missing_dropped = initial_links_cnt - len(links_df)
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)

    # Resolve duplicates on tmdbId by keeping first
    dup_links_cnt = links_df.duplicated(subset=["tmdbId"]).sum()
    links_df = links_df.drop_duplicates(subset=["tmdbId"], keep="first").reset_index(drop=True)

    # Join with clean movies
    merged_links = links_df.merge(
        df_final[["tmdb_id", "language", "title", "release_year"]],
        left_on="tmdbId",
        right_on="tmdb_id",
        how="inner",
    )

    output_links_path.parent.mkdir(parents=True, exist_ok=True)
    merged_links.to_parquet(output_links_path, index=False)
    print(f"Saved MovieLens links join ({len(merged_links):,} rows) to: {output_links_path}")

    # Overlap count per language
    overlap_per_lang = merged_links["language"].value_counts().to_dict()

    # Print FUNNEL
    print("\n" + "=" * 78)
    print("FILTERING FUNNEL (Rows per language after each step)")
    print("=" * 78)
    header = f"{'Step':<50} | {'en':>8} | {'hi':>8} | {'mr':>8} | {'Total':>8}"
    print(header)
    print("-" * len(header))
    for step_idx in range(7):
        name = step_names[step_idx]
        en_c = funnel_counts[step_idx]["en"]
        hi_c = funnel_counts[step_idx]["hi"]
        mr_c = funnel_counts[step_idx]["mr"]
        tot = en_c + hi_c + mr_c
        print(f"{name:<50} | {en_c:>8,} | {hi_c:>8,} | {mr_c:>8,} | {tot:>8,}")

    # Print REPORT SUMMARY
    print("\n" + "=" * 78)
    print("FINAL REPORT SUMMARY")
    print("=" * 78)

    print("\n1. Final Counts per Language:")
    for lang in ["en", "hi", "mr"]:
        cnt = funnel_counts[6][lang]
        print(f"   - {lang}: {cnt:,} rows")

    print("\n2. MovieLens Overlap Counts per Language:")
    for lang in ["en", "hi", "mr"]:
        cnt = overlap_per_lang.get(lang, 0)
        print(f"   - {lang}: {cnt:,} matched in MovieLens links")

    print("\n3. Remaining Null % per Column in movies.parquet:")
    null_pcts = (df_final.isna().mean() * 100).to_dict()
    for col, null_p in null_pcts.items():
        print(f"   - {col:<18}: {null_p:6.2f}% null")

    # Title + Year duplicates check
    title_year_dups = df_final.duplicated(subset=["title", "release_year"]).sum()
    print(f"\n4. Title + Year duplicates remaining in processed dataset: {title_year_dups:,}")

    # Check mr threshold condition
    mr_cnt = funnel_counts[6]["mr"]
    if mr_cnt < 150:
        print(f"\n[WARNING / ALERT]: Marathi (mr) dataset count ({mr_cnt}) is below 150 rows!")

    return df_final


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "TMDB_movie_dataset_v11.csv"
    if not raw_csv.exists():
        raw_csv = Path(r"C:\Users\RUTUJA\Downloads\archive (2)\TMDB_movie_dataset_v11.csv")

    links_csv = project_root / "data" / "raw" / "ml-latest-small" / "links.csv"
    movies_out = project_root / "data" / "processed" / "movies.parquet"
    links_out = project_root / "data" / "processed" / "ml_links.parquet"

    run_pipeline(raw_csv, links_csv, movies_out, links_out)
