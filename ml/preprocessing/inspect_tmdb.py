"""
inspect_tmdb.py
---------------
Phase 1 - Task 1: Inspect the raw TMDB Kaggle CSV.

Reads the file in chunks for memory safety (636 MB on disk).
Produces a structured report saved to data/processed/tmdb_inspection_report.md
and printed to stdout.

Run from the project root:
    python -m ml.preprocessing.inspect_tmdb
"""

from __future__ import annotations

import os
import sys
import json
import random
import textwrap
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Locate project root: two levels up from this file (ml/preprocessing/inspect_tmdb.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TMDB CSV: prefer data/raw/ symlink, fall back to Downloads location
_TMDB_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "TMDB_movie_dataset_v11.csv",
    Path(r"C:\Users\RUTUJA\Downloads\archive (2)\TMDB_movie_dataset_v11.csv"),
]

TMDB_CSV: Path | None = next((p for p in _TMDB_CANDIDATES if p.exists()), None)

REPORT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = REPORT_DIR / "tmdb_inspection_report.md"

# Chunk size for memory-safe reading (~100 k rows at a time)
CHUNK_SIZE = 100_000

# Languages of interest for detailed analysis
FOCUS_LANGS = ("en", "hi", "mr")

# Columns that commonly contain JSON / pipe-separated lists in TMDB exports
LIST_LIKE_HINTS = {
    "genres", "keywords", "production_companies", "production_countries",
    "spoken_languages", "cast", "crew",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 78) -> str:
    return char * width


def _hdr(title: str, level: int = 2) -> str:
    prefix = "#" * level
    return f"\n{prefix} {title}\n"


def _absent(col: str) -> str:
    return f"  WARNING: Column `{col}` is absent in this dataset."


def _safe_sample(values: "pd.Series[Any]", n: int = 3, seed: int = 42) -> list[str]:
    """Return up to n non-null, non-empty sample values as strings."""
    non_empty = values.dropna()
    if non_empty.dtype == object:
        non_empty = non_empty[non_empty.str.strip() != ""]
    if len(non_empty) == 0:
        return []
    random.seed(seed)
    k = min(n, len(non_empty))
    idx = random.sample(range(len(non_empty)), k)
    return [str(non_empty.iloc[i])[:300] for i in idx]


# ---------------------------------------------------------------------------
# Pass-1: collect column names, dtypes, row count
# ---------------------------------------------------------------------------

def pass1_schema(csv_path: Path) -> tuple[int, list[str], dict[str, str]]:
    """
    Read only the header + count rows via chunked iteration.
    Returns (total_rows, columns, dtype_map).
    """
    total_rows = 0
    columns: list[str] = []
    dtype_votes: dict[str, Counter] = {}

    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="warn",
    )):
        if i == 0:
            columns = list(chunk.columns)
            for col in columns:
                dtype_votes[col] = Counter()
        total_rows += len(chunk)
        for col in columns:
            if col in chunk.columns:
                dtype_votes[col][str(chunk[col].dtype)] += 1

    dtype_map = {col: ctr.most_common(1)[0][0] for col, ctr in dtype_votes.items()}
    return total_rows, columns, dtype_map


# ---------------------------------------------------------------------------
# Pass-2: null counts, duplicate detection, language stats, per-lang details
# ---------------------------------------------------------------------------

def pass2_main(csv_path: Path, columns: list[str]) -> dict[str, Any]:
    """
    Single chunked pass collecting all aggregate statistics.
    """
    null_counts: Counter = Counter()
    id_values: list[Any] = []
    title_year_pairs: list[tuple[Any, Any]] = []
    lang_counter: Counter = Counter()

    # Per-language accumulators
    lang_rows: Counter = Counter()
    lang_nonempty_overview: Counter = Counter()
    lang_vote_ge1: Counter = Counter()
    lang_vote_ge5: Counter = Counter()
    lang_vote_ge10: Counter = Counter()
    lang_vote_ge50: Counter = Counter()
    lang_has_release_date: Counter = Counter()
    lang_decade: dict[str, Counter] = {lg: Counter() for lg in FOCUS_LANGS}
    lang_adult_true: Counter = Counter()
    lang_status_not_released: Counter = Counter()

    # Overview lengths per language (cap to avoid RAM blow-up)
    ov_lengths: dict[str, list[int]] = {lg: [] for lg in FOCUS_LANGS}
    OV_CAP = 200_000

    # Sample rows for mr and hi
    sample_rows: dict[str, list[dict[str, Any]]] = {"mr": [], "hi": []}
    SAMPLE_CAP = 200

    # Poster / backdrop availability
    poster_present = 0
    backdrop_present = 0
    total_rows_seen = 0

    # Detect available columns once
    has_id = "id" in columns
    has_title = "title" in columns
    has_release_date = "release_date" in columns
    has_lang = "original_language" in columns
    has_overview = "overview" in columns
    has_vote_count = "vote_count" in columns
    has_adult = "adult" in columns
    has_status = "status" in columns
    has_poster = "poster_path" in columns
    has_backdrop = "backdrop_path" in columns

    for chunk in pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="warn",
    ):
        total_rows_seen += len(chunk)

        # -- nulls --
        for col in columns:
            if col in chunk.columns:
                null_counts[col] += int(chunk[col].isna().sum())

        # -- id duplicate tracking --
        if has_id:
            id_values.extend(chunk["id"].dropna().tolist())

        # -- title+year duplicate tracking --
        if has_title and has_release_date:
            years = pd.to_datetime(chunk["release_date"], errors="coerce").dt.year
            pairs = list(zip(chunk["title"].fillna(""), years.fillna(-1).astype(int)))
            title_year_pairs.extend(pairs)

        # -- language distribution --
        if has_lang:
            lang_counter.update(chunk["original_language"].dropna().tolist())

        # -- poster / backdrop --
        if has_poster:
            poster_present += int(chunk["poster_path"].notna().sum())
        if has_backdrop:
            backdrop_present += int(chunk["backdrop_path"].notna().sum())

        # -- per-focus-language stats --
        if has_lang:
            for lg in FOCUS_LANGS:
                mask = chunk["original_language"] == lg
                sub = chunk[mask]
                n = len(sub)
                lang_rows[lg] += n

                if has_overview:
                    ov = sub["overview"].fillna("").astype(str).str.strip()
                    lang_nonempty_overview[lg] += int((ov != "").sum())
                    if len(ov_lengths[lg]) < OV_CAP:
                        ov_lengths[lg].extend(ov.apply(len).tolist())

                if has_vote_count:
                    vc = pd.to_numeric(sub["vote_count"], errors="coerce").fillna(0)
                    lang_vote_ge1[lg] += int((vc >= 1).sum())
                    lang_vote_ge5[lg] += int((vc >= 5).sum())
                    lang_vote_ge10[lg] += int((vc >= 10).sum())
                    lang_vote_ge50[lg] += int((vc >= 50).sum())

                if has_release_date:
                    rd = pd.to_datetime(sub["release_date"], errors="coerce")
                    lang_has_release_date[lg] += int(rd.notna().sum())
                    decades = (rd.dt.year // 10 * 10).dropna().astype(int)
                    lang_decade[lg].update(decades.tolist())

                if has_adult:
                    adult_col = sub["adult"].astype(str).str.strip().str.lower()
                    lang_adult_true[lg] += int((adult_col == "true").sum())

                if has_status:
                    stat = sub["status"].fillna("").astype(str).str.strip()
                    lang_status_not_released[lg] += int(
                        ((stat != "") & (stat != "Released")).sum()
                    )

                # Collect sample rows for mr / hi
                if lg in sample_rows and n > 0 and len(sample_rows[lg]) < SAMPLE_CAP:
                    sample_cols_avail = [
                        c for c in ["title", "overview", "release_date", "vote_count"]
                        if c in sub.columns
                    ]
                    sub_reset = sub[sample_cols_avail].reset_index(drop=True)
                    for idx in range(min(n, SAMPLE_CAP - len(sample_rows[lg]))):
                        row_dict: dict[str, Any] = {}
                        for c in sample_cols_avail:
                            row_dict[c] = sub_reset.at[idx, c]
                        sample_rows[lg].append(row_dict)

    return {
        "null_counts": null_counts,
        "id_values": id_values,
        "title_year_pairs": title_year_pairs,
        "lang_counter": lang_counter,
        "lang_rows": lang_rows,
        "lang_nonempty_overview": lang_nonempty_overview,
        "lang_vote_ge1": lang_vote_ge1,
        "lang_vote_ge5": lang_vote_ge5,
        "lang_vote_ge10": lang_vote_ge10,
        "lang_vote_ge50": lang_vote_ge50,
        "lang_has_release_date": lang_has_release_date,
        "lang_decade": lang_decade,
        "lang_adult_true": lang_adult_true,
        "lang_status_not_released": lang_status_not_released,
        "ov_lengths": ov_lengths,
        "sample_rows": sample_rows,
        "poster_present": poster_present,
        "backdrop_present": backdrop_present,
        "total_rows_seen": total_rows_seen,
        "has_id": has_id,
        "has_title": has_title,
        "has_release_date": has_release_date,
        "has_lang": has_lang,
        "has_overview": has_overview,
        "has_vote_count": has_vote_count,
        "has_adult": has_adult,
        "has_status": has_status,
        "has_poster": has_poster,
        "has_backdrop": has_backdrop,
    }


# ---------------------------------------------------------------------------
# Pass-3: list/JSON-like column samples (first 5 chunks only)
# ---------------------------------------------------------------------------

def pass3_list_columns(csv_path: Path, columns: list[str]) -> dict[str, list[str]]:
    """
    Detect columns that look like JSON arrays or pipe-separated lists.
    Reads only the first 5 chunks for speed.
    Returns {col: [sample1, sample2, sample3]}.
    """
    candidates: set[str] = set()
    first_chunk: pd.DataFrame | None = None

    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="warn",
    )):
        if i == 0:
            first_chunk = chunk
            for col in columns:
                if col not in chunk.columns:
                    continue
                sample_vals = chunk[col].dropna().astype(str).head(200)
                for val in sample_vals:
                    stripped = val.strip()
                    if (
                        (stripped.startswith("[") and stripped.endswith("]"))
                        or (stripped.startswith("{") and stripped.endswith("}"))
                        or "|" in stripped
                        or col.lower() in LIST_LIKE_HINTS
                    ):
                        candidates.add(col)
                        break
        if i >= 4:
            break

    result: dict[str, list[str]] = {}
    if first_chunk is not None:
        for col in sorted(candidates):
            if col in first_chunk.columns:
                result[col] = _safe_sample(first_chunk[col], n=3)
    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    csv_path: Path,
    file_size_bytes: int,
    total_rows: int,
    columns: list[str],
    dtype_map: dict[str, str],
    pass2: dict[str, Any],
    list_cols: dict[str, list[str]],
) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    null_counts = pass2["null_counts"]
    lang_counter = pass2["lang_counter"]
    total = total_rows

    # -----------------------------------------------------------------------
    add("# TMDB Dataset Inspection Report")
    add()
    add("> Generated by `ml/preprocessing/inspect_tmdb.py`")
    add()

    # === Section 1: File Overview ===
    add(_hdr("1. File Overview", 2))
    add("| Field | Value |")
    add("|-------|-------|")
    add(f"| Filename | `{csv_path.name}` |")
    add(f"| Full path | `{csv_path}` |")
    add(f"| File size | {file_size_bytes:,} bytes  ({file_size_bytes/1_048_576:.1f} MB) |")
    add(f"| Total rows | {total:,} |")
    add(f"| Total columns | {len(columns)} |")
    add()
    add("**Columns and inferred dtypes:**")
    add()
    add("| # | Column | Dtype |")
    add("|---|--------|-------|")
    for i, col in enumerate(columns, 1):
        add(f"| {i} | `{col}` | `{dtype_map.get(col, 'unknown')}` |")

    # === Section 2: Null % per column ===
    add(_hdr("2. Null Percentage per Column", 2))
    add("| Column | Null Count | Null % |")
    add("|--------|-----------|--------|")
    for col in columns:
        nc = null_counts.get(col, 0)
        pct = 100.0 * nc / total if total > 0 else 0.0
        add(f"| `{col}` | {nc:,} | {pct:.2f}% |")

    # === Section 3: Duplicates ===
    add(_hdr("3. Duplicate Analysis", 2))

    if pass2["has_id"]:
        id_vals = pass2["id_values"]
        id_total = len(id_vals)
        id_unique = len(set(id_vals))
        id_dups = id_total - id_unique
        add("**By `id` column:**")
        add(f"- Total id values (non-null): {id_total:,}")
        add(f"- Unique ids: {id_unique:,}")
        add(f"- Duplicate id rows (extra occurrences): {id_dups:,}")
    else:
        add(_absent("id"))

    add()

    if pass2["has_title"] and pass2["has_release_date"]:
        pairs = pass2["title_year_pairs"]
        pair_total = len(pairs)
        pair_unique = len(set(pairs))
        pair_dups = pair_total - pair_unique
        add("**By `title + year` (year extracted from `release_date`):**")
        add(f"- Total (title, year) pairs: {pair_total:,}")
        add(f"- Unique (title, year) pairs: {pair_unique:,}")
        add(f"- Duplicate (title, year) rows (extra occurrences): {pair_dups:,}")
    else:
        add("  WARNING: Cannot compute title+year duplicates: `title` or `release_date` column absent.")

    # === Section 4: List/JSON-like columns ===
    add(_hdr("4. List / JSON-like Columns", 2))
    if list_cols:
        for col, samples in list_cols.items():
            add(f"**`{col}`** — {len(samples)} sample(s):")
            for j, s in enumerate(samples, 1):
                display = s[:250] + ("..." if len(s) > 250 else "")
                add(f"  - Sample {j}: `{display}`")
            add()
    else:
        add("No list/JSON-like columns detected in sampled rows.")

    # === Section 5: original_language distribution ===
    add(_hdr("5. `original_language` Distribution", 2))
    if pass2["has_lang"]:
        top20 = lang_counter.most_common(20)
        add("**Top 20 languages:**")
        add()
        add("| Rank | Language | Count |")
        add("|------|----------|-------|")
        for rank, (lang, cnt) in enumerate(top20, 1):
            add(f"| {rank} | `{lang}` | {cnt:,} |")
        add()
        add("**Exact counts for focus languages:**")
        add()
        add("| Language | Count |")
        add("|----------|-------|")
        for lg in ("en", "hi", "mr"):
            cnt = lang_counter.get(lg, 0)
            add(f"| `{lg}` | {cnt:,} |")
    else:
        add(_absent("original_language"))

    # === Section 6: Per-language detailed stats ===
    add(_hdr("6. Per-Language Detailed Statistics (en / hi / mr)", 2))

    if not pass2["has_lang"]:
        add(_absent("original_language"))
    else:
        for lg in FOCUS_LANGS:
            n_rows = pass2["lang_rows"][lg]
            add(f"### Language: `{lg}`")
            add()
            add("| Metric | Value |")
            add("|--------|-------|")
            add(f"| Total rows | {n_rows:,} |")

            if pass2["has_overview"]:
                ov_cnt = pass2["lang_nonempty_overview"][lg]
                ov_pct = 100.0 * ov_cnt / n_rows if n_rows else 0.0
                add(f"| Non-empty overview | {ov_cnt:,} ({ov_pct:.1f}%) |")
            else:
                add("| Non-empty overview | `overview` column absent |")

            if pass2["has_vote_count"]:
                add(f"| vote_count >= 1 | {pass2['lang_vote_ge1'][lg]:,} |")
                add(f"| vote_count >= 5 | {pass2['lang_vote_ge5'][lg]:,} |")
                add(f"| vote_count >= 10 | {pass2['lang_vote_ge10'][lg]:,} |")
                add(f"| vote_count >= 50 | {pass2['lang_vote_ge50'][lg]:,} |")
            else:
                add("| vote_count thresholds | `vote_count` column absent |")

            if pass2["has_release_date"]:
                rd_cnt = pass2["lang_has_release_date"][lg]
                rd_pct = 100.0 * rd_cnt / n_rows if n_rows else 0.0
                add(f"| Has release_date | {rd_cnt:,} ({rd_pct:.1f}%) |")
            else:
                add("| Has release_date | `release_date` column absent |")

            if pass2["has_adult"]:
                adult_cnt = pass2["lang_adult_true"][lg]
                add(f"| adult = True | {adult_cnt:,} |")
            else:
                add("| adult = True | `adult` column absent |")

            if pass2["has_status"]:
                snr_cnt = pass2["lang_status_not_released"][lg]
                add(f"| status != Released | {snr_cnt:,} |")
            else:
                add("| status != Released | `status` column absent |")

            add()

            # Rows per decade
            decade_counter = pass2["lang_decade"].get(lg, {})
            if pass2["has_release_date"] and decade_counter:
                add(f"**Rows per decade (`{lg}`):**")
                add()
                add("| Decade | Count |")
                add("|--------|-------|")
                for decade in sorted(decade_counter.keys()):
                    add(f"| {decade}s | {decade_counter[decade]:,} |")
                add()

    # === Section 7: Random sample rows for mr and hi ===
    add(_hdr("7. Random Sample Rows (mr and hi, 5 each)", 2))

    for lg in ("mr", "hi"):
        pool = pass2["sample_rows"].get(lg, [])
        n_pool = len(pool)
        add(f"### Language: `{lg}` (pool size: {n_pool})")
        if n_pool == 0:
            add(f"  No rows found for language `{lg}`.")
            add()
            continue
        random.seed(99)
        chosen = random.sample(pool, min(5, n_pool))
        for i, row in enumerate(chosen, 1):
            title = row.get("title", "N/A")
            overview = str(row.get("overview", "")).strip()
            ov_display = (overview[:200] + "...") if len(overview) > 200 else (overview or "---")
            release = str(row.get("release_date", "")).strip() or "N/A"
            year = release[:4] if len(release) >= 4 else "N/A"
            vote = row.get("vote_count", "N/A")
            add(f"**Row {i}:**")
            add(f"- **title**: {title}")
            add(f"- **year**: {year}")
            add(f"- **vote_count**: {vote}")
            add(f"- **overview**: {ov_display}")
            add()

    # === Section 8: Overview length statistics per language ===
    add(_hdr("8. Overview Length Statistics (chars) per Language", 2))

    if not pass2["has_overview"]:
        add(_absent("overview"))
    else:
        add("| Language | Count (non-zero len) | Median | p10 | p90 |")
        add("|----------|--------------------|--------|-----|-----|")
        for lg in FOCUS_LANGS:
            lengths = [l for l in pass2["ov_lengths"][lg] if l > 0]
            if lengths:
                arr = np.array(lengths, dtype=float)
                median = float(np.median(arr))
                p10 = float(np.percentile(arr, 10))
                p90 = float(np.percentile(arr, 90))
                add(f"| `{lg}` | {len(lengths):,} | {median:.0f} | {p10:.0f} | {p90:.0f} |")
            else:
                add(f"| `{lg}` | 0 | --- | --- | --- |")

    # === Section 9: Poster / backdrop path availability ===
    add(_hdr("9. Poster & Backdrop Path Availability", 2))

    add("| Column | Present (non-null) | % of total rows |")
    add("|--------|--------------------|-----------------|")
    if pass2["has_poster"]:
        pp = pass2["poster_present"]
        pp_pct = 100.0 * pp / total if total else 0.0
        add(f"| `poster_path` | {pp:,} | {pp_pct:.2f}% |")
    else:
        add("| `poster_path` | --- | Column absent |")

    if pass2["has_backdrop"]:
        bp = pass2["backdrop_present"]
        bp_pct = 100.0 * bp / total if total else 0.0
        add(f"| `backdrop_path` | {bp:,} | {bp_pct:.2f}% |")
    else:
        add("| `backdrop_path` | --- | Column absent |")

    add()
    add(_sep())
    add()
    add("_End of report._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if TMDB_CSV is None:
        print(
            "ERROR: TMDB CSV not found. Expected locations:\n"
            + "\n".join(f"  {p}" for p in _TMDB_CANDIDATES),
            file=sys.stderr,
        )
        sys.exit(1)

    file_size = TMDB_CSV.stat().st_size

    print("=" * 60)
    print("TMDB Inspection")
    print("=" * 60)
    print(f"File : {TMDB_CSV}")
    print(f"Size : {file_size:,} bytes ({file_size / 1_048_576:.1f} MB)")
    print(f"Chunk: {CHUNK_SIZE:,} rows")
    print()

    # --- Pass 1: schema ---
    print("[1/3] Pass 1: reading schema & counting rows ...")
    total_rows, columns, dtype_map = pass1_schema(TMDB_CSV)
    print(f"      -> {total_rows:,} rows, {len(columns)} columns")

    # --- Pass 2: main stats ---
    print("[2/3] Pass 2: collecting null/duplicate/language/per-lang stats ...")
    p2 = pass2_main(TMDB_CSV, columns)
    print(f"      -> done (rows verified: {p2['total_rows_seen']:,})")

    # --- Pass 3: list columns ---
    print("[3/3] Pass 3: detecting list/JSON-like columns (first 5 chunks) ...")
    list_cols = pass3_list_columns(TMDB_CSV, columns)
    print(f"      -> {len(list_cols)} list-like columns found: {list(list_cols.keys())}")

    # --- Build report ---
    print("\nBuilding report ...")
    report_md = build_report(
        csv_path=TMDB_CSV,
        file_size_bytes=file_size,
        total_rows=total_rows,
        columns=columns,
        dtype_map=dtype_map,
        pass2=p2,
        list_cols=list_cols,
    )

    # --- Save report ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"\nReport saved -> {REPORT_PATH}")

    # --- Print report ---
    print("\n" + "=" * 78)
    print(report_md)
    print("=" * 78)


if __name__ == "__main__":
    main()
