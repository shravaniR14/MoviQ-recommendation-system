"""
fetch_credits.py
----------------
Fetch top 5 cast names and director(s) for TMDB movies using TMDB API (/movie/{id}/credits).

Features:
- ThreadPoolExecutor for fast concurrent fetching (--workers, default 5).
- Thread-safe caching: Main thread writes to credits_cache.jsonl sequentially.
- Resumable: Skips tmdb_ids already present in cache.
- Retries: Exponential backoff on 429 and 5xx response codes.
- CLI: Supports --limit N and --workers N flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fetch_credits")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_PARQUET = PROJECT_ROOT / "data" / "processed" / "movies.parquet"
CREDITS_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "credits_cache.jsonl"
ENV_PATH = PROJECT_ROOT / ".env"

TMDB_API_BASE = "https://api.themoviedb.org/3"


def load_api_key() -> str:
    """Load TMDB_API_KEY from .env or environment."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    return api_key


def fetch_movie_credits(
    tmdb_id: int, api_key: str, max_retries: int = 5, backoff_factor: float = 1.0
) -> dict[str, Any]:
    """
    Fetch credits for a single movie tmdb_id from TMDB API.
    Returns dict with keys: tmdb_id, cast, director, status, error.
    """
    url = f"{TMDB_API_BASE}/movie/{tmdb_id}/credits"
    headers = {}
    params = {}

    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                cast_raw = data.get("cast", [])
                crew_raw = data.get("crew", [])

                cast_sorted = sorted(cast_raw, key=lambda x: x.get("order", 999))
                cast_names = [member.get("name", "").strip() for member in cast_sorted if member.get("name")][:5]

                directors = [
                    member.get("name", "").strip()
                    for member in crew_raw
                    if member.get("job") == "Director" and member.get("name")
                ]
                director_str = ", ".join(directors) if directors else None

                return {
                    "tmdb_id": tmdb_id,
                    "cast": cast_names,
                    "director": director_str,
                    "status": "success",
                    "error": None,
                }
            elif resp.status_code == 404:
                logger.warning(f"TMDB ID {tmdb_id} not found (404).")
                return {
                    "tmdb_id": tmdb_id,
                    "cast": [],
                    "director": None,
                    "status": "not_found",
                    "error": "404 Not Found",
                }
            elif resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = int(retry_after)
                else:
                    sleep_time = backoff_factor * (2 ** attempt)
                logger.warning(
                    f"HTTP {resp.status_code} for tmdb_id {tmdb_id}. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"HTTP {resp.status_code} for tmdb_id {tmdb_id}: {resp.text[:200]}")
                return {
                    "tmdb_id": tmdb_id,
                    "cast": [],
                    "director": None,
                    "status": "error",
                    "error": f"HTTP {resp.status_code}",
                }
        except requests.RequestException as e:
            sleep_time = backoff_factor * (2 ** attempt)
            logger.warning(f"Request exception for tmdb_id {tmdb_id}: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    return {
        "tmdb_id": tmdb_id,
        "cast": [],
        "director": None,
        "status": "error",
        "error": "Max retries exceeded",
    }


def load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    """Load cached credits from jsonl file into dict tmdb_id -> record."""
    cache: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tmdb_id = int(record["tmdb_id"])
                cache[tmdb_id] = record
            except Exception as e:
                logger.warning(f"Failed to parse cache line: {line[:100]} ({e})")
    return cache


def main(limit: Optional[int] = None, workers: int = 5):
    api_key = load_api_key()
    if not api_key:
        logger.error(
            "TMDB_API_KEY is missing! Please set TMDB_API_KEY in your .env file or environment variables."
        )
        sys.exit(1)

    if not MOVIES_PARQUET.exists():
        logger.error(f"Movies file not found: {MOVIES_PARQUET}")
        sys.exit(1)

    df_movies = pd.read_parquet(MOVIES_PARQUET)
    all_ids = df_movies["tmdb_id"].dropna().astype(int).tolist()
    logger.info(f"Loaded {len(all_ids):,} total movie IDs from {MOVIES_PARQUET}")

    cache = load_cache(CREDITS_CACHE_PATH)
    logger.info(f"Loaded {len(cache):,} cached records from {CREDITS_CACHE_PATH}")

    pending_ids = [tid for tid in all_ids if tid not in cache]
    logger.info(f"{len(pending_ids):,} IDs remaining to fetch.")

    if limit is not None:
        pending_ids = pending_ids[:limit]
        logger.info(f"Applying --limit {limit}: processing {len(pending_ids)} IDs.")

    if not pending_ids:
        logger.info("No pending IDs to fetch.")
        return

    CREDITS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    success_cnt = 0
    error_cnt = 0
    start_time = time.time()

    logger.info(f"Starting fetch with ThreadPoolExecutor (workers={workers})...")

    with open(CREDITS_CACHE_PATH, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(fetch_movie_credits, tmdb_id, api_key): tmdb_id
                for tmdb_id in pending_ids
            }

            completed_cnt = 0
            for future in as_completed(future_to_id):
                completed_cnt += 1
                record = future.result()

                # Main thread writes sequentially to cache file
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                if record["status"] == "success":
                    success_cnt += 1
                else:
                    error_cnt += 1

                if completed_cnt % 5 == 0 or completed_cnt == len(pending_ids):
                    elapsed_so_far = time.time() - start_time
                    rate = completed_cnt / elapsed_so_far if elapsed_so_far > 0 else 0.0
                    logger.info(
                        f"Progress [{completed_cnt}/{len(pending_ids)}] ({rate:.1f} req/s) - "
                        f"tmdb_id: {record['tmdb_id']} - Cast: {record.get('cast')} - Director: {record.get('director')}"
                    )

    elapsed_total = time.time() - start_time
    avg_req_per_sec = len(pending_ids) / elapsed_total if elapsed_total > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Fetch completed for batch in {elapsed_total:.2f} seconds ({avg_req_per_sec:.2f} req/s).")
    logger.info(f"Successes: {success_cnt}, Errors/Not Found: {error_cnt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TMDB credits for movies dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of movies to fetch.")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads (default 5).")
    args = parser.parse_args()

    main(limit=args.limit, workers=args.workers)
