"""
config.py
---------
Configuration parameters and thresholds for dataset preprocessing and cleaning.
"""

from __future__ import annotations
import datetime

# Language-specific minimum vote count thresholds
MIN_VOTE_COUNT: dict[str, int] = {
    "en": 50,
    "hi": 5,
    "mr": 1,
}

TARGET_LANGUAGES: set[str] = {"en", "hi", "mr"}

# Filtering rules
REQUIRED_STATUS: str = "Released"
ALLOW_ADULT: bool = False

REQUIRE_OVERVIEW: bool = True
REQUIRE_RELEASE_DATE: bool = True
REQUIRE_POSTER_PATH: bool = True

MIN_RUNTIME: float = 0.0  # runtime > 0
MIN_YEAR: int = 1900
MAX_YEAR: int = datetime.datetime.now().year  # 2026

MIN_VOTE_AVERAGE: float = 0.0
MAX_VOTE_AVERAGE: float = 10.0

# CSV columns to read
USE_COLS: list[str] = [
    "id",
    "title",
    "original_title",
    "original_language",
    "overview",
    "tagline",
    "genres",
    "keywords",
    "release_date",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "poster_path",
    "backdrop_path",
    "imdb_id",
    "status",
    "adult",
]

CHUNK_SIZE: int = 100_000
