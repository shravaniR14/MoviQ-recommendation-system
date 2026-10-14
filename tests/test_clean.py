"""
test_clean.py
-------------
Pytest unit tests for ml/preprocessing/clean.py functions.
"""

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.clean import (
    clean_text,
    parse_comma_list,
    normalize_genres,
    parse_year,
    deduplicate_by_id,
)


def test_clean_text():
    """Test text cleaning handles empty overviews, whitespace, None, and NaN."""
    assert clean_text("  Inception  ") == "Inception"
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""


def test_parse_year():
    """Test year parsing from release dates and handling of bad/out-of-range years."""
    assert parse_year("1995-11-22") == 1995
    assert parse_year("2020/05/10") == 2020
    assert parse_year("1984") == 1984
    assert parse_year("invalid_date") is None
    assert parse_year("") is None
    assert parse_year(None) is None
    assert parse_year(np.nan) is None


def test_normalize_genres():
    """Test malformed genre strings, whitespace, and case normalization."""
    assert normalize_genres("Action, , Comedy, Sci-Fi") == ["Action", "Comedy", "Sci-Fi"]
    assert normalize_genres("  drama , animation  ") == ["Drama", "Animation"]
    assert normalize_genres("") == []
    assert normalize_genres(None) == []
    assert normalize_genres(np.nan) == []


def test_parse_comma_list():
    """Test parsing list-like strings with commas."""
    assert parse_comma_list("tag1, tag2, tag3") == ["tag1", "tag2", "tag3"]
    assert parse_comma_list("tag1,,tag2") == ["tag1", "tag2"]
    assert parse_comma_list("") == []
    assert parse_comma_list(None) == []


def test_deduplicate_by_id():
    """Test deduplication by tmdb_id keeping the row with highest vote_count."""
    df = pd.DataFrame([
        {"tmdb_id": 101, "title": "Movie A", "vote_count_num": 50},
        {"tmdb_id": 101, "title": "Movie A (dup)", "vote_count_num": 500},
        {"tmdb_id": 102, "title": "Movie B", "vote_count_num": 10},
    ])

    dedup = deduplicate_by_id(df, id_col="tmdb_id", vote_col="vote_count_num")
    assert len(dedup) == 2
    
    # Verify the row for id 101 kept vote_count_num = 500
    row_101 = dedup[dedup["tmdb_id"] == 101].iloc[0]
    assert row_101["vote_count_num"] == 500
    assert row_101["title"] == "Movie A (dup)"
