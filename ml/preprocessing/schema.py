"""
schema.py
---------
Pydantic Movie model matching dataset structure.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Movie(BaseModel):
    tmdb_id: int
    title: str
    original_title: str
    language: str
    overview: str
    tagline: Optional[str] = None
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    release_year: int
    runtime: float
    vote_average: float
    vote_count: int
    popularity: float
    poster_path: str
    backdrop_path: Optional[str] = None
    imdb_id: Optional[str] = None

    # Cast and director fields reserved for future enrichment tasks
    cast: Optional[list[str]] = None
    director: Optional[str] = None
