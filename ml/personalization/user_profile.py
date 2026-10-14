"""
user_profile.py
---------------
Dynamic user profile tracker for Moviq.
Tracks user ratings, watched history, search queries, genre/language affinities,
and maintains an evolving 768-dimensional semantic taste centroid.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd


class UserProfile:
    """Represents a single user's dynamic taste profile, preferences, and interaction history."""

    def __init__(self, user_id: str | int):
        self.user_id = str(user_id)
        self.ratings: dict[int, float] = {}  # tmdb_id -> rating (0.5 - 5.0)
        self.watched: set[int] = set()       # set of tmdb_ids consumed
        self.searches: list[str] = []        # history of search query strings
        self.genre_scores: dict[str, float] = {}  # genre -> accumulated affinity score
        self.language_scores: dict[str, float] = {}  # language -> affinity score
        self.taste_embedding: np.ndarray | None = None  # (768,) float32 normalized vector
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    @property
    def is_cold_start(self) -> bool:
        """True if the user has no explicit ratings and no watch history."""
        return len(self.ratings) == 0 and len(self.watched) == 0

    @property
    def maturity(self) -> str:
        """
        User maturity stage:
        - 'cold': no ratings and no watches (relies heavily on query/content/popularity)
        - 'nascent': 1 to 4 ratings/watches (blended content + CF fold-in)
        - 'warm': 5+ ratings/watches (strong collaborative filtering signals)
        """
        interaction_count = len(self.ratings) + len(self.watched)
        if interaction_count == 0:
            return "cold"
        elif interaction_count < 5:
            return "nascent"
        return "warm"

    def update_rating(
        self,
        tmdb_id: int,
        rating: float,
        genres: list[str] | np.ndarray | None = None,
        language: str | None = None,
        movie_embedding: np.ndarray | None = None,
    ) -> None:
        """Record an explicit rating and update genre, language, and embedding taste vector."""
        tmdb_id = int(tmdb_id)
        rating = float(np.clip(rating, 0.5, 5.0))
        self.ratings[tmdb_id] = rating
        self.watched.add(tmdb_id)
        self.updated_at = time.time()

        # Centered rating weight: 5.0 -> +2.5, 3.0 -> +0.5, 1.0 -> -1.5
        weight = rating - 2.5

        # Update genre affinity
        if genres is not None:
            for g in genres:
                g_str = str(g).strip().title()
                if g_str:
                    self.genre_scores[g_str] = max(
                        0.0,
                        self.genre_scores.get(g_str, 0.0) + weight,
                    )

        # Update language affinity if positive rating
        if language and weight > 0:
            lang_str = str(language).strip().lower()
            self.language_scores[lang_str] = self.language_scores.get(lang_str, 0.0) + weight

        # Update taste centroid if positive rating and embedding is provided
        if movie_embedding is not None and weight > 0:
            self._blend_embedding(movie_embedding, weight=weight)

    def update_watch(
        self,
        tmdb_id: int,
        genres: list[str] | np.ndarray | None = None,
        language: str | None = None,
        movie_embedding: np.ndarray | None = None,
    ) -> None:
        """Record an implicit movie view/watch."""
        tmdb_id = int(tmdb_id)
        self.watched.add(tmdb_id)
        self.updated_at = time.time()

        # Implicit watch provides a moderate positive weight (+1.0)
        implicit_weight = 1.0

        if genres is not None:
            for g in genres:
                g_str = str(g).strip().title()
                if g_str:
                    self.genre_scores[g_str] = self.genre_scores.get(g_str, 0.0) + implicit_weight

        if language:
            lang_str = str(language).strip().lower()
            self.language_scores[lang_str] = self.language_scores.get(lang_str, 0.0) + implicit_weight

        if movie_embedding is not None:
            self._blend_embedding(movie_embedding, weight=implicit_weight)

    def update_search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> None:
        """Record a search query string and softly incorporate query intent into taste profile."""
        query_clean = str(query).strip()
        if not query_clean:
            return

        self.searches.append(query_clean)
        # Keep maximum 50 recent searches
        if len(self.searches) > 50:
            self.searches.pop(0)

        self.updated_at = time.time()

        # If query embedding is available and user is cold, it can bootstrap initial taste vector
        if query_embedding is not None:
            if self.taste_embedding is None:
                q_norm = np.linalg.norm(query_embedding)
                if q_norm > 0:
                    self.taste_embedding = (query_embedding / q_norm).astype(np.float32)
            else:
                # Soft blending with current search intent
                self._blend_embedding(query_embedding, weight=0.3)

    def _blend_embedding(self, vec: np.ndarray, weight: float = 1.0) -> None:
        """Exponential moving average / weighted update of user's taste embedding vector."""
        vec = np.asarray(vec, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return
        unit_vec = vec / norm

        if self.taste_embedding is None:
            self.taste_embedding = unit_vec.copy()
        else:
            # Learning rate adaptive to maturity: cold/nascent users adapt faster than warm users
            alpha = 0.4 if self.maturity != "warm" else 0.2
            blended = (1.0 - alpha) * self.taste_embedding + alpha * (unit_vec * weight)
            blended_norm = np.linalg.norm(blended)
            if blended_norm > 0:
                self.taste_embedding = (blended / blended_norm).astype(np.float32)

    def get_top_genres(self, n: int = 5) -> list[str]:
        """Return the top-N preferred genres sorted by accumulated affinity."""
        sorted_genres = sorted(self.genre_scores.items(), key=lambda x: x[1], reverse=True)
        return [g for g, score in sorted_genres[:n] if score > 0]

    def get_preferred_languages(self, n: int = 2) -> list[str]:
        """Return top preferred languages."""
        sorted_langs = sorted(self.language_scores.items(), key=lambda x: x[1], reverse=True)
        return [lang for lang, score in sorted_langs[:n] if score > 0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize user profile to dictionary."""
        return {
            "user_id": self.user_id,
            "ratings": {str(k): float(v) for k, v in self.ratings.items()},
            "watched": list(self.watched),
            "searches": self.searches,
            "genre_scores": self.genre_scores,
            "language_scores": self.language_scores,
            "taste_embedding": self.taste_embedding.tolist() if self.taste_embedding is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Deserialize user profile from dictionary."""
        profile = cls(user_id=data["user_id"])
        profile.ratings = {int(k): float(v) for k, v in data.get("ratings", {}).items()}
        profile.watched = set(data.get("watched", []))
        profile.searches = list(data.get("searches", []))
        profile.genre_scores = dict(data.get("genre_scores", {}))
        profile.language_scores = dict(data.get("language_scores", {}))
        if data.get("taste_embedding"):
            profile.taste_embedding = np.array(data["taste_embedding"], dtype=np.float32)
        profile.created_at = data.get("created_at", time.time())
        profile.updated_at = data.get("updated_at", time.time())
        return profile


class UserProfileManager:
    """Manages creation, retrieval, and persistent storage of user profiles."""

    def __init__(self, storage_path: Path | str | None = None):
        self.profiles: dict[str, UserProfile] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            self.load()

    def get_or_create(self, user_id: str | int) -> UserProfile:
        """Retrieve existing profile or create a fresh one."""
        u_key = str(user_id)
        if u_key not in self.profiles:
            self.profiles[u_key] = UserProfile(user_id=u_key)
        return self.profiles[u_key]

    def save(self) -> None:
        """Save profiles to JSON file."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {uid: prof.to_dict() for uid, prof in self.profiles.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load(self) -> None:
        """Load profiles from JSON file."""
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.profiles = {uid: UserProfile.from_dict(d) for uid, d in raw.items()}
