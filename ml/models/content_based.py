"""Dense Content-Based Recommender using Multilingual E5 embeddings & metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = DATA_DIR / "embeddings"


class DenseContentRecommender:
    """
    Content-Based Recommender powered by 768-dimensional multilingual E5 embeddings
    combined with structured metadata (genres, director, language).
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        embeddings: np.ndarray | None = None,
        use_mmap: bool = False,
    ):
        if df is None:
            enriched_path = DATA_DIR / "movies_enriched.parquet"
            if enriched_path.exists():
                self.df = pd.read_parquet(enriched_path).reset_index(drop=True)
            else:
                self.df = pd.read_parquet(DATA_DIR / "movies.parquet").reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        if embeddings is None:
            emb_path = EMB_DIR / "emb_e5_multilingual.npy"
            if use_mmap:
                self.embeddings = np.load(emb_path, mmap_mode="r")
            else:
                self.embeddings = np.load(emb_path)
        else:
            self.embeddings = embeddings

        # Normalize embeddings if not already normalized
        norms = np.linalg.norm(self.embeddings[:5], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            norm_all = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norm_all[norm_all == 0] = 1.0
            self.embeddings = self.embeddings / norm_all

        # Prebuild fast lookup dictionaries
        self.tmdb_to_idx: dict[int, int] = {
            int(tmdb_id): idx for idx, tmdb_id in enumerate(self.df["tmdb_id"].values)
        }

        # Title lookup: if multiple movies share a title, pick the one with most votes
        self.title_to_idx: dict[str, int] = {}
        sorted_by_votes = self.df.sort_values("vote_count", ascending=False)
        for idx, row in sorted_by_votes.iterrows():
            t_lower = str(row["title"]).strip().lower()
            if t_lower and t_lower not in self.title_to_idx:
                self.title_to_idx[t_lower] = idx

        # Pre-process genres as sets of strings for Jaccard calculation
        self._genre_sets: list[set[str]] = [
            set(g) if isinstance(g, (list, np.ndarray)) else set()
            for g in self.df["genres"].values
        ]

        # Pre-process directors as lower-case sets
        self._director_sets: list[set[str]] = [
            {d.strip().lower() for d in str(d_str).split(",")} if pd.notna(d_str) else set()
            for d_str in self.df["director"].values
        ]

    def find(self, identifier: str | int) -> int | None:
        """Find the internal row index given a TMDB ID or Movie Title."""
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            return self.tmdb_to_idx.get(int(identifier))
        title_str = str(identifier).strip().lower()
        return self.title_to_idx.get(title_str)

    def get_movie_by_idx(self, idx: int) -> dict[str, Any]:
        """Return movie metadata dictionary for a given internal row index."""
        row = self.df.iloc[idx]
        return {
            "tmdb_id": int(row["tmdb_id"]),
            "title": str(row["title"]),
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "language": str(row["language"]),
            "genres": list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
            "director": str(row["director"]) if pd.notna(row["director"]) else None,
            "vote_average": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
            "vote_count": int(row["vote_count"]) if pd.notna(row["vote_count"]) else 0,
            "overview": str(row.get("overview", "")),
            "poster_path": str(row.get("poster_path", "")),
        }

    def similar(
        self,
        identifier: str | int,
        k: int = 10,
        language: str | None = None,
        min_rating: float | None = None,
        genre_weight: float = 0.15,
        director_bonus: float = 0.05,
    ) -> pd.DataFrame:
        """
        Find top-K movies most similar to a query movie using dense embedding similarity
        plus optional genre Jaccard overlap and director matching.
        """
        target_idx = self.find(identifier)
        if target_idx is None:
            return pd.DataFrame()

        target_vec = self.embeddings[target_idx]
        target_genres = self._genre_sets[target_idx]
        target_directors = self._director_sets[target_idx]

        # Base dense cosine similarity (20,273 values in ~5ms)
        scores = self.embeddings @ target_vec

        # Metadata adjustments if requested
        if genre_weight > 0 or director_bonus > 0:
            scores = scores.copy()
            if genre_weight > 0 and len(target_genres) > 0:
                for i, g_set in enumerate(self._genre_sets):
                    if g_set:
                        intersection = len(target_genres & g_set)
                        union = len(target_genres | g_set)
                        jaccard = intersection / union if union > 0 else 0.0
                        scores[i] += genre_weight * jaccard

            if director_bonus > 0 and len(target_directors) > 0:
                for i, d_set in enumerate(self._director_sets):
                    if d_set and (target_directors & d_set):
                        scores[i] += director_bonus

        # Exclude the query movie itself
        scores[target_idx] = -1.0

        # Optional filters
        mask = np.ones(len(self.df), dtype=bool)
        mask[target_idx] = False

        if language:
            mask &= (self.df["language"].values == language)
        if min_rating is not None:
            mask &= (self.df["vote_average"].values >= min_rating)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame()

        valid_scores = scores[valid_indices]
        top_k_in_valid = valid_scores.argsort()[::-1][:k]
        top_indices = valid_indices[top_k_in_valid]

        out = self.df.loc[
            top_indices,
            ["tmdb_id", "title", "language", "release_year", "vote_average", "vote_count", "genres", "director"],
        ].copy()
        out["score"] = scores[top_indices].round(4)
        return out.reset_index(drop=True)

    def recommend_from_profile(
        self,
        liked_identifiers: list[str | int],
        weights: list[float] | None = None,
        k: int = 10,
        language: str | None = None,
        min_rating: float | None = None,
    ) -> pd.DataFrame:
        """
        Recommend movies for a user by aggregating their liked movies into a user taste centroid vector.
        """
        indices: list[int] = []
        clean_weights: list[float] = []

        for i, item in enumerate(liked_identifiers):
            idx = self.find(item)
            if idx is not None:
                indices.append(idx)
                w = weights[i] if (weights is not None and i < len(weights)) else 1.0
                clean_weights.append(max(0.1, float(w)))

        if not indices:
            return pd.DataFrame()

        # Compute weighted user centroid
        vectors = self.embeddings[indices]
        w_arr = np.array(clean_weights, dtype=np.float32)[:, np.newaxis]
        user_vector = np.sum(vectors * w_arr, axis=0)
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector /= norm

        scores = self.embeddings @ user_vector

        # Mask out already consumed movies
        mask = np.ones(len(self.df), dtype=bool)
        for idx in indices:
            mask[idx] = False

        if language:
            mask &= (self.df["language"].values == language)
        if min_rating is not None:
            mask &= (self.df["vote_average"].values >= min_rating)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame()

        valid_scores = scores[valid_indices]
        top_k_in_valid = valid_scores.argsort()[::-1][:k]
        top_indices = valid_indices[top_k_in_valid]

        out = self.df.loc[
            top_indices,
            ["tmdb_id", "title", "language", "release_year", "vote_average", "vote_count", "genres", "director"],
        ].copy()
        out["score"] = scores[top_indices].round(4)
        return out.reset_index(drop=True)
