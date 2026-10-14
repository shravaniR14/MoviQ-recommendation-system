"""
hybrid.py
---------
Hybrid Recommendation Engine for Moviq.
Combines:
1. Collaborative Filtering (Regularized SVD + Item-Item Cosine Similarity)
2. Dense Content-Based Filtering (Multilingual E5 embeddings + genre metadata)
3. Semantic Search (Qdrant vector store)
4. Dynamic User Profile Personalization (searches, ratings, watches, genre affinities)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from ml.models.collaborative import CollaborativeFilteringRecommender
from ml.models.content_based import DenseContentRecommender
from ml.personalization.user_profile import UserProfile, UserProfileManager


class HybridRecommender:
    """
    Multi-signal Hybrid Recommender integrating Collaborative Filtering,
    Dense Content Embeddings, and Semantic Vector Search with adaptive weighting.
    """

    def __init__(
        self,
        content_recommender: DenseContentRecommender | None = None,
        cf_recommender: CollaborativeFilteringRecommender | None = None,
        profile_manager: UserProfileManager | None = None,
        searcher: Any | None = None,
    ):
        # 1. Content Recommender
        self.content_rec = content_recommender or DenseContentRecommender()

        # 2. Collaborative Filtering Recommender
        self.cf_rec = cf_recommender or CollaborativeFilteringRecommender()

        # 3. User Profile Manager
        self.profile_manager = profile_manager or UserProfileManager()

        # 4. Optional lazy-loaded Semantic Searcher
        self._searcher = searcher

        # Precompute catalog metadata mappings for rapid indexing
        self.df = self.content_rec.df
        self.tmdb_to_idx = self.content_rec.tmdb_to_idx
        self.idx_to_tmdb = {idx: tmdb for tmdb, idx in self.tmdb_to_idx.items()}

        # Bayesian rating priors across catalog
        vote_counts = self.df["vote_count"].fillna(0).values.astype(float)
        vote_averages = self.df["vote_average"].fillna(self.cf_rec.global_mean).values.astype(float)
        C = float(np.median(vote_counts[vote_counts > 0])) if np.any(vote_counts > 0) else 10.0
        m = float(np.mean(vote_averages))
        # Weighted rating = (v / (v + m)) * R + (m / (v + m)) * C
        self.bayesian_ratings = (vote_counts / (vote_counts + C)) * vote_averages + (C / (vote_counts + C)) * m
        # Normalize to [0, 1]
        self.normalized_priors = np.clip((self.bayesian_ratings - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)

    @property
    def searcher(self) -> Any:
        """Lazy load MovieSearcher on demand."""
        if self._searcher is None:
            from ml.vectorstore.search import MovieSearcher
            self._searcher = MovieSearcher()
        return self._searcher

    def get_or_create_profile(self, user_id: str | int) -> UserProfile:
        """Access or instantiate a user profile."""
        return self.profile_manager.get_or_create(user_id)

    def record_interaction(
        self,
        user_id: str | int,
        interaction_type: str,
        tmdb_id: int | None = None,
        rating: float | None = None,
        query: str | None = None,
    ) -> UserProfile:
        """
        Record a user event ('rating', 'watch', or 'search') and dynamically
        update the user's taste embedding and genre preferences.
        """
        profile = self.get_or_create_profile(user_id)

        genres = None
        language = None
        embedding = None

        if tmdb_id is not None:
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is not None:
                movie_row = self.content_rec.get_movie_by_idx(idx)
                genres = movie_row.get("genres", [])
                language = movie_row.get("language")
                embedding = self.content_rec.embeddings[idx]

        if interaction_type == "rating" and tmdb_id is not None and rating is not None:
            profile.update_rating(
                tmdb_id=tmdb_id,
                rating=rating,
                genres=genres,
                language=language,
                movie_embedding=embedding,
            )

        elif interaction_type == "watch" and tmdb_id is not None:
            profile.update_watch(
                tmdb_id=tmdb_id,
                genres=genres,
                language=language,
                movie_embedding=embedding,
            )

        elif interaction_type == "search" and query:
            q_emb = None
            if self._searcher is not None:
                try:
                    q_emb = self._searcher.model.encode(f"query: {query}", normalize_embeddings=True)
                except Exception:
                    pass
            profile.update_search(query=query, query_embedding=q_emb)

        return profile

    def _get_adaptive_weights(self, maturity: str, has_query: bool = False) -> tuple[float, float, float]:
        """
        Return adaptive weights (w_cf, w_content, w_semantic) based on maturity and query presence.
        """
        if has_query:
            # Query is focal; personalization refines and re-ranks
            if maturity == "warm":
                return 0.20, 0.20, 0.60
            elif maturity == "nascent":
                return 0.15, 0.15, 0.70
            else:  # cold
                return 0.05, 0.15, 0.80
        else:
            # No query: pure personalized recommendation
            if maturity == "warm":
                return 0.65, 0.35, 0.0
            elif maturity == "nascent":
                return 0.45, 0.55, 0.0
            else:  # cold
                return 0.15, 0.85, 0.0

    def recommend(
        self,
        user_id: str | int,
        k: int = 10,
        language: str | None = None,
        min_rating: float | None = None,
        min_votes: int = 5,
    ) -> pd.DataFrame:
        """
        Generate personalized recommendations for a user combining:
        - Collaborative Filtering (SVD predictions / fold-in)
        - Content-Based taste similarity (user embedding centroid)
        - Genre affinity boost
        - Quality / Bayesian priors
        """
        profile = self.get_or_create_profile(user_id)
        u_int_id = int(user_id) if str(user_id).isdigit() else None
        is_known_cf_user = u_int_id is not None and u_int_id in self.cf_rec.u_map
        maturity = "warm" if is_known_cf_user else profile.maturity
        w_cf, w_content, _ = self._get_adaptive_weights(maturity, has_query=False)

        n_movies = len(self.df)

        # -------------------------------------------------------------
        # 1. Collaborative Filtering Signal (S_cf in [0, 1])
        # -------------------------------------------------------------
        cf_scores = np.zeros(n_movies, dtype=np.float32)

        # Check if user is known in MovieLens or has session ratings
        u_int_id = int(user_id) if str(user_id).isdigit() else None
        is_known_cf_user = u_int_id is not None and u_int_id in self.cf_rec.u_map

        if is_known_cf_user and self.cf_rec.P is not None and self.cf_rec.Q is not None:
            u_idx = self.cf_rec.u_map[u_int_id]
            u_b = self.cf_rec.user_bias.get(u_int_id, 0.0)
            raw_cf = self.cf_rec.global_mean + u_b + self.cf_rec.all_item_biases + (self.cf_rec.P[u_idx] @ self.cf_rec.Q.T)
            # Map item indices back to catalog rows
            for j, tmdb in self.cf_rec.inv_i_map.items():
                cat_idx = self.tmdb_to_idx.get(tmdb)
                if cat_idx is not None:
                    cf_scores[cat_idx] = float(np.clip((raw_cf[j] - 0.5) / 4.5, 0.0, 1.0))

        elif len(profile.ratings) > 0 and self.cf_rec.Q is not None:
            # New user session fold-in via ridge regression on item factors
            valid_items = [tmdb for tmdb in profile.ratings if tmdb in self.cf_rec.i_map]
            if valid_items:
                user_vals = [profile.ratings[tmdb] for tmdb in valid_items]
                u_mean = float(np.mean(user_vals))
                u_b = (u_mean - self.cf_rec.global_mean) * len(valid_items) / (len(valid_items) + self.cf_rec.reg_c)

                residuals = []
                q_rows = []
                for tmdb in valid_items:
                    i_idx = self.cf_rec.i_map[tmdb]
                    i_b = self.cf_rec.item_bias.get(tmdb, 0.0)
                    residuals.append(profile.ratings[tmdb] - (self.cf_rec.global_mean + u_b + i_b))
                    q_rows.append(self.cf_rec.Q[i_idx])

                Q_sub = np.array(q_rows, dtype=np.float32)
                r_sub = np.array(residuals, dtype=np.float32)
                reg_eye = 0.05 * np.eye(self.cf_rec.Q.shape[1], dtype=np.float32)
                p_u = np.linalg.solve(Q_sub.T @ Q_sub + reg_eye, Q_sub.T @ r_sub)

                raw_cf = self.cf_rec.global_mean + u_b + self.cf_rec.all_item_biases + (p_u @ self.cf_rec.Q.T)
                for j, tmdb in self.cf_rec.inv_i_map.items():
                    cat_idx = self.tmdb_to_idx.get(tmdb)
                    if cat_idx is not None:
                        cf_scores[cat_idx] = float(np.clip((raw_cf[j] - 0.5) / 4.5, 0.0, 1.0))
            else:
                cf_scores = self.normalized_priors.copy()
        else:
            # Cold-start fallback: use Bayesian normalized priors
            cf_scores = self.normalized_priors.copy()

        # -------------------------------------------------------------
        # 2. Dense Content-Based Signal (S_content in [0, 1])
        # -------------------------------------------------------------
        content_scores = np.zeros(n_movies, dtype=np.float32)

        if profile.taste_embedding is not None:
            cos_sims = self.content_rec.embeddings @ profile.taste_embedding
            content_scores = np.clip((cos_sims + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        elif len(profile.watched) > 0:
            # Centroid of watched items
            watched_indices = [self.tmdb_to_idx[t] for t in profile.watched if t in self.tmdb_to_idx]
            if watched_indices:
                w_vec = np.mean(self.content_rec.embeddings[watched_indices], axis=0)
                w_norm = np.linalg.norm(w_vec)
                if w_norm > 0:
                    w_vec /= w_norm
                    cos_sims = self.content_rec.embeddings @ w_vec
                    content_scores = np.clip((cos_sims + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        else:
            # Completely cold user: content score defaults to Bayesian quality prior
            content_scores = self.normalized_priors.copy()

        # -------------------------------------------------------------
        # 3. Genre Affinity Boost
        # -------------------------------------------------------------
        genre_boost = np.zeros(n_movies, dtype=np.float32)
        top_genres = profile.get_top_genres(n=3)
        if top_genres:
            top_genre_set = set(top_genres)
            for idx, g_set in enumerate(self.content_rec._genre_sets):
                if g_set and (top_genre_set & g_set):
                    genre_boost[idx] = 0.08  # up to 8% boost for preferred genres

        # -------------------------------------------------------------
        # 4. Hybrid Score Fusion
        # -------------------------------------------------------------
        hybrid_scores = (w_cf * cf_scores) + (w_content * content_scores) + genre_boost

        # -------------------------------------------------------------
        # 5. Filtering: exclude watched/rated + apply criteria
        # -------------------------------------------------------------
        mask = np.ones(n_movies, dtype=bool)

        # Exclude seen movies
        for seen_tmdb in profile.watched:
            cat_idx = self.tmdb_to_idx.get(seen_tmdb)
            if cat_idx is not None:
                mask[cat_idx] = False

        if is_known_cf_user:
            seen_cf = self.cf_rec.user_seen_items.get(u_int_id, set())
            for seen_tmdb in seen_cf:
                cat_idx = self.tmdb_to_idx.get(seen_tmdb)
                if cat_idx is not None:
                    mask[cat_idx] = False

        if language:
            mask &= (self.df["language"].values == language)
        if min_rating is not None:
            mask &= (self.df["vote_average"].values >= min_rating)
        if min_votes:
            mask &= (self.df["vote_count"].values >= min_votes)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame()

        valid_scores = hybrid_scores[valid_indices]
        top_k_indices = valid_indices[valid_scores.argsort()[::-1][:k]]

        results = []
        for idx in top_k_indices:
            row = self.df.iloc[idx]
            results.append({
                "tmdb_id": int(row["tmdb_id"]),
                "title": str(row["title"]),
                "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
                "language": str(row["language"]),
                "genres": list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
                "vote_average": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
                "poster_path": str(row["poster_path"]) if pd.notna(row.get("poster_path")) else None,
                "backdrop_path": str(row["backdrop_path"]) if pd.notna(row.get("backdrop_path")) else None,
                "overview": str(row["overview"]) if pd.notna(row.get("overview")) else "",
                "runtime": int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
                "cast": list(row["cast"]) if isinstance(row.get("cast"), (list, np.ndarray)) else [],
                "director": str(row["director"]) if pd.notna(row.get("director")) else None,
                "hybrid_score": round(float(hybrid_scores[idx]), 4),
                "cf_score": round(float(cf_scores[idx]), 4),
                "content_score": round(float(content_scores[idx]), 4),
                "match_source": f"Hybrid ({maturity.capitalize()}: {int(w_cf*100)}% CF, {int(w_content*100)}% Content)",
            })

        return pd.DataFrame(results)

    def search(
        self,
        query: str,
        user_id: str | int | None = None,
        k: int = 10,
        language: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_rating: float | None = None,
        min_votes: int | None = None,
        genre: str | None = None,
    ) -> pd.DataFrame:
        """
        Personalized Hybrid Search:
        Retrieves candidates matching semantic intent via vector search,
        then re-ranks them using the user's collaborative and content taste profile.
        """
        # 1. Vector Search Retrieval (fetch 3x candidate pool)
        raw_results = self.searcher.search(
            query=query,
            top_k=max(k * 3, 25),
            language=language,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            min_votes=min_votes,
            genre=genre,
        )

        if not raw_results:
            return pd.DataFrame()

        # Update user profile with search query if user_id is provided
        profile = None
        if user_id is not None:
            profile = self.record_interaction(user_id=user_id, interaction_type="search", query=query)

        maturity = profile.maturity if profile else "cold"
        w_cf, w_content, w_semantic = self._get_adaptive_weights(maturity, has_query=True)

        scored_candidates = []
        for res in raw_results:
            payload = res.payload
            tmdb_id = int(payload.get("tmdb_id", 0))
            cat_idx = self.tmdb_to_idx.get(tmdb_id)

            # Semantic score from Qdrant vector search
            semantic_score = float(np.clip(res.score, 0.0, 1.0))

            # Collaborative filtering score
            if profile and len(profile.ratings) > 0:
                cf_pred = self.cf_rec.predict_rating(user_id=hash(profile.user_id) % 10000, tmdb_id=tmdb_id)
                cf_score = float(np.clip((cf_pred - 0.5) / 4.5, 0.0, 1.0))
            elif cat_idx is not None:
                cf_score = float(self.normalized_priors[cat_idx])
            else:
                cf_score = 0.5

            # Content taste score
            if profile and profile.taste_embedding is not None and cat_idx is not None:
                cos_sim = float(self.content_rec.embeddings[cat_idx] @ profile.taste_embedding)
                content_score = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
            else:
                content_score = semantic_score

            # Fused hybrid score
            hybrid_score = (
                (w_semantic * semantic_score)
                + (w_cf * cf_score)
                + (w_content * content_score)
            )

            cat_row = self.df.iloc[cat_idx] if cat_idx is not None else None
            poster_path = str(cat_row["poster_path"]) if cat_row is not None and pd.notna(cat_row.get("poster_path")) else payload.get("poster_path")
            backdrop_path = str(cat_row["backdrop_path"]) if cat_row is not None and pd.notna(cat_row.get("backdrop_path")) else payload.get("backdrop_path")
            overview = str(cat_row["overview"]) if cat_row is not None and pd.notna(cat_row.get("overview")) else str(payload.get("overview", ""))
            runtime = int(cat_row["runtime"]) if cat_row is not None and pd.notna(cat_row.get("runtime")) else payload.get("runtime")
            cast = list(cat_row["cast"]) if cat_row is not None and isinstance(cat_row.get("cast"), (list, np.ndarray)) else payload.get("cast", [])
            director = str(cat_row["director"]) if cat_row is not None and pd.notna(cat_row.get("director")) else payload.get("director")

            scored_candidates.append({
                "tmdb_id": tmdb_id,
                "title": payload.get("title"),
                "release_year": payload.get("release_year"),
                "language": payload.get("language"),
                "genres": payload.get("genres"),
                "vote_average": payload.get("vote_average"),
                "poster_path": poster_path,
                "backdrop_path": backdrop_path,
                "overview": overview,
                "runtime": runtime,
                "cast": cast,
                "director": director,
                "semantic_score": round(semantic_score, 4),
                "cf_score": round(cf_score, 4),
                "content_score": round(content_score, 4),
                "hybrid_score": round(hybrid_score, 4),
            })

        # Re-rank by hybrid score descending
        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return pd.DataFrame(scored_candidates[:k])

    def similar_movies_hybrid(
        self,
        identifier: str | int,
        k: int = 10,
        language: str | None = None,
        cf_weight: float = 0.4,
    ) -> pd.DataFrame:
        """
        Find similar movies combining Dense Content Similarity (E5 vectors + metadata)
        with Item-to-Item Collaborative Filtering (users who co-rated both movies).
        """
        target_idx = self.content_rec.find(identifier)
        if target_idx is None:
            return pd.DataFrame()

        target_tmdb = int(self.df.iloc[target_idx]["tmdb_id"])

        # 1. Content-based similarities
        content_recs = self.content_rec.similar(identifier=target_tmdb, k=k * 2, language=language)
        content_map = {
            int(row["tmdb_id"]): float(row["score"])
            for _, row in content_recs.iterrows()
        }

        # 2. Collaborative item-item similarities
        cf_recs = self.cf_rec.similar_items(tmdb_id=target_tmdb, k=k * 2)
        cf_map = {
            int(row["tmdb_id"]): float(row["cf_similarity"])
            for _, row in cf_recs.iterrows()
        } if not cf_recs.empty else {}

        # Merge candidate IDs
        candidate_ids = set(content_map.keys()) | set(cf_map.keys())
        results = []

        for cand_id in candidate_ids:
            cand_idx = self.tmdb_to_idx.get(cand_id)
            if cand_idx is None:
                continue

            movie_row = self.content_rec.get_movie_by_idx(cand_idx)

            # Language filter
            if language and movie_row.get("language") != language:
                continue

            s_content = content_map.get(cand_id, 0.0)
            s_cf = cf_map.get(cand_id, 0.0)

            # If movie has no CF ratings, rely 100% on content
            if cand_id in cf_map and target_tmdb in self.cf_rec.i_map:
                fused_score = ((1.0 - cf_weight) * s_content) + (cf_weight * s_cf)
            else:
                fused_score = s_content

            cand_row = self.df.iloc[cand_idx]
            results.append({
                "tmdb_id": cand_id,
                "title": movie_row.get("title"),
                "release_year": movie_row.get("release_year"),
                "language": movie_row.get("language"),
                "genres": movie_row.get("genres"),
                "vote_average": movie_row.get("vote_average"),
                "poster_path": str(cand_row["poster_path"]) if pd.notna(cand_row.get("poster_path")) else None,
                "backdrop_path": str(cand_row["backdrop_path"]) if pd.notna(cand_row.get("backdrop_path")) else None,
                "overview": str(cand_row["overview"]) if pd.notna(cand_row.get("overview")) else "",
                "runtime": int(cand_row["runtime"]) if pd.notna(cand_row.get("runtime")) else None,
                "cast": list(cand_row["cast"]) if isinstance(cand_row.get("cast"), (list, np.ndarray)) else [],
                "director": str(cand_row["director"]) if pd.notna(cand_row.get("director")) else None,
                "hybrid_similarity": round(float(fused_score), 4),
                "content_similarity": round(float(s_content), 4),
                "cf_similarity": round(float(s_cf), 4),
            })

        results.sort(key=lambda x: x["hybrid_similarity"], reverse=True)
        return pd.DataFrame(results[:k])
