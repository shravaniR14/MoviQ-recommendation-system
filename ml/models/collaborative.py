"""Collaborative Filtering Recommender using Regularized SVD & Item-Item Similarity."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_RATINGS_PATH = DATA_DIR / "raw" / "ml-latest-small" / "ratings.csv"
ML_LINKS_PATH = DATA_DIR / "processed" / "ml_links.parquet"
MOVIES_PATH = DATA_DIR / "processed" / "movies_enriched.parquet"
if not MOVIES_PATH.exists():
    MOVIES_PATH = DATA_DIR / "processed" / "movies.parquet"


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering engine combining:
    1. Regularized SVD (Matrix Factorization) for user rating prediction & ranking.
    2. Item-to-Item Collaborative Cosine Similarity on sparse interaction vectors.
    3. Fold-in capability for new users with arbitrary session ratings.
    """

    def __init__(
        self,
        n_factors: int = 25,
        reg_c: float = 10.0,
        ratings_df: pd.DataFrame | None = None,
        movies_df: pd.DataFrame | None = None,
    ):
        self.n_factors = n_factors
        self.reg_c = reg_c

        # Load movies metadata
        if movies_df is None:
            self.movies_df = pd.read_parquet(MOVIES_PATH).reset_index(drop=True)
        else:
            self.movies_df = movies_df.reset_index(drop=True)

        # Build movies metadata lookup
        self.tmdb_to_row = {
            int(row["tmdb_id"]): row for _, row in self.movies_df.iterrows()
        }

        # Load ratings and merge with TMDB links
        if ratings_df is None:
            raw_ratings = pd.read_csv(RAW_RATINGS_PATH)
            links = pd.read_parquet(ML_LINKS_PATH)
            merged = raw_ratings.merge(links[["movieId", "tmdb_id"]], on="movieId")
            self.ratings_df = merged[["userId", "tmdb_id", "rating"]].dropna().copy()
            self.ratings_df["userId"] = self.ratings_df["userId"].astype(int)
            self.ratings_df["tmdb_id"] = self.ratings_df["tmdb_id"].astype(int)
        else:
            self.ratings_df = ratings_df.copy()

        # Placeholders for fitted model state
        self.is_fitted = False
        self.global_mean: float = 3.5
        self.user_bias: dict[int, float] = {}
        self.item_bias: dict[int, float] = {}
        self.u_map: dict[int, int] = {}
        self.i_map: dict[int, int] = {}
        self.inv_u_map: dict[int, int] = {}
        self.inv_i_map: dict[int, int] = {}
        self.P: np.ndarray | None = None  # (n_users, n_factors)
        self.Q: np.ndarray | None = None  # (n_items, n_factors)
        self.all_item_biases: np.ndarray | None = None
        self.item_user_sparse: csr_matrix | None = None  # (n_items, n_users)
        self.user_seen_items: dict[int, set[int]] = {}

        # Fit automatically on initialization
        self.fit()

    def fit(self, train_df: pd.DataFrame | None = None) -> CollaborativeFilteringRecommender:
        """Fit the Regularized SVD and Item-Item Sparse Matrices on the ratings dataset."""
        df = train_df if train_df is not None else self.ratings_df

        self.global_mean = float(df["rating"].mean())

        # Damped regularized user and item biases
        user_stats = df.groupby("userId")["rating"].agg(["count", "mean"])
        self.user_bias = (
            (user_stats["mean"] - self.global_mean)
            * user_stats["count"]
            / (user_stats["count"] + self.reg_c)
        ).to_dict()

        item_stats = df.groupby("tmdb_id")["rating"].agg(["count", "mean"])
        self.item_bias = (
            (item_stats["mean"] - self.global_mean)
            * item_stats["count"]
            / (item_stats["count"] + self.reg_c)
        ).to_dict()

        # Build continuous index mappings
        unique_users = np.unique(df["userId"])
        unique_items = np.unique(df["tmdb_id"])

        self.u_map = {u: idx for idx, u in enumerate(unique_users)}
        self.i_map = {tmdb: idx for idx, tmdb in enumerate(unique_items)}
        self.inv_u_map = {idx: u for u, idx in self.u_map.items()}
        self.inv_i_map = {idx: tmdb for tmdb, idx in self.i_map.items()}

        # User seen items mapping
        self.user_seen_items = df.groupby("userId")["tmdb_id"].apply(set).to_dict()

        # Residual Matrix: R - (mu + b_u + b_i)
        u_b = df["userId"].map(self.user_bias).fillna(0.0).values
        i_b = df["tmdb_id"].map(self.item_bias).fillna(0.0).values
        residuals = df["rating"].values - (self.global_mean + u_b + i_b)

        rows = [self.u_map[u] for u in df["userId"]]
        cols = [self.i_map[tmdb] for tmdb in df["tmdb_id"]]
        n_u, n_i = len(unique_users), len(unique_items)

        R_res = csr_matrix((residuals, (rows, cols)), shape=(n_u, n_i))

        # Perform SVD on residuals
        k = min(self.n_factors, n_u - 1, n_i - 1)
        U, s, Vt = svds(R_res, k=k)

        # Scale factors by square root of singular values
        s_sqrt = np.sqrt(s)
        self.P = U * s_sqrt
        self.Q = (Vt.T) * s_sqrt

        # Precompute item biases array for fast vector dot-products
        self.all_item_biases = np.array(
            [self.item_bias.get(self.inv_i_map[j], 0.0) for j in range(n_i)],
            dtype=np.float32,
        )

        # Item-user sparse matrix for item-to-item collaborative filtering
        self.item_user_sparse = csr_matrix(
            (df["rating"].values, (cols, rows)),
            shape=(n_i, n_u),
        )

        self.is_fitted = True
        return self

    def predict_rating(self, user_id: int, tmdb_id: int) -> float:
        """
        Predict rating r_hat in [0.5, 5.0] for a given user and TMDB item.
        Handles cold-start items and users smoothly.
        """
        u_b = self.user_bias.get(user_id, 0.0)
        i_b = self.item_bias.get(tmdb_id, 0.0)
        baseline = self.global_mean + u_b + i_b

        if (
            self.P is not None
            and self.Q is not None
            and user_id in self.u_map
            and tmdb_id in self.i_map
        ):
            u_idx = self.u_map[user_id]
            i_idx = self.i_map[tmdb_id]
            interaction = float(np.dot(self.P[u_idx], self.Q[i_idx]))
            pred = baseline + interaction
        else:
            pred = baseline

        return float(np.clip(pred, 0.5, 5.0))

    def recommend_for_user(
        self,
        user_id: int,
        k: int = 10,
        filter_seen: bool = True,
        min_votes: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-K recommendations for an existing user based on SVD predicted ratings.
        """
        if not self.is_fitted or self.P is None or self.Q is None:
            raise RuntimeError("Model is not fitted.")

        if user_id not in self.u_map:
            # Cold-start fallback: return top items by regularized item bias
            return self._cold_start_recommendations(k=k, min_votes=min_votes)

        u_idx = self.u_map[user_id]
        u_b = self.user_bias.get(user_id, 0.0)

        # Vectorized scoring over all known collaborative items
        scores = self.global_mean + u_b + self.all_item_biases + (self.P[u_idx] @ self.Q.T)

        # Mask out seen items
        if filter_seen:
            seen_set = self.user_seen_items.get(user_id, set())
            for tmdb in seen_set:
                if tmdb in self.i_map:
                    scores[self.i_map[tmdb]] = -1e9

        top_indices = scores.argsort()[::-1][: k * 2]  # get candidate pool
        results = []

        for idx in top_indices:
            tmdb = self.inv_i_map[idx]
            movie = self.tmdb_to_row.get(tmdb)
            if movie is None:
                continue
            if min_votes and movie.get("vote_count", 0) < min_votes:
                continue

            results.append({
                "tmdb_id": tmdb,
                "title": movie.get("title"),
                "release_year": movie.get("release_year"),
                "language": movie.get("language"),
                "genres": movie.get("genres"),
                "predicted_rating": round(float(np.clip(scores[idx], 0.5, 5.0)), 3),
            })
            if len(results) >= k:
                break

        return pd.DataFrame(results)

    def recommend_for_new_user(
        self,
        user_ratings: dict[int, float],
        k: int = 10,
        min_votes: int = 10,
    ) -> pd.DataFrame:
        """
        Fold-in recommendation for an active session/new user given a dictionary of {tmdb_id: rating}.
        Project user ratings into the latent factor space using pseudo-inverse:
        p_u = (Q^T Q + reg * I)^(-1) Q^T r_u
        """
        if not self.is_fitted or self.Q is None:
            raise RuntimeError("Model is not fitted.")

        valid_items = [tmdb for tmdb in user_ratings if tmdb in self.i_map]
        if not valid_items:
            return self._cold_start_recommendations(k=k, min_votes=min_votes)

        # Compute user mean & bias
        user_vals = [user_ratings[tmdb] for tmdb in valid_items]
        u_mean = float(np.mean(user_vals))
        u_b = (u_mean - self.global_mean) * len(valid_items) / (len(valid_items) + self.reg_c)

        # User rating residuals
        residuals = []
        q_rows = []
        for tmdb in valid_items:
            i_idx = self.i_map[tmdb]
            i_b = self.item_bias.get(tmdb, 0.0)
            res = user_ratings[tmdb] - (self.global_mean + u_b + i_b)
            residuals.append(res)
            q_rows.append(self.Q[i_idx])

        Q_sub = np.array(q_rows, dtype=np.float32)  # (M, k)
        r_sub = np.array(residuals, dtype=np.float32)  # (M,)

        # Ridge regression solve: p_u = (Q^T Q + lambda I)^(-1) Q^T r
        k_dim = self.Q.shape[1]
        reg_eye = 0.05 * np.eye(k_dim, dtype=np.float32)
        p_u = np.linalg.solve(Q_sub.T @ Q_sub + reg_eye, Q_sub.T @ r_sub)

        # Score all catalog items
        scores = self.global_mean + u_b + self.all_item_biases + (p_u @ self.Q.T)

        # Mask already rated items
        for tmdb in valid_items:
            scores[self.i_map[tmdb]] = -1e9

        top_indices = scores.argsort()[::-1][: k * 2]
        results = []

        for idx in top_indices:
            tmdb = self.inv_i_map[idx]
            movie = self.tmdb_to_row.get(tmdb)
            if movie is None:
                continue
            if min_votes and movie.get("vote_count", 0) < min_votes:
                continue

            results.append({
                "tmdb_id": tmdb,
                "title": movie.get("title"),
                "release_year": movie.get("release_year"),
                "language": movie.get("language"),
                "genres": movie.get("genres"),
                "predicted_rating": round(float(np.clip(scores[idx], 0.5, 5.0)), 3),
            })
            if len(results) >= k:
                break

        return pd.DataFrame(results)

    def similar_items(
        self,
        tmdb_id: int,
        k: int = 10,
        min_common_users: int = 5,
    ) -> pd.DataFrame:
        """
        Item-to-Item Collaborative Filtering: find movies that users co-rated similarly.
        Uses cosine similarity on the sparse item-user rating matrix.
        """
        if not self.is_fitted or self.item_user_sparse is None:
            raise RuntimeError("Model is not fitted.")

        if tmdb_id not in self.i_map:
            return pd.DataFrame()

        target_idx = self.i_map[tmdb_id]
        target_vec = self.item_user_sparse[target_idx]

        # Cosine similarity against all items
        sims = cosine_similarity(target_vec, self.item_user_sparse).ravel()
        sims[target_idx] = -1.0  # exclude itself

        top_indices = sims.argsort()[::-1][: k * 3]
        results = []

        for idx in top_indices:
            cand_tmdb = self.inv_i_map[idx]
            movie = self.tmdb_to_row.get(cand_tmdb)
            if movie is None or sims[idx] <= 0:
                continue

            results.append({
                "tmdb_id": cand_tmdb,
                "title": movie.get("title"),
                "release_year": movie.get("release_year"),
                "language": movie.get("language"),
                "genres": movie.get("genres"),
                "cf_similarity": round(float(sims[idx]), 4),
            })
            if len(results) >= k:
                break

        return pd.DataFrame(results)

    def _cold_start_recommendations(self, k: int = 10, min_votes: int = 10) -> pd.DataFrame:
        """Return top items by regularized item bias for cold-start cases."""
        sorted_items = sorted(self.item_bias.items(), key=lambda x: x[1], reverse=True)
        results = []
        for tmdb, bias in sorted_items:
            movie = self.tmdb_to_row.get(tmdb)
            if movie is None:
                continue
            if min_votes and movie.get("vote_count", 0) < min_votes:
                continue
            results.append({
                "tmdb_id": tmdb,
                "title": movie.get("title"),
                "release_year": movie.get("release_year"),
                "language": movie.get("language"),
                "genres": movie.get("genres"),
                "predicted_rating": round(float(np.clip(self.global_mean + bias, 0.5, 5.0)), 3),
            })
            if len(results) >= k:
                break
        return pd.DataFrame(results)

    def evaluate(self, test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
        """
        Strict offline evaluation on a train/test split following ML best practices.
        Reports:
        - RMSE & MAE for rating prediction.
        - Precision@10, Recall@10, NDCG@10 for top-10 ranking of relevant items (rating >= 4.0).
        """
        train, test = train_test_split(self.ratings_df, test_size=test_size, random_state=random_state)

        # Temporary evaluator model fit strictly on train
        eval_model = CollaborativeFilteringRecommender(
            n_factors=self.n_factors,
            reg_c=self.reg_c,
            ratings_df=train,
            movies_df=self.movies_df,
        )

        # Vectorized test rating predictions
        u_b = test["userId"].map(eval_model.user_bias).fillna(0.0).values
        i_b = test["tmdb_id"].map(eval_model.item_bias).fillna(0.0).values
        base_preds = eval_model.global_mean + u_b + i_b

        u_idx_series = test["userId"].map(eval_model.u_map)
        i_idx_series = test["tmdb_id"].map(eval_model.i_map)
        valid_mask = u_idx_series.notna() & i_idx_series.notna()

        dot_preds = np.zeros(len(test), dtype=np.float32)
        if valid_mask.any() and eval_model.P is not None and eval_model.Q is not None:
            valid_u = u_idx_series[valid_mask].astype(int).values
            valid_i = i_idx_series[valid_mask].astype(int).values
            dot_preds[valid_mask] = np.sum(eval_model.P[valid_u] * eval_model.Q[valid_i], axis=1)

        y_pred = np.clip(base_preds + dot_preds, 0.5, 5.0)
        y_true = test["rating"].values

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # Baseline: Global Mean Prediction on test split
        baseline_preds = np.full_like(y_true, eval_model.global_mean)
        baseline_rmse = float(np.sqrt(np.mean((y_true - baseline_preds) ** 2)))
        baseline_mae = float(np.mean(np.abs(y_true - baseline_preds)))

        # Ranking metrics: top-10 recommendations for users with >= 3 positive ratings (>= 4.0) in test
        K = 10
        test_positives = test[test["rating"] >= 4.0].groupby("userId")["tmdb_id"].apply(set).to_dict()

        precisions, recalls, ndcgs = [], [], []

        for u, rel_set in test_positives.items():
            if len(rel_set) < 3 or u not in eval_model.u_map:
                continue

            rec_df = eval_model.recommend_for_user(user_id=u, k=K, filter_seen=True, min_votes=0)
            rec_tmdbs = rec_df["tmdb_id"].tolist()

            hits = len(set(rec_tmdbs) & rel_set)
            precisions.append(hits / K)
            recalls.append(hits / len(rel_set))

            dcg = sum((1.0 / np.log2(rank + 2)) for rank, tmdb in enumerate(rec_tmdbs) if tmdb in rel_set)
            idcg = sum((1.0 / np.log2(rank + 2)) for rank in range(min(K, len(rel_set))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        return {
            "test_rmse": round(rmse, 4),
            "baseline_rmse": round(baseline_rmse, 4),
            "test_mae": round(mae, 4),
            "baseline_mae": round(baseline_mae, 4),
            "rmse_improvement_pct": round((baseline_rmse - rmse) / baseline_rmse * 100, 2),
            "precision_at_10": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall_at_10": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
            "evaluated_users": len(precisions),
            "test_samples": len(test),
        }
