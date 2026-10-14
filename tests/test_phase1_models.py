"""
test_phase1_models.py
---------------------
Pytest unit and integration tests for Phase 1 ML recommendation models:
- DenseContentRecommender (Multilingual E5 embeddings + metadata)
- CollaborativeFilteringRecommender (SVD matrix factorization + item similarity)
"""

import numpy as np
import pandas as pd
import pytest

from ml.models.content_based import DenseContentRecommender
from ml.models.collaborative import CollaborativeFilteringRecommender


@pytest.fixture(scope="module")
def content_recommender():
    """Shared fixture for DenseContentRecommender."""
    return DenseContentRecommender()


@pytest.fixture(scope="module")
def cf_recommender():
    """Shared fixture for CollaborativeFilteringRecommender."""
    return CollaborativeFilteringRecommender(n_factors=20)


# =====================================================================
# DENSE CONTENT-BASED RECOMMENDER TESTS
# =====================================================================

def test_content_recommender_initialization(content_recommender):
    """Verify that DenseContentRecommender loads the catalog and embeddings correctly."""
    assert len(content_recommender.df) > 0
    assert content_recommender.embeddings.ndim == 2
    assert content_recommender.embeddings.shape[0] == len(content_recommender.df)
    assert content_recommender.embeddings.shape[1] == 768  # E5 base dimension


def test_content_recommender_find(content_recommender):
    """Verify title and TMDB ID lookups."""
    # Lookup by title
    idx = content_recommender.find("Interstellar")
    assert idx is not None
    assert content_recommender.df.iloc[idx]["title"].lower() == "interstellar"

    # Lookup by TMDB ID
    tmdb_id = int(content_recommender.df.iloc[idx]["tmdb_id"])
    assert content_recommender.find(tmdb_id) == idx


def test_content_recommender_similar(content_recommender):
    """Verify similar movies retrieval, schema, and exclusion of self."""
    recs = content_recommender.similar("Interstellar", k=5)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert "score" in recs.columns
    assert "tmdb_id" in recs.columns
    assert "title" in recs.columns

    # Verify query movie is excluded from recommendations
    assert "interstellar" not in recs["title"].str.lower().tolist()


def test_content_recommender_language_filter(content_recommender):
    """Verify language filtering returns only movies in requested language."""
    recs = content_recommender.similar("Interstellar", k=5, language="hi")
    assert len(recs) <= 5
    if len(recs) > 0:
        assert (recs["language"] == "hi").all()


def test_content_recommender_user_profile(content_recommender):
    """Verify taste centroid recommendation from liked movies list."""
    liked = ["Interstellar", "Inception"]
    recs = content_recommender.recommend_from_profile(liked, k=5)
    assert len(recs) == 5

    # Consumed movies should be filtered out
    titles_lower = recs["title"].str.lower().tolist()
    assert "interstellar" not in titles_lower
    assert "inception" not in titles_lower


# =====================================================================
# COLLABORATIVE FILTERING RECOMMENDER TESTS
# =====================================================================

def test_cf_recommender_initialization(cf_recommender):
    """Verify SVD matrices and item mappings are properly fitted."""
    assert cf_recommender.is_fitted
    assert cf_recommender.P is not None
    assert cf_recommender.Q is not None
    assert cf_recommender.P.shape[1] == 20
    assert cf_recommender.Q.shape[1] == 20
    assert 0.5 <= cf_recommender.global_mean <= 5.0


def test_cf_predict_rating(cf_recommender):
    """Verify rating prediction bounds and known item prediction."""
    # Sample user and movie from training ratings
    sample = cf_recommender.ratings_df.iloc[0]
    u, tmdb = int(sample["userId"]), int(sample["tmdb_id"])

    pred = cf_recommender.predict_rating(u, tmdb)
    assert 0.5 <= pred <= 5.0

    # Cold-start user or item should still return bounded prior
    cold_pred = cf_recommender.predict_rating(999999, 999999)
    assert 0.5 <= cold_pred <= 5.0


def test_cf_recommend_for_user(cf_recommender):
    """Verify personalized recommendation for known user filters seen movies."""
    user_id = 1
    recs = cf_recommender.recommend_for_user(user_id=user_id, k=5, filter_seen=True)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert "predicted_rating" in recs.columns

    # Verify none of the recommended movies were already seen by user
    seen = cf_recommender.user_seen_items.get(user_id, set())
    rec_tmdbs = set(recs["tmdb_id"].tolist())
    assert len(rec_tmdbs & seen) == 0


def test_cf_similar_items(cf_recommender):
    """Verify item-to-item collaborative filtering on a known blockbuster."""
    # Toy Story TMDB ID = 862
    sims = cf_recommender.similar_items(tmdb_id=862, k=5)
    assert isinstance(sims, pd.DataFrame)
    assert len(sims) == 5
    assert "cf_similarity" in sims.columns
    assert (sims["cf_similarity"] > 0).all()
    assert 862 not in sims["tmdb_id"].tolist()


def test_cf_new_user_fold_in(cf_recommender):
    """Verify folding in a new user with session ratings produces valid recommendations."""
    # User rates Toy Story (862) with 5.0 and Titanic (597) with 4.5
    user_ratings = {862: 5.0, 597: 4.5}
    recs = cf_recommender.recommend_for_new_user(user_ratings, k=5)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert "predicted_rating" in recs.columns
    # Rated items should not appear in recommendations
    assert 862 not in recs["tmdb_id"].tolist()
    assert 597 not in recs["tmdb_id"].tolist()


def test_cf_offline_evaluation(cf_recommender):
    """
    Verify that offline train/test evaluation executes properly and that
    the SVD collaborative model beats the naive global-mean baseline on unseen data.
    """
    metrics = cf_recommender.evaluate(test_size=0.2, random_state=42)

    # Required metric keys
    expected_keys = {
        "test_rmse",
        "baseline_rmse",
        "test_mae",
        "baseline_mae",
        "rmse_improvement_pct",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "evaluated_users",
        "test_samples",
    }
    assert expected_keys.issubset(metrics.keys())

    # Comparative evaluation: SVD model must strictly outperform the naive global-mean baseline
    assert metrics["test_rmse"] < metrics["baseline_rmse"]
    assert metrics["test_mae"] < metrics["baseline_mae"]
    assert metrics["rmse_improvement_pct"] > 0.0

    # Functional validity of ranking metrics: values must be valid bounded metrics in [0.0, 1.0]
    assert 0.0 <= metrics["precision_at_10"] <= 1.0
    assert 0.0 <= metrics["recall_at_10"] <= 1.0
    assert 0.0 <= metrics["ndcg_at_10"] <= 1.0
    assert metrics["evaluated_users"] > 0
    assert metrics["test_samples"] > 0
