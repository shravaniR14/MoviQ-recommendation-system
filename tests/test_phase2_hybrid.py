"""
test_phase2_hybrid.py
---------------------
Pytest unit and integration tests for Phase 2:
- Dynamic UserProfile tracking (ratings, watches, searches, genre/language affinities, taste vector)
- HybridRecommender (adaptive fusion of Collaborative Filtering, Content Embeddings, and User Profile)
"""

import numpy as np
import pandas as pd
import pytest

from ml.models.collaborative import CollaborativeFilteringRecommender
from ml.models.content_based import DenseContentRecommender
from ml.models.hybrid import HybridRecommender
from ml.personalization.user_profile import UserProfile, UserProfileManager


@pytest.fixture(scope="module")
def shared_content_rec():
    return DenseContentRecommender()


@pytest.fixture(scope="module")
def shared_cf_rec():
    return CollaborativeFilteringRecommender(n_factors=15)


@pytest.fixture(scope="module")
def shared_hybrid(shared_content_rec, shared_cf_rec):
    return HybridRecommender(
        content_recommender=shared_content_rec,
        cf_recommender=shared_cf_rec,
    )


# =====================================================================
# USER PROFILE TESTS
# =====================================================================

def test_user_profile_initial_cold_start():
    """Verify that a newly instantiated profile is cold-start."""
    profile = UserProfile(user_id="test_user_cold")
    assert profile.is_cold_start is True
    assert profile.maturity == "cold"
    assert len(profile.ratings) == 0
    assert len(profile.watched) == 0
    assert profile.taste_embedding is None


def test_user_profile_rating_and_taste_vector():
    """Verify rating updates, genre affinity, and embedding blending."""
    profile = UserProfile(user_id="test_user_ratings")
    fake_embedding = np.ones(768, dtype=np.float32) / np.sqrt(768)

    profile.update_rating(
        tmdb_id=157336,  # Interstellar
        rating=5.0,
        genres=["Science Fiction", "Drama"],
        language="en",
        movie_embedding=fake_embedding,
    )

    assert profile.is_cold_start is False
    assert profile.maturity == "nascent"
    assert profile.ratings[157336] == 5.0
    assert 157336 in profile.watched
    assert profile.genre_scores.get("Science Fiction", 0) > 0
    assert profile.taste_embedding is not None
    assert profile.taste_embedding.shape == (768,)
    # Norm should be preserved close to 1.0
    assert np.isclose(np.linalg.norm(profile.taste_embedding), 1.0, atol=1e-3)


def test_user_profile_maturity_transition():
    """Verify maturity transitions from cold -> nascent -> warm as interactions accrue."""
    profile = UserProfile(user_id="test_progression")
    assert profile.maturity == "cold"

    # Add 2 ratings -> nascent
    for i in range(1, 3):
        profile.update_rating(tmdb_id=100 + i, rating=4.0)
    assert profile.maturity == "nascent"

    # Add 3 more ratings -> warm (5 total)
    for i in range(3, 6):
        profile.update_rating(tmdb_id=100 + i, rating=4.0)
    assert profile.maturity == "warm"


def test_user_profile_serialization():
    """Verify profile serialization and deserialization via dictionary."""
    profile = UserProfile(user_id="serial_user")
    profile.update_rating(tmdb_id=862, rating=4.5, genres=["Animation", "Comedy"])
    profile.update_search("toy story cartoon")

    data = profile.to_dict()
    assert data["user_id"] == "serial_user"
    assert "862" in data["ratings"]
    assert "toy story cartoon" in data["searches"]

    restored = UserProfile.from_dict(data)
    assert restored.user_id == "serial_user"
    assert restored.ratings[862] == 4.5
    assert 862 in restored.watched
    assert "toy story cartoon" in restored.searches


def test_user_profile_manager_lifecycle(tmp_path):
    """Verify profile manager retrieval and persistence to file."""
    store_file = tmp_path / "profiles.json"
    manager = UserProfileManager(storage_path=store_file)

    p1 = manager.get_or_create("user_alpha")
    p1.update_rating(tmdb_id=27205, rating=5.0)  # Inception
    manager.save()

    assert store_file.exists()

    # Load in a fresh manager instance
    fresh_manager = UserProfileManager(storage_path=store_file)
    p1_loaded = fresh_manager.get_or_create("user_alpha")
    assert p1_loaded.ratings[27205] == 5.0


# =====================================================================
# HYBRID RECOMMENDER TESTS
# =====================================================================

def test_hybrid_recommend_cold_start(shared_hybrid):
    """Verify recommendations for cold-start user with no history."""
    recs = shared_hybrid.recommend(user_id="cold_user_999", k=5)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert "hybrid_score" in recs.columns
    assert "cf_score" in recs.columns
    assert "content_score" in recs.columns
    assert "match_source" in recs.columns
    assert "Cold" in recs.iloc[0]["match_source"]
    assert (recs["hybrid_score"] >= 0.0).all()


def test_hybrid_recommend_filters_watched(shared_hybrid):
    """Verify that consumed/rated movies are excluded from recommendations."""
    uid = "filter_user_1"
    # User rates Inception (27205) and Interstellar (157336)
    shared_hybrid.record_interaction(uid, "rating", tmdb_id=27205, rating=5.0)
    shared_hybrid.record_interaction(uid, "rating", tmdb_id=157336, rating=5.0)

    recs = shared_hybrid.recommend(user_id=uid, k=10)
    rec_ids = recs["tmdb_id"].tolist()

    assert 27205 not in rec_ids
    assert 157336 not in rec_ids


def test_hybrid_recommend_personalization_evolution(shared_hybrid):
    """Verify that preferences shift when new ratings are logged."""
    uid = "evolving_user"
    initial_recs = shared_hybrid.recommend(uid, k=5)

    # User rates 3 Hindi movies highly
    shared_hybrid.record_interaction(uid, "rating", tmdb_id=20453, rating=5.0)  # 3 Idiots
    shared_hybrid.record_interaction(uid, "rating", tmdb_id=19404, rating=5.0)  # Dilwale Dulhania Le Jayenge

    updated_recs = shared_hybrid.recommend(uid, k=5)
    assert len(updated_recs) == 5

    # Match source should reflect Nascent state
    assert "Nascent" in updated_recs.iloc[0]["match_source"]

    # Top recommendations should change to reflect user's newly logged tastes
    initial_titles = set(initial_recs["title"].tolist())
    updated_titles = set(updated_recs["title"].tolist())
    assert initial_titles != updated_titles


def test_hybrid_recommend_warm_user(shared_hybrid):
    """Verify warm user recommendation leverages MovieLens collaborative factors."""
    # User ID 1 is a known MovieLens user
    recs = shared_hybrid.recommend(user_id="1", k=5)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert "Warm" in recs.iloc[0]["match_source"]


def test_hybrid_similar_movies(shared_hybrid):
    """Verify item-to-item hybrid similarity combines content and collaborative signals."""
    # Toy Story (862)
    sims = shared_hybrid.similar_movies_hybrid(identifier=862, k=5)
    assert isinstance(sims, pd.DataFrame)
    assert len(sims) == 5
    assert "hybrid_similarity" in sims.columns
    assert "content_similarity" in sims.columns
    assert "cf_similarity" in sims.columns
    # Query movie must be excluded
    assert 862 not in sims["tmdb_id"].tolist()


def test_hybrid_adaptive_weights_logic(shared_hybrid):
    """Verify adaptive weights change according to user maturity."""
    w_cf_cold, w_cont_cold, _ = shared_hybrid._get_adaptive_weights("cold", has_query=False)
    w_cf_nasc, w_cont_nasc, _ = shared_hybrid._get_adaptive_weights("nascent", has_query=False)
    w_cf_warm, w_cont_warm, _ = shared_hybrid._get_adaptive_weights("warm", has_query=False)

    # As user matures, collaborative filtering weight increases
    assert w_cf_cold < w_cf_nasc < w_cf_warm
    # And content weight decreases proportionally
    assert w_cont_cold > w_cont_nasc > w_cont_warm
