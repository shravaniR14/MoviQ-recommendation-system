"""
test_phase4_api.py
------------------
FastAPI integration and endpoint tests for Phase 4:
- GET  / and GET /health
- POST /recommend/personalized
- GET  /recommend/similar/{tmdb_id}
- POST /recommend/hybrid-search
- POST /user/interaction
- POST /ask
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_hybrid_recommender


@pytest.fixture(scope="module")
def client():
    """Shared TestClient instance with pre-warmed recommender."""
    # Ensure recommender is initialized
    get_hybrid_recommender()
    with TestClient(app) as test_client:
        yield test_client


# =====================================================================
# SYSTEM & HEALTH TESTS
# =====================================================================

def test_api_root(client):
    """Verify root / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "Moviq" in data["message"]


def test_api_health(client):
    """Verify /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["recommender_ready"] is True


# =====================================================================
# RECOMMENDATION ENDPOINT TESTS
# =====================================================================

def test_recommend_personalized_cold_user(client):
    """Verify /recommend/personalized for a brand new user with no history."""
    payload = {
        "user_id": "test_api_cold_user",
        "k": 5,
        "language": "en",
    }
    response = client.post("/recommend/personalized", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == "test_api_cold_user"
    assert data["maturity"] == "cold"
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5

    first = data["recommendations"][0]
    expected_fields = {"tmdb_id", "title", "hybrid_score", "match_source", "genres"}
    assert expected_fields.issubset(first.keys())
    assert first["language"] == "en"


def test_user_interaction_and_personalization_flow(client):
    """
    Verify complete flow:
    1. Cold user
    2. Record positive ratings
    3. Verify profile updates
    4. Verify personalized recommendations adapt and filter rated movies
    """
    uid = "test_flow_user_123"

    # Step 1: Log rating for Interstellar (tmdb_id 157336) and Inception (27205)
    rating_payload = {
        "user_id": uid,
        "interaction_type": "rating",
        "tmdb_id": 157336,
        "rating": 5.0,
    }
    r1 = client.post("/user/interaction", json=rating_payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "success"
    assert d1["maturity"] == "nascent"
    assert d1["total_ratings"] == 1

    # Log second rating
    r2 = client.post("/user/interaction", json={
        "user_id": uid,
        "interaction_type": "rating",
        "tmdb_id": 27205,
        "rating": 5.0,
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["total_ratings"] == 2

    # Log search interaction
    r_search = client.post("/user/interaction", json={
        "user_id": uid,
        "interaction_type": "search",
        "query": "space science fiction",
    })
    assert r_search.status_code == 200

    # Step 2: Request personalized recommendations
    rec_response = client.post("/recommend/personalized", json={
        "user_id": uid,
        "k": 6,
    })
    assert rec_response.status_code == 200
    rec_data = rec_response.json()

    assert rec_data["maturity"] == "nascent"
    rec_ids = [m["tmdb_id"] for m in rec_data["recommendations"]]

    # Rated movies must be filtered out
    assert 157336 not in rec_ids
    assert 27205 not in rec_ids


def test_user_interaction_validation_errors(client):
    """Verify input validation catches missing or malformed interaction fields."""
    # Rating without tmdb_id
    r1 = client.post("/user/interaction", json={
        "user_id": "u_bad",
        "interaction_type": "rating",
        "rating": 4.0,
    })
    assert r1.status_code == 422

    # Rating out of bounds (> 5.0)
    r2 = client.post("/user/interaction", json={
        "user_id": "u_bad",
        "interaction_type": "rating",
        "tmdb_id": 862,
        "rating": 6.5,
    })
    assert r2.status_code == 422

    # Search without query
    r3 = client.post("/user/interaction", json={
        "user_id": "u_bad",
        "interaction_type": "search",
    })
    assert r3.status_code == 422


def test_recommend_similar_movies(client):
    """Verify /recommend/similar/{tmdb_id} for a valid movie."""
    # Toy Story (862)
    response = client.get("/recommend/similar/862?k=5")
    assert response.status_code == 200
    data = response.json()

    assert data["tmdb_id"] == 862
    assert "Toy Story" in data["target_title"]
    assert data["count"] == 5
    assert len(data["similar_movies"]) == 5

    first = data["similar_movies"][0]
    assert "hybrid_similarity" in first
    assert "content_similarity" in first
    assert 862 not in [m["tmdb_id"] for m in data["similar_movies"]]


def test_recommend_similar_movies_not_found(client):
    """Verify /recommend/similar/{tmdb_id} returns 404 for unknown movie."""
    response = client.get("/recommend/similar/999999999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_recommend_similar_movies_invalid_k(client):
    """Verify validation on query parameters."""
    response = client.get("/recommend/similar/862?k=0")
    assert response.status_code == 422


def test_hybrid_search(client):
    """Verify /recommend/hybrid-search endpoint."""
    payload = {
        "query": "friendship and adventure",
        "user_id": "search_tester",
        "k": 5,
    }
    response = client.post("/recommend/hybrid-search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "friendship and adventure"
    assert data["count"] == 5
    assert len(data["results"]) == 5

    first = data["results"][0]
    assert "semantic_score" in first
    assert "hybrid_score" in first


def test_ask_endpoint_schema_and_handling(client):
    """Verify /ask endpoint validates inputs and handles responses."""
    payload = {
        "question": "Recommend 3 uplifting comedy movies.",
        "language": "en",
        "user_id": "ask_tester",
    }
    response = client.post("/ask", json=payload)
    # The endpoint should either succeed (200) with generated text
    # or return 503/502 if external Google GenAI is temporarily busy or rate-limited
    assert response.status_code in (200, 502, 503)
    data = response.json()
    if response.status_code == 200:
        assert data["question"] == payload["question"]
        assert len(data["answer"]) > 0
    else:
        assert "detail" in data
