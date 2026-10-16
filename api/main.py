"""
main.py
-------
FastAPI REST API for Moviq: AI-powered Movie Discovery and Hybrid Recommendation System.
Provides:
- GET  /                                  (API status)
- GET  /health                            (Health check)
- POST /ask                               (ML-backed RAG movie assistant)
- POST /recommend/personalized            (Personalized hybrid recommendations)
- GET  /recommend/similar/{tmdb_id}       (Item-to-item hybrid similar movies)
- POST /recommend/hybrid-search           (Semantic vector search with personal re-ranking)
- POST /user/interaction                  (Log ratings, watches, and searches)
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from google.genai.errors import APIError

from ml.models.hybrid import HybridRecommender
from ml.rag.generator import MoviqAssistant

logger = logging.getLogger("moviq.api")

app = FastAPI(
    title="Moviq API",
    description="AI-powered movie discovery and hybrid ML recommendation API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://movi-q-recommendation-system.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
hybrid_recommender: HybridRecommender | None = None
assistant: MoviqAssistant | None = None


def get_hybrid_recommender() -> HybridRecommender:
    """Lazy-load or access singleton HybridRecommender."""
    global hybrid_recommender
    if hybrid_recommender is None:
        hybrid_recommender = HybridRecommender()
    return hybrid_recommender


def get_assistant() -> MoviqAssistant:
    """Lazy-load or access singleton MoviqAssistant."""
    global assistant
    if assistant is None:
        rec = get_hybrid_recommender()
        try:
            assistant = MoviqAssistant(hybrid_recommender=rec)
        except Exception as exc:
            logger.warning(f"Failed to initialize AI assistant: {exc}")
            assistant = None
    return assistant


@app.on_event("startup")
def load_models_on_startup():
    """Warm up recommendation models on application startup."""
    try:
        get_hybrid_recommender()
        get_assistant()
        logger.info("Moviq models successfully loaded on startup.")
    except Exception as exc:
        logger.error(f"Error during model startup: {exc}")


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class MovieQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=500, description="User's query or movie discovery question")
    language: str = Field(default="hi", pattern="^(en|hi|mr)$", description="Target response language (en, hi, mr)")
    user_id: str | None = Field(default=None, max_length=100, description="Optional user ID for personalized context")


class PersonalizedRecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100, description="User identifier")
    k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    language: str | None = Field(default=None, pattern="^(en|hi|mr)$", description="Filter by language (en, hi, mr)")
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0, description="Minimum TMDB vote average threshold")


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Natural language search or theme query")
    user_id: str | None = Field(default=None, max_length=100, description="Optional user ID for personalized re-ranking")
    k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    language: str | None = Field(default=None, pattern="^(en|hi|mr)$", description="Filter by language (en, hi, mr)")
    min_year: int | None = Field(default=None, ge=1900, le=2030, description="Earliest release year")
    max_year: int | None = Field(default=None, ge=1900, le=2030, description="Latest release year")
    min_rating: float | None = Field(default=None, ge=0.0, le=10.0, description="Minimum vote average threshold")
    min_votes: int | None = Field(default=None, ge=0, description="Minimum vote count threshold")
    genre: str | None = Field(default=None, max_length=50, description="Filter by specific genre")


class UserInteractionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100, description="User identifier")
    interaction_type: str = Field(pattern="^(rating|watch|search)$", description="Type of interaction")
    tmdb_id: int | None = Field(default=None, description="TMDB movie ID (required for rating and watch)")
    rating: float | None = Field(default=None, ge=0.5, le=5.0, description="Rating value 0.5 to 5.0 (required for rating)")
    query: str | None = Field(default=None, max_length=500, description="Search query string (required for search)")

    @model_validator(mode="after")
    def validate_interaction(self):
        if self.interaction_type == "rating":
            if self.tmdb_id is None:
                raise ValueError("tmdb_id is required when interaction_type is 'rating'")
            if self.rating is None:
                raise ValueError("rating is required when interaction_type is 'rating'")
        elif self.interaction_type == "watch":
            if self.tmdb_id is None:
                raise ValueError("tmdb_id is required when interaction_type is 'watch'")
        elif self.interaction_type == "search":
            if not self.query or not self.query.strip():
                raise ValueError("query is required when interaction_type is 'search'")
        return self


# =====================================================================
# SYSTEM & HEALTH ENDPOINTS
# =====================================================================

@app.get("/", tags=["System"])
def home():
    """Root welcoming endpoint."""
    return {
        "message": "Welcome to Moviq API!",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health", tags=["System"])
def health():
    """Health check endpoint."""
    rec = hybrid_recommender is not None
    return {
        "status": "healthy",
        "recommender_ready": rec,
        "assistant_ready": assistant is not None,
    }


# =====================================================================
# RAG ASSISTANT ENDPOINT (ML-BACKED)
# =====================================================================

@app.post("/ask", tags=["Assistant"])
def ask_movie_question(request: MovieQuestion):
    """
    AI-powered movie assistant answering natural language questions
    grounded in ML-retrieved and personalized movie context.
    """
    curr_assistant = get_assistant()
    if curr_assistant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Movie assistant is not ready yet.",
        )

    try:
        answer = curr_assistant.ask(
            question=request.question,
            language=request.language,
            user_id=request.user_id,
        )

        return {
            "question": request.question,
            "language": request.language,
            "user_id": request.user_id,
            "answer": answer,
        }

    except APIError as exc:
        # Log detailed information for debugging without exposing credentials
        error_code = getattr(exc, "code", None)
        logger.error(f"Gemini APIError encountered: {exc!r}, code={error_code}")
        # Log full traceback (does not expose secrets)
        logger.exception("Full Gemini APIError traceback")
        if error_code in (503, 429):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is temporarily busy or rate-limited. Please try again shortly.",
            ) from exc

        # For other API errors, treat as a bad gateway and include minimal debug info in logs
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI service could not process your request.",
        ) from exc

    except Exception as exc:
        logger.error(f"Error in ask endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred in the movie assistant.",
        ) from exc


# =====================================================================
# RECOMMENDATION ENDPOINTS
# =====================================================================

@app.post("/recommend/personalized", tags=["Recommendations"])
def recommend_personalized(request: PersonalizedRecommendationRequest):
    """
    Generate personalized recommendations for a user.
    Fuses Collaborative Filtering (SVD/fold-in), Content Embeddings (taste centroid),
    and Bayesian quality priors based on user maturity (cold vs. nascent vs. warm).
    """
    try:
        rec = get_hybrid_recommender()
        profile = rec.get_or_create_profile(request.user_id)
        df_recs = rec.recommend(
            user_id=request.user_id,
            k=request.k,
            language=request.language,
            min_rating=request.min_rating,
        )

        return {
            "user_id": request.user_id,
            "maturity": profile.maturity,
            "count": len(df_recs),
            "recommendations": df_recs.to_dict(orient="records") if not df_recs.empty else [],
        }
    except Exception as exc:
        logger.error(f"Error generating personalized recommendations: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate personalized recommendations.",
        ) from exc


@app.get("/recommend/similar/{tmdb_id}", tags=["Recommendations"])
def recommend_similar_movies(
    tmdb_id: int = Path(..., description="Target movie TMDB ID"),
    k: int = Query(default=10, ge=1, le=50, description="Number of similar movies to return"),
    language: str | None = Query(default=None, pattern="^(en|hi|mr)$", description="Filter by language"),
):
    """
    Find movies similar to a target movie, combining:
    - Dense Multilingual Content Embeddings (narrative, mood, themes)
    - Item-to-Item Collaborative Filtering (users who co-rated both movies)
    """
    rec = get_hybrid_recommender()
    if int(tmdb_id) not in rec.tmdb_to_idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with TMDB ID {tmdb_id} not found in catalog.",
        )

    try:
        df_sims = rec.similar_movies_hybrid(
            identifier=tmdb_id,
            k=k,
            language=language,
        )

        target_movie = rec.content_rec.get_movie_by_idx(rec.tmdb_to_idx[int(tmdb_id)])

        return {
            "tmdb_id": tmdb_id,
            "target_title": target_movie.get("title"),
            "count": len(df_sims),
            "similar_movies": df_sims.to_dict(orient="records") if not df_sims.empty else [],
        }
    except Exception as exc:
        logger.error(f"Error finding similar movies for {tmdb_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve similar movies.",
        ) from exc


@app.post("/recommend/hybrid-search", tags=["Recommendations"])
def hybrid_search(request: HybridSearchRequest):
    """
    Personalized Hybrid Search:
    Retrieves candidates using Qdrant multilingual vector search,
    then re-ranks candidates using the user's collaborative and content taste profile.
    """
    try:
        rec = get_hybrid_recommender()
        df_results = rec.search(
            query=request.query,
            user_id=request.user_id,
            k=request.k,
            language=request.language,
            min_year=request.min_year,
            max_year=request.max_year,
            min_rating=request.min_rating,
            min_votes=request.min_votes,
            genre=request.genre,
        )

        return {
            "query": request.query,
            "user_id": request.user_id,
            "count": len(df_results),
            "results": df_results.to_dict(orient="records") if not df_results.empty else [],
        }
    except Exception as exc:
        logger.error(f"Error in hybrid search: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute hybrid search.",
        ) from exc


# =====================================================================
# USER INTERACTION & PROFILE ENDPOINT
# =====================================================================

@app.post("/user/interaction", tags=["User Profile"])
def record_user_interaction(request: UserInteractionRequest):
    """
    Record an explicit rating, watch event, or search query.
    Dynamically updates the user's taste embedding vector and genre affinities.
    """
    try:
        rec = get_hybrid_recommender()
        profile = rec.record_interaction(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            tmdb_id=request.tmdb_id,
            rating=request.rating,
            query=request.query,
        )

        return {
            "status": "success",
            "user_id": request.user_id,
            "interaction_recorded": request.interaction_type,
            "maturity": profile.maturity,
            "total_ratings": len(profile.ratings),
            "total_watched": len(profile.watched),
            "top_genres": profile.get_top_genres(n=5),
            "preferred_languages": profile.get_preferred_languages(n=2),
        }
    except Exception as exc:
        logger.error(f"Error recording interaction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record user interaction.",
        ) from exc


# =====================================================================
# MOVIE CATALOG & DISCOVERY ENDPOINTS
# =====================================================================

@app.get("/movies/popular", tags=["Catalog"])
def get_popular_movies(
    genre: str | None = Query(default=None, description="Optional genre filter"),
    language: str | None = Query(default=None, pattern="^(en|hi|mr)$", description="Language filter"),
    k: int = Query(default=16, ge=1, le=50, description="Number of popular movies to return"),
):
    """
    Get top popular movies from the enriched catalog for discovery grids and hero carousels.
    """
    rec = get_hybrid_recommender()
    df = rec.df

    if language:
        df = df[df["language"] == language]

    if genre and genre.lower() != "all":
        g_lower = genre.lower()
        df = df[df["genres"].apply(lambda g_list: any(isinstance(g_list, (list, np.ndarray)) and g_lower in [str(x).lower() for x in g_list]))]

    sorted_df = df.sort_values(by=["popularity", "vote_average"], ascending=[False, False]).head(k)

    results = []
    for _, row in sorted_df.iterrows():
        results.append({
            "tmdb_id": int(row["tmdb_id"]),
            "title": str(row["title"]),
            "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "language": str(row["language"]),
            "genres": list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
            "vote_average": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
            "vote_count": int(row["vote_count"]) if pd.notna(row.get("vote_count")) else 0,
            "popularity": float(row["popularity"]) if pd.notna(row.get("popularity")) else 0.0,
            "poster_path": str(row["poster_path"]) if pd.notna(row.get("poster_path")) else None,
            "backdrop_path": str(row["backdrop_path"]) if pd.notna(row.get("backdrop_path")) else None,
            "overview": str(row["overview"]) if pd.notna(row.get("overview")) else "",
            "runtime": int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
            "cast": list(row["cast"]) if isinstance(row.get("cast"), (list, np.ndarray)) else [],
            "director": str(row["director"]) if pd.notna(row.get("director")) else None,
        })
    return {"count": len(results), "movies": results}


@app.get("/movies/{tmdb_id}", tags=["Catalog"])
def get_movie_details(tmdb_id: int = Path(..., description="TMDB ID")):
    """
    Get full metadata for a single movie by TMDB ID.
    """
    rec = get_hybrid_recommender()
    if int(tmdb_id) not in rec.tmdb_to_idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with TMDB ID {tmdb_id} not found in catalog.",
        )
    idx = rec.tmdb_to_idx[int(tmdb_id)]
    row = rec.df.iloc[idx]
    return {
        "tmdb_id": int(row["tmdb_id"]),
        "title": str(row["title"]),
        "original_title": str(row.get("original_title", row["title"])),
        "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
        "language": str(row["language"]),
        "genres": list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
        "vote_average": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
        "vote_count": int(row["vote_count"]) if pd.notna(row.get("vote_count")) else 0,
        "popularity": float(row["popularity"]) if pd.notna(row.get("popularity")) else 0.0,
        "poster_path": str(row["poster_path"]) if pd.notna(row.get("poster_path")) else None,
        "backdrop_path": str(row["backdrop_path"]) if pd.notna(row.get("backdrop_path")) else None,
        "overview": str(row["overview"]) if pd.notna(row.get("overview")) else "",
        "tagline": str(row["tagline"]) if pd.notna(row.get("tagline")) else "",
        "runtime": int(row["runtime"]) if pd.notna(row.get("runtime")) else None,
        "cast": list(row["cast"]) if isinstance(row.get("cast"), (list, np.ndarray)) else [],
        "director": str(row["director"]) if pd.notna(row.get("director")) else None,
    }

