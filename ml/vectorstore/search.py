
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_local"
COLLECTION_NAME = "movies"
MODEL_NAME = "intfloat/multilingual-e5-base"
VECTOR_SIZE = 768


class MovieSearcher:
    _model = None

    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Prefer Qdrant Cloud when configured.
        if qdrant_url:
            if not qdrant_api_key:
                raise RuntimeError(
                    "QDRANT_API_KEY is required when QDRANT_URL is set."
                )

            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,
            )
            logger.info("Connected to Qdrant Cloud.")

        else:
            # Do not silently use local storage in Render.
            if os.getenv("RENDER"):
                raise RuntimeError(
                    "QDRANT_URL and QDRANT_API_KEY must be configured "
                    "in the Render environment."
                )

            if not QDRANT_PATH.exists():
                raise RuntimeError(
                    f"Local Qdrant storage not found: {QDRANT_PATH}"
                )

            self.client = QdrantClient(path=str(QDRANT_PATH))
            logger.info("Connected to local Qdrant.")

        # Verify the collection before loading the model.
        try:
            collections = self.client.get_collections().collections
            names = {collection.name for collection in collections}

            if COLLECTION_NAME not in names:
                raise RuntimeError(
                    f"Qdrant collection '{COLLECTION_NAME}' was not found."
                )

            info = self.client.get_collection(COLLECTION_NAME)

            if info.config.params.vectors.size != VECTOR_SIZE:
                raise RuntimeError(
                    f"Expected {VECTOR_SIZE}-dimensional vectors, but "
                    f"collection has {info.config.params.vectors.size}."
                )

            logger.info(
                "Verified collection '%s' with %s points.",
                COLLECTION_NAME,
                info.points_count,
            )

        except Exception:
            self.client.close()
            raise

        # Load the embedding model only once per process.
        if MovieSearcher._model is None:
            logger.info("Loading embedding model: %s", MODEL_NAME)
            MovieSearcher._model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded.")

        self.model = MovieSearcher._model

    def search(
        self,
        query,
        top_k=5,
        language=None,
        min_year=None,
        max_year=None,
        min_rating=None,
        min_votes=None,
        max_runtime=None,
        genre=None,
    ):
        """Search movies using multilingual semantic embeddings."""

        query_vector = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )

        query_filter = self._build_filter(
            language=language,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            min_votes=min_votes,
            max_runtime=max_runtime,
            genre=genre,
        )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=max(1, min(int(top_k), 100)),
            with_payload=True,
        )

        # Preserve Claude's original return format: Qdrant ScoredPoint objects.
        return results.points

    @staticmethod
    def _build_filter(
        language=None,
        min_year=None,
        max_year=None,
        min_rating=None,
        min_votes=None,
        max_runtime=None,
        genre=None,
    ):
        conditions = []

        if language:
            conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        if min_year is not None:
            conditions.append(
                models.FieldCondition(
                    key="release_year",
                    range=models.Range(gte=min_year),
                )
            )

        if max_year is not None:
            conditions.append(
                models.FieldCondition(
                    key="release_year",
                    range=models.Range(lte=max_year),
                )
            )

        if min_rating is not None:
            conditions.append(
                models.FieldCondition(
                    key="vote_average",
                    range=models.Range(gte=min_rating),
                )
            )

        if min_votes is not None:
            conditions.append(
                models.FieldCondition(
                    key="vote_count",
                    range=models.Range(gte=min_votes),
                )
            )

        if max_runtime is not None:
            conditions.append(
                models.FieldCondition(
                    key="runtime",
                    range=models.Range(lte=max_runtime),
                )
            )

        if genre:
            conditions.append(
                models.FieldCondition(
                    key="genres",
                    match=models.MatchValue(value=genre),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)


def print_results(query, results):
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):
        movie = result.payload or {}

        print(f"\n{i}. {movie.get('title')}")
        print(f"   TMDB ID: {movie.get('tmdb_id')}")
        print(f"   Language: {movie.get('language')}")
        print(f"   Year: {movie.get('release_year')}")
        print(f"   Rating: {movie.get('vote_average')}")
        print(f"   Genres: {movie.get('genres')}")
        print(f"   Score: {result.score:.4f}")


def main():
    searcher = MovieSearcher()

    try:
        test_cases = [
            ("Interstellar", {}),
            ("3 Idiots", {"language": "hi"}),
            ("Sairat", {"language": "mr"}),
            ("दोस्ती वाली कॉमेडी फिल्म", {"language": "hi"}),
            ("romantic drama", {"language": "mr"}),
            (
                "science fiction",
                {"language": "mr", "min_year": 2100},
            ),
        ]

        for query, filters in test_cases:
            results = searcher.search(query, top_k=5, **filters)
            print_results(query, results)

    finally:
        searcher.client.close()


if __name__ == "__main__":
    main()
