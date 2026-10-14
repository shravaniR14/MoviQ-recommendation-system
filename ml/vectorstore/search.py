from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_local"

COLLECTION_NAME = "movies"

MODEL_NAME = "intfloat/multilingual-e5-base"

VECTOR_SIZE = 768


class MovieSearcher:

    def __init__(self):

        self.client = QdrantClient(
            path=str(QDRANT_PATH)
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

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
            limit=top_k,
            with_payload=True,
        )

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
                    match=models.MatchValue(
                        value=language
                    ),
                )
            )

        if min_year is not None:
            conditions.append(
                models.FieldCondition(
                    key="release_year",
                    range=models.Range(
                        gte=min_year
                    ),
                )
            )

        if max_year is not None:
            conditions.append(
                models.FieldCondition(
                    key="release_year",
                    range=models.Range(
                        lte=max_year
                    ),
                )
            )

        if min_rating is not None:
            conditions.append(
                models.FieldCondition(
                    key="vote_average",
                    range=models.Range(
                        gte=min_rating
                    ),
                )
            )

        if min_votes is not None:
            conditions.append(
                models.FieldCondition(
                    key="vote_count",
                    range=models.Range(
                        gte=min_votes
                    ),
                )
            )

        if max_runtime is not None:
            conditions.append(
                models.FieldCondition(
                    key="runtime",
                    range=models.Range(
                        lte=max_runtime
                    ),
                )
            )

        if genre:
            conditions.append(
                models.FieldCondition(
                    key="genres",
                    match=models.MatchValue(
                        value=genre
                    ),
                )
            )

        if not conditions:
            return None

        return models.Filter(
            must=conditions
        )


def print_results(
    query,
    results,
):

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(
        results,
        start=1,
    ):

        movie = result.payload

        print(
            f"\n{i}. {movie.get('title')}"
        )

        print(
            f"   TMDB ID: {movie.get('tmdb_id')}"
        )

        print(
            f"   Language: {movie.get('language')}"
        )

        print(
            f"   Year: {movie.get('release_year')}"
        )

        print(
            f"   Rating: {movie.get('vote_average')}"
        )

        print(
            f"   Genres: {movie.get('genres')}"
        )

        print(
            f"   Score: {result.score:.4f}"
        )


def main():

    searcher = MovieSearcher()

    # -----------------------------------------------------
    # 1. Basic English search
    # -----------------------------------------------------

    query = "Interstellar"

    results = searcher.search(
        query,
        top_k=5,
    )

    print_results(
        query,
        results,
    )

    # -----------------------------------------------------
    # 2. Hindi movie
    # -----------------------------------------------------

    query = "3 Idiots"

    results = searcher.search(
        query,
        top_k=5,
        language="hi",
    )

    print_results(
        query,
        results,
    )

    # -----------------------------------------------------
    # 3. Marathi movie
    # -----------------------------------------------------

    query = "Sairat"

    results = searcher.search(
        query,
        top_k=5,
        language="mr",
    )

    print_results(
        query,
        results,
    )

    # -----------------------------------------------------
    # 4. Hindi natural-language query
    # -----------------------------------------------------

    query = "दोस्ती वाली कॉमेडी फिल्म"

    results = searcher.search(
        query,
        top_k=5,
        language="hi",
    )

    print_results(
        query,
        results,
    )

    # -----------------------------------------------------
    # 5. Marathi filter
    # -----------------------------------------------------

    query = "romantic drama"

    results = searcher.search(
        query,
        top_k=5,
        language="mr",
    )

    print_results(
        query,
        results,
    )

    # -----------------------------------------------------
    # 6. Impossible filter
    # -----------------------------------------------------

    query = "science fiction"

    results = searcher.search(
        query,
        top_k=5,
        language="mr",
        min_year=2100,
    )

    print_results(
        query,
        results,
    )


if __name__ == "__main__":
    main()