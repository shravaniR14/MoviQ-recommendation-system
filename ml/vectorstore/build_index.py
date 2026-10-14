from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "emb_e5_multilingual.npy"
)

EMBEDDING_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "emb_ids.parquet"
)

MOVIES_TEXT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_text.parquet"
)

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_local"

COLLECTION_NAME = "movies"

VECTOR_SIZE = 768

BATCH_SIZE = 500


# =========================================================
# HELPER
# =========================================================

def clean_payload_value(value):
    """Convert pandas/NumPy values into Qdrant-safe values."""

    # Lists, tuples, NumPy arrays
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    # NumPy scalar
    if isinstance(value, np.generic):
        value = value.item()

    # None / NaN / NaT
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("MOVIQ - Building Local Qdrant Index")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Check files
    # -----------------------------------------------------

    print("\n[1/5] Checking files...")

    required_files = [
        EMBEDDINGS_PATH,
        EMBEDDING_IDS_PATH,
        MOVIES_TEXT_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"✓ {path.name}")

    # -----------------------------------------------------
    # 2. Load embeddings
    # -----------------------------------------------------

    print("\n[2/5] Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got {embeddings.ndim}D."
        )

    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional vectors, "
            f"but found {embeddings.shape[1]}."
        )

    # -----------------------------------------------------
    # 3. Load embedding IDs
    # -----------------------------------------------------

    print("\nLoading embedding IDs...")

    emb_ids = pd.read_parquet(
        EMBEDDING_IDS_PATH
    )

    print(
        f"Embedding ID rows: {len(emb_ids)}"
    )

    print(
        f"Embedding ID columns: {list(emb_ids.columns)}"
    )

    if "tmdb_id" not in emb_ids.columns:
        raise ValueError(
            "emb_ids.parquet must contain 'tmdb_id'."
        )

    # -----------------------------------------------------
    # 4. Load movie metadata
    # -----------------------------------------------------

    print("\nLoading movie metadata...")

    movies = pd.read_parquet(
        MOVIES_TEXT_PATH
    )

    print(
        f"Movie rows: {len(movies)}"
    )

    if "tmdb_id" not in movies.columns:
        raise ValueError(
            "movies_text.parquet must contain 'tmdb_id'."
        )

    # -----------------------------------------------------
    # 5. Validate counts
    # -----------------------------------------------------

    if len(embeddings) != len(emb_ids):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) "
            f"does not match ID count ({len(emb_ids)})."
        )

    print(
        "✓ Embedding and ID counts match."
    )

    # -----------------------------------------------------
    # 6. Merge metadata
    # -----------------------------------------------------

    print("\nMerging movie metadata...")

    data = emb_ids.merge(
        movies,
        on="tmdb_id",
        how="left",
        validate="one_to_one",
    )

    if len(data) != len(embeddings):
        raise ValueError(
            "Merge changed the number of rows."
        )

    if data["title"].isna().any():

        missing_count = int(
            data["title"].isna().sum()
        )

        raise ValueError(
            f"{missing_count} movies could not be "
            f"matched with metadata."
        )

    print(
        f"✓ Final rows for Qdrant: {len(data)}"
    )

    # -----------------------------------------------------
    # 7. Create Qdrant storage
    # -----------------------------------------------------

    print("\nCreating Qdrant local storage...")

    QDRANT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    # -----------------------------------------------------
    # 8. Create collection
    # -----------------------------------------------------

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        print(
            "Deleting old collection..."
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    print(
        f"Creating collection '{COLLECTION_NAME}'..."
    )

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=models.VectorParams(

            size=VECTOR_SIZE,

            distance=models.Distance.COSINE,
        ),
    )

    # -----------------------------------------------------
    # 9. Payload columns
    # -----------------------------------------------------

    payload_columns = [

        "tmdb_id",
        "title",
        "language",
        "release_year",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "genres",
        "keywords",
        "director",
        "cast",
        "overview",
        "poster_path",
    ]

    missing_columns = [
        column
        for column in payload_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing payload columns:\n"
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # 10. Index movies
    # -----------------------------------------------------

    print("\nIndexing movies into Qdrant...")

    total = len(data)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = data.iloc[start:end]

        points = []

        for row_index, (_, row) in enumerate(
            batch.iterrows()
        ):

            vector_index = (
                start + row_index
            )

            payload = {}

            for column in payload_columns:

                value = clean_payload_value(
                    row[column]
                )

                payload[column] = value

            point_id = int(
                row["tmdb_id"]
            )

            points.append(

                models.PointStruct(

                    id=point_id,

                    vector=embeddings[
                        vector_index
                    ].tolist(),

                    payload=payload,
                )
            )

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=points,
        )

        print(
            f"Indexed {end:,}/{total:,} movies"
        )

    # -----------------------------------------------------
    # 11. Verify
    # -----------------------------------------------------

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print("\n")
    print("=" * 60)
    print("QDRANT INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Points     : {collection_info.points_count:,}"
    )

    print(
        f"Vector size: {VECTOR_SIZE}"
    )

    print(
        "Distance   : COSINE"
    )

    print(
        f"Storage    : {QDRANT_PATH}"
    )

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()