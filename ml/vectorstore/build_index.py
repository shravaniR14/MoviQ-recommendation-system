
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
    / "emb_e5_multilingual.npy"
)

EMBEDDING_IDS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
    / "emb_ids.parquet"
)

MOVIES_TEXT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "movies_text.parquet"
)

COLLECTION_NAME = "movies"
VECTOR_SIZE = 768
BATCH_SIZE = 500

load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# HELPERS
# =========================================================

def clean_payload_value(value):
    """Convert pandas and NumPy values to Qdrant-safe values."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_payload_value(item) for item in value]

    if isinstance(value, np.generic):
        value = value.item()

    if value is None or value is pd.NA:
        return None

    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
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
    print("MOVIQ - Qdrant Cloud Index Builder")
    print("=" * 60)

    # 1. Validate environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "QDRANT_URL or QDRANT_API_KEY is missing. "
            "Check your project .env file."
        )

    # 2. Validate local data files
    print("\n[1/6] Checking data files...")

    required_files = [
        EMBEDDINGS_PATH,
        EMBEDDING_IDS_PATH,
        MOVIES_TEXT_PATH,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"Found: {path.name}")

    # 3. Load existing embeddings and metadata
    print("\n[2/6] Loading existing embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    if embeddings.ndim != 2 or embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Expected embeddings shaped (N, {VECTOR_SIZE}); "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    emb_ids = pd.read_parquet(EMBEDDING_IDS_PATH)
    movies = pd.read_parquet(MOVIES_TEXT_PATH)

    for name, frame in [
        ("Embedding IDs", emb_ids),
        ("Movie metadata", movies),
    ]:
        if "tmdb_id" not in frame.columns:
            raise ValueError(f"{name} must contain a 'tmdb_id' column.")

        if frame["tmdb_id"].isna().any():
            raise ValueError(f"{name} contains missing tmdb_id values.")

        if frame["tmdb_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate tmdb_id values.")

    if len(embeddings) != len(emb_ids):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match "
            f"ID count ({len(emb_ids)})."
        )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding ID rows: {len(emb_ids):,}")
    print(f"Movie metadata rows: {len(movies):,}")

    # 4. Merge metadata and validate payload
    print("\n[3/6] Validating movie metadata...")

    data = emb_ids.merge(
        movies,
        on="tmdb_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((data["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} embedding IDs have no matching movie metadata."
        )

    data = data.drop(columns=["_merge"])

    if len(data) != len(embeddings):
        raise ValueError("Metadata merge changed the embedding count.")

    if "title" not in data.columns or data["title"].isna().any():
        raise ValueError("Movie metadata contains missing titles.")

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
        column for column in payload_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required payload columns: "
            + ", ".join(missing_columns)
        )

    # Qdrant point IDs must be valid integers and unique.
    numeric_ids = pd.to_numeric(data["tmdb_id"], errors="raise")

    if not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError("tmdb_id values must be integers.")

    point_ids = numeric_ids.astype("int64")

    if point_ids.duplicated().any():
        raise ValueError("Duplicate Qdrant point IDs detected.")

    print(f"Validated movies: {len(data):,}")

    # 5. Connect to Qdrant Cloud
    print("\n[4/6] Connecting to Qdrant Cloud...")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60,
    )

    try:
        collections = client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        # Safety: never delete or overwrite an existing collection.
        if COLLECTION_NAME in existing_names:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' already exists in Qdrant "
                "Cloud. No data was deleted or uploaded. Verify the "
                "collection before deciding how to proceed."
            )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print(f"Created collection: {COLLECTION_NAME}")

        # 6. Upload vectors in batches
        print("\n[5/6] Uploading existing movie vectors...")

        total = len(data)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = data.iloc[start:end]

            points = []

            for row_index, (_, row) in enumerate(batch.iterrows()):
                vector_index = start + row_index

                payload = {
                    column: clean_payload_value(row[column])
                    for column in payload_columns
                }

                vector = np.asarray(
                    embeddings[vector_index], dtype=np.float32
                )

                points.append(
                    models.PointStruct(
                        id=int(point_ids.iloc[vector_index]),
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            print(f"Uploaded {end:,}/{total:,} movies")

        # Verify upload
        print("\n[6/6] Verifying cloud collection...")

        info = client.get_collection(COLLECTION_NAME)
        actual_count = info.points_count

        print("=" * 60)

        if actual_count != total:
            raise RuntimeError(
                f"Upload verification failed: expected {total:,} points, "
                f"found {actual_count:,}."
            )

        print("QDRANT CLOUD INDEX CREATED SUCCESSFULLY")
        print(f"Collection : {COLLECTION_NAME}")
        print(f"Points     : {actual_count:,}")
        print(f"Vector size: {VECTOR_SIZE}")
        print("Distance   : COSINE")
        print("=" * 60)

    finally:
        client.close()


if __name__ == "__main__":
    main()
