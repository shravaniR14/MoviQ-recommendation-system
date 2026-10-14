"""Builds one text document per movie (used for TF-IDF now, embeddings later)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def to_list(x) -> list[str]:
    # Parquet returns numpy arrays / None; normalise to a plain list.
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [str(i) for i in list(x) if str(i).strip()]


def movie_document(row) -> str:
    # Skip empty parts so missing keywords/cast don't add noise.
    parts = [f"Title: {row['title']}"]
    if row.get("original_title") and row["original_title"] != row["title"]:
        parts.append(f"Original title: {row['original_title']}")
    if row.get("overview"):
        parts.append(f"Overview: {row['overview']}")
    if row.get("tagline"):
        parts.append(f"Tagline: {row['tagline']}")
    for label, col in [("Genres", "genres"), ("Keywords", "keywords"),
                       ("Director", "director"), ("Cast", "cast")]:
        vals = to_list(row.get(col))
        if col == "director" and isinstance(row.get(col), str):
            vals = [row[col]]
        if vals:
            parts.append(f"{label}: {', '.join(vals)}")
    return "\n".join(parts)


def main() -> None:
    path = PROCESSED / "movies_enriched.parquet"
    df = pd.read_parquet(path if path.exists() else PROCESSED / "movies.parquet")
    df["movie_text"] = df.apply(movie_document, axis=1)
    df.to_parquet(PROCESSED / "movies_text.parquet", index=False)
    print(f"Saved {len(df):,} rows. Sample:\n")
    print(df.movie_text.iloc[0])
    print("\nMean text length:", int(df.movie_text.str.len().mean()))


if __name__ == "__main__":
    main()
