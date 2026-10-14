"""Baseline content-based recommender: TF-IDF + cosine similarity."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[2]


class TfidfRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        # Custom token_pattern keeps Devanagari (Hindi/Marathi) words intact.
        self.vec = TfidfVectorizer(
            token_pattern=r"(?u)[\w\u0900-\u097F]+",
            ngram_range=(1, 2), min_df=2, max_features=100_000,
            sublinear_tf=True, strip_accents=None,
        )
        self.matrix = self.vec.fit_transform(self.df.movie_text)

    def find(self, title: str) -> int | None:
        hits = self.df.index[self.df.title.str.lower() == title.lower()]
        if len(hits) == 0:
            return None
        # If several movies share a title, take the one with most votes.
        return self.df.loc[hits, "vote_count"].idxmax()

    def similar(self, title: str, k: int = 10) -> pd.DataFrame:
        idx = self.find(title)
        if idx is None:
            return pd.DataFrame()
        sims = linear_kernel(self.matrix[idx], self.matrix).ravel()
        sims[idx] = -1  # exclude the movie itself
        top = sims.argsort()[::-1][:k]
        out = self.df.loc[top, ["title", "language", "release_year", "vote_average"]].copy()
        out["score"] = sims[top].round(3)
        return out


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "processed" / "movies_text.parquet")
    rec = TfidfRecommender(df)
    print("Matrix:", rec.matrix.shape)
    for t in ["Interstellar", "The Dark Knight", "3 Idiots", "Sairat"]:
        res = rec.similar(t, 5)
        print(f"\nSimilar to {t}:")
        print(res.to_string(index=False) if len(res) else "  (title not found)")
