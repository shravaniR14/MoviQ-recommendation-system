from ml.vectorstore.search import MovieSearcher


class MovieContextBuilder:

    def __init__(self, hybrid_recommender=None):
        self.hybrid_recommender = hybrid_recommender
        self._searcher = None if hybrid_recommender is not None else MovieSearcher()

    @property
    def searcher(self):
        if self._searcher is None:
            self._searcher = MovieSearcher()
        return self._searcher

    def retrieve(self, query, top_k=5, language=None, user_id=None):
        if self.hybrid_recommender is not None:
            df = self.hybrid_recommender.search(
                query=query,
                user_id=user_id,
                k=top_k,
                language=language,
            )
            if df.empty:
                return []

            context = []
            for _, row in df.iterrows():
                tmdb_id = row.get("tmdb_id")
                movie_dict = {}
                if tmdb_id is not None and int(tmdb_id) in self.hybrid_recommender.tmdb_to_idx:
                    idx = self.hybrid_recommender.tmdb_to_idx[int(tmdb_id)]
                    movie_dict = self.hybrid_recommender.content_rec.get_movie_by_idx(idx)

                context.append({
                    "title": row.get("title") or movie_dict.get("title"),
                    "year": row.get("release_year") or movie_dict.get("release_year"),
                    "language": row.get("language") or movie_dict.get("language"),
                    "rating": row.get("vote_average") or movie_dict.get("vote_average"),
                    "genres": row.get("genres") or movie_dict.get("genres"),
                    "director": movie_dict.get("director"),
                    "cast": movie_dict.get("cast"),
                    "overview": movie_dict.get("overview", ""),
                    "hybrid_score": row.get("hybrid_score"),
                })
            return context

        results = self.searcher.search(
            query=query,
            top_k=top_k,
            language=language,
        )

        context = []
        for result in results:
            movie = result.payload
            context.append({
                "title": movie.get("title"),
                "year": movie.get("release_year"),
                "language": movie.get("language"),
                "rating": movie.get("vote_average"),
                "genres": movie.get("genres"),
                "director": movie.get("director"),
                "cast": movie.get("cast"),
                "overview": movie.get("overview"),
            })

        return context


def main():
    builder = MovieContextBuilder()

    query = "movies about friendship and college life"

    results = builder.retrieve(
        query=query,
        top_k=5,
        language="hi",
    )

    print("\n" + "=" * 70)
    print("RETRIEVED MOVIE CONTEXT")
    print("=" * 70)

    for i, movie in enumerate(results, 1):
        print(f"\n{i}. {movie['title']} ({movie['year']})")
        print(f"   Genres: {movie['genres']}")
        print(f"   Rating: {movie['rating']}")
        print(f"   Overview: {movie['overview']}")


if __name__ == "__main__":
    main()