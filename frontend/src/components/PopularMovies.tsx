"use client";

import React from "react";
import { Flame, Star, Plus, Check, ChevronRight } from "lucide-react";
import { Movie } from "../lib/types";
import { getTmdbPosterUrl } from "../lib/api";

interface PopularMoviesProps {
  movies: Movie[];
  activeGenre: string;
  onGenreChange: (genre: string) => void;
  onSelectMovie: (movie: Movie) => void;
  onToggleMyList: (movie: Movie) => void;
  isMovieInList: (tmdbId: number) => boolean;
  onViewAll?: () => void;
}

const GENRE_PILLS = [
  "All",
  "Action",
  "Comedy",
  "Drama",
  "Sci-Fi",
  "Thriller",
  "Horror",
  "Romance",
];

export function PopularMovies({
  movies,
  activeGenre,
  onGenreChange,
  onSelectMovie,
  onToggleMyList,
  isMovieInList,
  onViewAll,
}: PopularMoviesProps) {
  return (
    <section className="w-full py-4 select-none">
      <div className="mb-4 flex flex-col justify-between gap-3 md:flex-row md:items-center">
        <div className="flex items-center gap-2.5">
          <Flame className="h-5 w-5 text-[#e50914]" />
          <h2 className="text-xl font-bold tracking-wide text-white">
            Popular Movies
          </h2>
        </div>

        <div className="no-scrollbar flex items-center gap-1.5 overflow-x-auto py-1">
          {GENRE_PILLS.map((genre) => {
            const isActive = activeGenre === genre;

            return (
              <button
                key={genre}
                onClick={() => onGenreChange(genre)}
                className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200 ${isActive
                    ? "bg-[#e50914] font-semibold text-white shadow-[0_0_12px_rgba(229,9,20,0.5)]"
                    : "bg-white/[0.05] text-zinc-300 hover:bg-white/[0.1] hover:text-white"
                  }`}
              >
                {genre}
              </button>
            );
          })}

          {onViewAll && (
            <button
              onClick={onViewAll}
              className="group ml-2 flex items-center gap-1 whitespace-nowrap text-xs font-semibold text-zinc-400 transition-colors hover:text-[#e50914]"
            >
              <span>View All</span>
              <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
            </button>
          )}
        </div>
      </div>

      <div className="no-scrollbar flex gap-3 overflow-x-auto py-2">
        {movies.map((movie) => {
          const inList = isMovieInList(movie.tmdb_id);

          return (
            <div
              key={movie.tmdb_id}
              onClick={() => onSelectMovie(movie)}
              className="group relative flex w-40 flex-shrink-0 cursor-pointer flex-col overflow-hidden rounded-2xl border border-white/[0.06] bg-[#12141e] transition-all duration-300 hover:-translate-y-1 hover:border-[#e50914] hover:shadow-[0_0_18px_rgba(229,9,20,0.35)] sm:w-48 md:w-52 lg:w-56"
            >
              <div className="relative aspect-[2/3] w-full overflow-hidden bg-zinc-900">
                <img
                  src={getTmdbPosterUrl(movie.poster_path, "w342")}
                  alt={movie.title}
                  className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                  loading="lazy"
                />

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleMyList(movie);
                  }}
                  className={`absolute right-2 top-2 z-10 flex h-7 w-7 items-center justify-center rounded-full transition-all ${inList
                      ? "bg-[#e50914] text-white shadow-[0_0_8px_#e50914]"
                      : "bg-black/60 text-white backdrop-blur-sm hover:bg-white/20"
                    }`}
                  aria-label={
                    inList ? "Remove from watchlist" : "Add to watchlist"
                  }
                  title={inList ? "In My List" : "Add to List"}
                >
                  {inList ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : (
                    <Plus className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>

              <div className="flex flex-1 flex-col justify-between p-2.5">
                <h3 className="truncate text-xs font-semibold text-white transition-colors group-hover:text-red-400">
                  {movie.title}
                </h3>

                <div className="mt-1 flex items-center justify-between text-[11px] text-zinc-400">
                  <span>{movie.release_year || "Unknown year"}</span>

                  {movie.vote_average != null && movie.vote_average > 0 && (
                    <div className="flex items-center gap-1 font-semibold text-amber-400">
                      <Star className="h-2.5 w-2.5 fill-amber-400 text-amber-400" />
                      <span>{movie.vote_average.toFixed(1)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}