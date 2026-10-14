"use client";

import React, { useRef } from "react";
import { Film, Star, ChevronLeft, ChevronRight, Play } from "lucide-react";
import { Movie } from "../lib/types";
import { getTmdbPosterUrl } from "../lib/api";

interface BecauseYouWatchedProps {
  targetTitle: string;
  movies: Movie[];
  onSelectMovie: (movie: Movie) => void;
  onWatchNow: (movie: Movie) => void;
  onViewAll?: () => void;
}

export function BecauseYouWatched({
  targetTitle,
  movies,
  onSelectMovie,
  onWatchNow,
  onViewAll,
}: BecauseYouWatchedProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  if (!movies || movies.length === 0) {
    return null;
  }

  const scrollLeft = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollBy({ left: -320, behavior: "smooth" });
    }
  };

  const scrollRight = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollBy({ left: 320, behavior: "smooth" });
    }
  };

  return (
    <section className="w-full py-4 select-none">
      {/* Header with Title & Carousel Arrows */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <Film className="w-5 h-5 text-[#e50914]" />
          <h2 className="text-xl font-bold text-white tracking-wide">
            Because You Watched <span className="text-[#ff3b45]">{targetTitle}</span>
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {onViewAll && (
            <button
              onClick={onViewAll}
              className="text-xs font-semibold text-zinc-400 hover:text-[#e50914] flex items-center gap-1 mr-3 transition-colors group"
            >
              <span>View All</span>
              <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </button>
          )}

          <button
            onClick={scrollLeft}
            className="w-8 h-8 rounded-full bg-white/[0.06] hover:bg-white/20 border border-white/10 text-white flex items-center justify-center transition-all hover:scale-105 active:scale-95"
            aria-label="Scroll left"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={scrollRight}
            className="w-8 h-8 rounded-full bg-white/[0.06] hover:bg-white/20 border border-white/10 text-white flex items-center justify-center transition-all hover:scale-105 active:scale-95"
            aria-label="Scroll right"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Horizontal Carousel */}
      <div
        ref={scrollContainerRef}
        className="flex items-stretch gap-4 overflow-x-auto no-scrollbar pb-2"
        style={{ scrollSnapType: "x mandatory" }}
      >
        {movies.map((movie) => {
          const genreText =
            movie.genres && movie.genres.length > 0 ? movie.genres[0] : "Thriller";

          return (
            <div
              key={movie.tmdb_id}
              onClick={() => onSelectMovie(movie)}
              style={{ scrollSnapAlign: "start" }}
              className="flex-shrink-0 w-44 md:w-48 group relative flex flex-col rounded-2xl overflow-hidden bg-[#12141e] border border-white/[0.06] hover:border-[#e50914] transition-all duration-300 hover:shadow-[0_0_20px_rgba(229,9,20,0.4)] hover:-translate-y-1.5 cursor-pointer"
            >
              {/* Poster Container */}
              <div className="relative aspect-[2/3] w-full overflow-hidden bg-zinc-900">
                <img
                  src={getTmdbPosterUrl(movie.poster_path, "w500")}
                  alt={movie.title}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  loading="lazy"
                />

                {/* Hover Play Button */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onWatchNow(movie);
                    }}
                    className="w-10 h-10 rounded-full bg-[#e50914] hover:bg-[#ff1e27] text-white flex items-center justify-center shadow-[0_0_15px_rgba(229,9,20,0.6)] transform scale-90 group-hover:scale-100 transition-transform"
                    aria-label={`Play ${movie.title}`}
                  >
                    <Play className="w-4 h-4 fill-white ml-0.5" />
                  </button>
                </div>
              </div>

              {/* Movie Info */}
              <div className="p-3 flex flex-col justify-between flex-1">
                <h3 className="text-sm font-semibold text-white truncate group-hover:text-red-400 transition-colors">
                  {movie.title}
                </h3>

                <div className="flex items-center justify-between text-xs text-zinc-400 mt-1">
                  <div className="flex items-center gap-1.5">
                    <span>{movie.release_year || "2020"}</span>
                    <span>•</span>
                    <span className="truncate max-w-[70px]">{genreText}</span>
                  </div>

                  {movie.vote_average && (
                    <div className="flex items-center gap-1 text-amber-400 font-semibold">
                      <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
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
