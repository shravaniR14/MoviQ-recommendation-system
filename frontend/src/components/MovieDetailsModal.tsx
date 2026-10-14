"use client";

import React, { useState } from "react";
import {
  X,
  Play,
  Star,
  Plus,
  Check,
  Calendar,
  Clock,
  Globe,
  Sparkles,
} from "lucide-react";
import { Movie } from "../lib/types";
import { getTmdbBackdropUrl, getTmdbPosterUrl } from "../lib/api";

interface MovieDetailsModalProps {
  movie: Movie | null;
  isOpen: boolean;
  onClose: () => void;
  onWatchNow: (movie: Movie) => void;
  onRateMovie: (movie: Movie, rating: number) => void;
  onToggleMyList: (movie: Movie) => void;
  isMovieInList: (tmdbId: number) => boolean;
  userRating?: number;
}

export function MovieDetailsModal({
  movie,
  isOpen,
  onClose,
  onWatchNow,
  onRateMovie,
  onToggleMyList,
  isMovieInList,
  userRating,
}: MovieDetailsModalProps) {
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const [ratedSuccess, setRatedSuccess] = useState(false);

  if (!isOpen || !movie) return null;

  const inList = isMovieInList(movie.tmdb_id);

  const handleStarClick = (stars: number) => {
    onRateMovie(movie, stars);
    setRatedSuccess(true);
    setTimeout(() => setRatedSuccess(false), 3000);
  };

  const formatRuntime = (mins?: number | null) => {
    if (!mins) return "2h 05m";
    const h = Math.floor(mins / 60);
    const m = mins % 60;
    return `${h}h ${m}m`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto bg-[#0d0f17] border border-white/[0.1] rounded-3xl shadow-[0_10px_50px_rgba(0,0,0,0.9)] no-scrollbar"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Backdrop Banner */}
        <div className="relative h-64 sm:h-80 w-full overflow-hidden">
          <img
            src={getTmdbBackdropUrl(movie.backdrop_path)}
            alt={movie.title}
            className="w-full h-full object-cover object-top brightness-75 contrast-105"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#0d0f17] via-[#0d0f17]/60 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-r from-[#0d0f17]/80 to-transparent w-1/2" />

          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 z-20 w-9 h-9 rounded-full bg-black/60 hover:bg-[#e50914] text-white border border-white/20 flex items-center justify-center transition-all hover:scale-105"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Floating Match Badge if Hybrid */}
          {movie.hybrid_score && (
            <div className="absolute bottom-4 left-6 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#e50914] text-xs font-bold text-white shadow-[0_0_15px_rgba(229,9,20,0.6)]">
              <Sparkles className="w-3.5 h-3.5" />
              <span>{Math.round(movie.hybrid_score * 100)}% Match</span>
            </div>
          )}
        </div>

        {/* Content Body */}
        <div className="p-6 sm:p-8 -mt-16 sm:-mt-20 relative z-10 flex flex-col md:flex-row gap-6">
          {/* Poster Column */}
          <div className="flex-shrink-0 w-36 sm:w-48 mx-auto md:mx-0">
            <div className="aspect-[2/3] w-full rounded-2xl overflow-hidden shadow-2xl border-2 border-white/10 bg-zinc-900">
              <img
                src={getTmdbPosterUrl(movie.poster_path, "w500")}
                alt={movie.title}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Info Column */}
          <div className="flex-1 flex flex-col justify-between">
            <div>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mb-2">
                {movie.title}
              </h2>

              {movie.tagline && (
                <p className="text-xs sm:text-sm text-zinc-400 italic mb-3">
                  &ldquo;{movie.tagline}&rdquo;
                </p>
              )}

              {/* Quick Specs */}
              <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-300 font-medium mb-4">
                <span className="flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5 text-zinc-400" />
                  {movie.release_year || "Unknown"}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-zinc-400" />
                  {formatRuntime(movie.runtime)}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1 uppercase">
                  <Globe className="w-3.5 h-3.5 text-zinc-400" />
                  {movie.language || "en"}
                </span>

                {movie.vote_average && (
                  <div className="flex items-center gap-1 text-amber-400 font-bold ml-auto">
                    <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
                    <span>{movie.vote_average.toFixed(1)} / 10</span>
                  </div>
                )}
              </div>

              {/* Genres */}
              {movie.genres && movie.genres.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {movie.genres.map((g) => (
                    <span
                      key={g}
                      className="text-xs px-2.5 py-1 rounded-md bg-white/[0.06] text-zinc-200 border border-white/[0.06]"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              )}

              {/* Synopsis */}
              <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed mb-5">
                {movie.overview || "No overview available for this title."}
              </p>

              {/* Cast & Director */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-zinc-400 mb-6 border-t border-white/[0.06] pt-4">
                {movie.director && (
                  <div>
                    <span className="font-semibold text-zinc-300">Director: </span>
                    <span>{movie.director}</span>
                  </div>
                )}
                {movie.cast && movie.cast.length > 0 && (
                  <div>
                    <span className="font-semibold text-zinc-300">Starring: </span>
                    <span className="truncate">{movie.cast.slice(0, 4).join(", ")}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Interactive Rating & Buttons */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-4 border-t border-white/[0.08]">
              {/* Star Rating Widget */}
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                  Rate this movie (ML Profile Update)
                </span>
                <div className="flex items-center gap-1.5">
                  {[1, 2, 3, 4, 5].map((star) => {
                    const isFilled =
                      (hoverRating !== null ? hoverRating : userRating || 0) >= star;
                    return (
                      <button
                        key={star}
                        type="button"
                        onMouseEnter={() => setHoverRating(star)}
                        onMouseLeave={() => setHoverRating(null)}
                        onClick={() => handleStarClick(star)}
                        className="p-1 hover:scale-125 transition-transform"
                        title={`Rate ${star} Stars`}
                      >
                        <Star
                          className={`w-5 h-5 ${
                            isFilled
                              ? "fill-amber-400 text-amber-400 drop-shadow-[0_0_8px_rgba(251,191,36,0.6)]"
                              : "text-zinc-600 hover:text-zinc-400"
                          } transition-colors`}
                        />
                      </button>
                    );
                  })}
                  {ratedSuccess && (
                    <span className="text-[11px] text-green-400 font-medium ml-2 animate-pulse">
                      ✓ Profile Updated!
                    </span>
                  )}
                </div>
              </div>

              {/* Primary Action Buttons */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => onWatchNow(movie)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#e50914] hover:bg-[#ff1e27] text-white text-xs font-semibold transition-all hover:scale-105 shadow-[0_0_15px_rgba(229,9,20,0.5)]"
                >
                  <Play className="w-3.5 h-3.5 fill-white" />
                  <span>Watch Now</span>
                </button>

                <button
                  onClick={() => onToggleMyList(movie)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 rounded-full text-xs font-medium border transition-all ${
                    inList
                      ? "bg-white/20 border-white/40 text-white"
                      : "bg-white/[0.05] hover:bg-white/10 border-white/20 text-white"
                  }`}
                >
                  {inList ? (
                    <Check className="w-3.5 h-3.5 text-green-400" />
                  ) : (
                    <Plus className="w-3.5 h-3.5" />
                  )}
                  <span>{inList ? "In My List" : "Add to List"}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
