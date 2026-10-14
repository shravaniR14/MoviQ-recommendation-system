"use client";

import React, { useState } from "react";
import { Play, Plus, Check, ChevronLeft, ChevronRight } from "lucide-react";
import { Movie } from "../lib/types";
import { getTmdbBackdropUrl } from "../lib/api";

interface HeroProps {
  movies: Movie[];
  onWatchNow: (movie: Movie) => void;
  onToggleMyList: (movie: Movie) => void;
  isMovieInList: (tmdbId: number) => boolean;
  onSelectMovie: (movie: Movie) => void;
}

export function Hero({
  movies,
  onWatchNow,
  onToggleMyList,
  isMovieInList,
  onSelectMovie,
}: HeroProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  if (!movies || movies.length === 0) {
    return (
      <div className="w-full h-[460px] rounded-3xl bg-[#12141d] animate-pulse flex items-center justify-center text-zinc-500">
        Loading featured titles...
      </div>
    );
  }

  const currentMovie = movies[currentIndex] || movies[0];
  const inList = isMovieInList(currentMovie.tmdb_id);

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev === 0 ? movies.length - 1 : prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev === movies.length - 1 ? 0 : prev + 1));
  };

  // Format runtime
  const formatRuntime = (mins?: number | null) => {
    if (!mins) return "2h 10m";
    const h = Math.floor(mins / 60);
    const m = mins % 60;
    return `${h}h ${m}m`;
  };

  // Get sample cast members for the Starring widget
  const castList =
    currentMovie.cast && currentMovie.cast.length > 0
      ? currentMovie.cast.slice(0, 3)
      : ["Tom Hiddleston", "Brie Larson", "Samuel L. Jackson"];

  return (
    <div className="relative w-full h-[480px] lg:h-[520px] rounded-3xl overflow-hidden shadow-2xl select-none group border border-white/[0.08]">
      {/* Background Backdrop Image */}
      <img
        src={getTmdbBackdropUrl(currentMovie.backdrop_path)}
        alt={currentMovie.title}
        className="absolute inset-0 w-full h-full object-cover object-top transition-all duration-700 brightness-[0.75] contrast-[1.05]"
      />

      {/* Cinematic Gradient Overlays */}
      <div className="absolute inset-0 bg-gradient-to-r from-[#08090d] via-[#08090d]/80 to-transparent w-[75%]" />
      <div className="absolute inset-0 bg-gradient-to-t from-[#08090d] via-[#08090d]/40 to-transparent h-full" />
      <div className="absolute inset-0 bg-radial-at-c from-transparent via-transparent to-[#08090d]/60" />

      {/* Left & Right Slider Arrows */}
      <button
        onClick={handlePrev}
        className="absolute left-4 top-1/2 -translate-y-1/2 z-20 w-11 h-11 rounded-full bg-black/40 hover:bg-[#e50914] text-white border border-white/10 hover:border-[#e50914] flex items-center justify-center transition-all opacity-80 hover:opacity-100 hover:scale-105 hover:shadow-[0_0_15px_rgba(229,9,20,0.6)]"
        aria-label="Previous featured movie"
      >
        <ChevronLeft className="w-5 h-5" />
      </button>
      <button
        onClick={handleNext}
        className="absolute right-4 top-1/2 -translate-y-1/2 z-20 w-11 h-11 rounded-full bg-black/40 hover:bg-[#e50914] text-white border border-white/10 hover:border-[#e50914] flex items-center justify-center transition-all opacity-80 hover:opacity-100 hover:scale-105 hover:shadow-[0_0_15px_rgba(229,9,20,0.6)]"
        aria-label="Next featured movie"
      >
        <ChevronRight className="w-5 h-5" />
      </button>

      {/* Hero Content */}
      <div className="relative z-10 h-full flex flex-col justify-between p-8 lg:p-12 max-w-3xl">
        <div className="flex-1 flex flex-col justify-center">
          {/* Movie Title */}
          <h1
            onClick={() => onSelectMovie(currentMovie)}
            className="text-4xl lg:text-5xl font-extrabold text-white tracking-tight leading-none mb-3 cursor-pointer hover:text-red-400 transition-colors uppercase drop-shadow-md line-clamp-2"
          >
            {currentMovie.title}
          </h1>

          {/* Metadata line */}
          <div className="flex flex-wrap items-center gap-3 text-xs lg:text-sm text-zinc-300 font-medium mb-4">
            <span className="text-zinc-200">{currentMovie.release_year || "2024"}</span>
            <span>•</span>
            <span>{formatRuntime(currentMovie.runtime)}</span>
            <span>•</span>
            <span className="px-1.5 py-0.5 rounded border border-zinc-500/50 text-[11px] font-semibold text-zinc-300">
              PG-13
            </span>
            <span>|</span>
            <span className="text-zinc-300">
              {currentMovie.genres && currentMovie.genres.length > 0
                ? currentMovie.genres.slice(0, 3).join(", ")
                : "Action, Adventure"}
            </span>
          </div>

          {/* Synopsis */}
          <p className="text-xs lg:text-sm text-zinc-300 line-clamp-3 leading-relaxed max-w-xl mb-6 drop-shadow">
            {currentMovie.overview ||
              "A team of explorers venture into a mythical uncharted island, encountering gigantic creatures and uncovering its ancient hidden secrets."}
          </p>

          {/* Action Buttons */}
          <div className="flex items-center gap-3.5">
            <button
              onClick={() => onWatchNow(currentMovie)}
              className="flex items-center gap-2.5 px-6 py-3 rounded-full bg-[#e50914] hover:bg-[#ff1e27] text-white text-sm font-semibold transition-all duration-200 hover:scale-105 shadow-[0_0_20px_rgba(229,9,20,0.5)] active:scale-95"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>Watch Now</span>
            </button>

            <button
              onClick={() => onToggleMyList(currentMovie)}
              className={`flex items-center gap-2 px-5 py-3 rounded-full text-sm font-medium border transition-all duration-200 ${
                inList
                  ? "bg-white/20 border-white/40 text-white"
                  : "bg-black/40 hover:bg-white/10 border-white/20 text-white hover:border-white/40"
              }`}
            >
              {inList ? <Check className="w-4 h-4 text-green-400" /> : <Plus className="w-4 h-4" />}
              <span>{inList ? "In My List" : "Add to List"}</span>
            </button>
          </div>
        </div>

        {/* Bottom Slide Indicators */}
        <div className="flex items-center gap-1.5 pt-4">
          {movies.slice(0, 5).map((_, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentIndex(idx)}
              className={`h-1 rounded-full transition-all duration-300 ${
                currentIndex === idx
                  ? "w-8 bg-[#e50914] shadow-[0_0_8px_#e50914]"
                  : "w-3 bg-zinc-600/60 hover:bg-zinc-400"
              }`}
              aria-label={`Go to slide ${idx + 1}`}
            />
          ))}
        </div>
      </div>

      {/* Starring Cast Widget (Bottom Right) */}
      <div className="absolute right-8 bottom-8 hidden md:flex flex-col items-end z-10 bg-black/40 backdrop-blur-md p-3.5 rounded-2xl border border-white/[0.08]">
        <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-2.5">
          Starring
        </div>
        <div className="flex items-center gap-3">
          {castList.map((actor, idx) => (
            <div key={idx} className="flex flex-col items-center gap-1 text-center w-16">
              <div className="w-11 h-11 rounded-full overflow-hidden border border-white/20 bg-zinc-800">
                <img
                  src={`https://images.unsplash.com/photo-${
                    1500000000000 + (idx * 54321) % 100000000
                  }?auto=format&fit=crop&w=120&h=120&q=80`}
                  alt={actor}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    // Fallback avatar
                    (e.target as HTMLElement).style.display = "none";
                  }}
                />
              </div>
              <span className="text-[10px] text-zinc-300 font-medium truncate w-full">
                {actor}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
