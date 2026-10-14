"use client";

import React from "react";
import { Sparkles } from "lucide-react";

interface QuickFiltersProps {
  language: string;
  onLanguageChange: (lang: string) => void;
  genre: string;
  onGenreChange: (genre: string) => void;
  sortBy: string;
  onSortByChange: (sort: string) => void;
  onResetFilters: () => void;
}

const LANGUAGES = [
  { label: "All", value: "All" },
  { label: "English", value: "en" },
  { label: "Hindi", value: "hi" },
  { label: "Marathi", value: "mr" },
];

const GENRES = [
  "All",
  "Action",
  "Comedy",
  "Drama",
  "Sci-Fi",
  "Thriller",
  "Horror",
  "Romance",
];

const SORT_OPTIONS = [
  { label: "Popularity", value: "popularity" },
  { label: "Rating", value: "rating" },
  { label: "Newest", value: "newest" },
];

export function QuickFilters({
  language,
  onLanguageChange,
  genre,
  onGenreChange,
  sortBy,
  onSortByChange,
  onResetFilters,
}: QuickFiltersProps) {
  return (
    <div className="w-full bg-[#0d0f17] border border-white/[0.08] rounded-3xl p-5 shadow-xl select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-3.5">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-[#e50914]" />
          <h3 className="text-sm font-bold text-white tracking-wide">Quick Filters</h3>
        </div>

        <button
          onClick={onResetFilters}
          className="text-[11px] font-semibold text-zinc-400 hover:text-[#e50914] transition-colors"
        >
          Reset All
        </button>
      </div>

      {/* Language Section */}
      <div className="mb-4">
        <div className="text-[11px] font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
          Language
        </div>
        <div className="flex flex-wrap gap-1.5">
          {LANGUAGES.map((item) => {
            const isActive = language === item.value;
            return (
              <button
                key={item.value}
                onClick={() => onLanguageChange(item.value)}
                className={`text-xs px-3 py-1 rounded-full font-medium transition-all ${
                  isActive
                    ? "bg-[#e50914] text-white font-semibold shadow-[0_0_10px_rgba(229,9,20,0.5)]"
                    : "bg-white/[0.05] hover:bg-white/[0.1] text-zinc-300 hover:text-white"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Genre Section */}
      <div className="mb-4">
        <div className="text-[11px] font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
          Genre
        </div>
        <div className="flex flex-wrap gap-1.5">
          {GENRES.map((g) => {
            const isActive = genre === g;
            return (
              <button
                key={g}
                onClick={() => onGenreChange(g)}
                className={`text-xs px-3 py-1 rounded-full font-medium transition-all ${
                  isActive
                    ? "bg-[#e50914] text-white font-semibold shadow-[0_0_10px_rgba(229,9,20,0.5)]"
                    : "bg-white/[0.05] hover:bg-white/[0.1] text-zinc-300 hover:text-white"
                }`}
              >
                {g}
              </button>
            );
          })}
        </div>
      </div>

      {/* Sort By Section */}
      <div>
        <div className="text-[11px] font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
          Sort By
        </div>
        <div className="flex flex-wrap gap-1.5">
          {SORT_OPTIONS.map((item) => {
            const isActive = sortBy === item.value;
            return (
              <button
                key={item.value}
                onClick={() => onSortByChange(item.value)}
                className={`text-xs px-3 py-1 rounded-full font-medium transition-all ${
                  isActive
                    ? "bg-[#e50914] text-white font-semibold shadow-[0_0_10px_rgba(229,9,20,0.5)]"
                    : "bg-white/[0.05] hover:bg-white/[0.1] text-zinc-300 hover:text-white"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
