"use client";

import React, { useState, useEffect, useRef } from "react";
import { Search, Bell, Sparkles, X } from "lucide-react";
import { searchMovies, getTmdbPosterUrl } from "../lib/api";
import { Movie } from "../lib/types";

interface NavbarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  userId: string;
  onSelectMovie: (movie: Movie) => void;
  onSearchSubmit: (query: string) => void;
  toggleAiAssistant: () => void;
  isAiOpen: boolean;
}

export function Navbar({
  currentTab,
  onTabChange,
  userId,
  onSelectMovie,
  onSearchSubmit,
  toggleAiAssistant,
  isAiOpen,
}: NavbarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Movie[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleSearchChange = (val: string) => {
    setSearchQuery(val);
    if (!val.trim() || val.length < 2) {
      setSearchResults([]);
      setShowDropdown(false);
    }
  };

  // Debounced search
  useEffect(() => {
    if (!searchQuery.trim() || searchQuery.length < 2) {
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const res = await searchMovies(searchQuery, userId, 6);
        setSearchResults(res.results || []);
        setShowDropdown(true);
      } catch (err) {
        console.error("Quick search error:", err);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery, userId]);

  // Click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && searchQuery.trim()) {
      setShowDropdown(false);
      onSearchSubmit(searchQuery.trim());
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full bg-[#08090d]/90 backdrop-blur-md border-b border-white/[0.06] px-4 lg:px-8 py-3">
      <div className="flex items-center justify-between gap-4 max-w-[1920px] mx-auto">
        {/* Left: Brand Logo & Navigation */}
        <div className="flex items-center gap-8">
          <div
            onClick={() => onTabChange("home")}
            className="flex items-center gap-1.5 cursor-pointer select-none group"
          >
            <span className="text-2xl font-black tracking-wider text-white">
              MOVI<span className="text-[#e50914] group-hover:drop-shadow-[0_0_10px_rgba(229,9,20,0.8)] transition-all">Q</span>
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-6">
            <button
              onClick={() => onTabChange("home")}
              className={`text-sm font-medium transition-all relative py-1 ${
                currentTab === "home"
                  ? "text-white font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-[#e50914] after:shadow-[0_0_8px_#e50914]"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Home
            </button>
            <button
              onClick={() => onTabChange("movies")}
              className={`text-sm font-medium transition-all relative py-1 ${
                currentTab === "movies"
                  ? "text-white font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-[#e50914] after:shadow-[0_0_8px_#e50914]"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Movies
            </button>
            <button
              onClick={() => {
                onTabChange("assistant");
                if (!isAiOpen) toggleAiAssistant();
              }}
              className={`text-sm font-medium transition-all relative py-1 flex items-center gap-1.5 ${
                currentTab === "assistant" || isAiOpen
                  ? "text-[#ff3b45] font-semibold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-[#e50914]" />
              AI Assistant
            </button>
          </nav>
        </div>

        {/* Center: Search Bar */}
        <div ref={dropdownRef} className="relative flex-1 max-w-xl mx-2">
          <div className="relative flex items-center">
            <Search className="absolute left-4 w-4 h-4 text-zinc-400 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => searchQuery.length >= 2 && setShowDropdown(true)}
              placeholder="Search movies, genres, actors..."
              className="w-full bg-[#12141c]/90 text-sm text-zinc-100 placeholder-zinc-500 pl-11 pr-10 py-2.5 rounded-full border border-white/[0.08] focus:border-[#e50914] focus:outline-none focus:ring-1 focus:ring-[#e50914]/50 transition-all shadow-inner"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setShowDropdown(false);
                }}
                className="absolute right-3.5 text-zinc-400 hover:text-zinc-200 p-0.5"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Quick Search Dropdown */}
          {showDropdown && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-[#12141d] border border-white/[0.1] rounded-2xl shadow-[0_10px_35px_rgba(0,0,0,0.8)] overflow-hidden z-50">
              {isSearching ? (
                <div className="p-4 text-xs text-zinc-400 flex items-center gap-2">
                  <div className="w-3 h-3 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
                  Semantic ML search in progress...
                </div>
              ) : searchResults.length > 0 ? (
                <div className="py-2">
                  <div className="px-4 py-1.5 text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center justify-between">
                    <span>ML Semantic Results</span>
                    <span className="text-[#e50914]">Press Enter for all</span>
                  </div>
                  {searchResults.map((movie) => (
                    <div
                      key={movie.tmdb_id}
                      onClick={() => {
                        onSelectMovie(movie);
                        setShowDropdown(false);
                      }}
                      className="px-4 py-2 hover:bg-white/[0.06] flex items-center gap-3 cursor-pointer transition-colors"
                    >
                      <img
                        src={getTmdbPosterUrl(movie.poster_path, "w342")}
                        alt={movie.title}
                        className="w-9 h-12 object-cover rounded-md flex-shrink-0 bg-zinc-800"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-white truncate">
                          {movie.title}
                        </div>
                        <div className="text-xs text-zinc-400 flex items-center gap-2">
                          <span>{movie.release_year || "Unknown"}</span>
                          {movie.genres && movie.genres.length > 0 && (
                            <>
                              <span>•</span>
                              <span className="truncate">{movie.genres.slice(0, 2).join(", ")}</span>
                            </>
                          )}
                          {movie.hybrid_score && (
                            <span className="text-[#ff3b45] font-semibold ml-auto">
                              {Math.round(movie.hybrid_score * 100)}% Match
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-xs text-zinc-400">
                  No movies found matching &ldquo;{searchQuery}&rdquo;.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Notifications & User Avatar */}
        <div className="flex items-center gap-3">
          <button
            onClick={toggleAiAssistant}
            className={`p-2.5 rounded-full transition-all ${
              isAiOpen
                ? "bg-[#e50914] text-white shadow-[0_0_12px_rgba(229,9,20,0.5)]"
                : "text-zinc-400 hover:text-white hover:bg-white/[0.06]"
            }`}
            title="Toggle MOVIQ AI Assistant"
          >
            <Sparkles className="w-4 h-4" />
          </button>

          <button
            className="p-2.5 rounded-full text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-all relative"
            title="Notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-2 right-2 w-2 h-2 bg-[#e50914] rounded-full ring-2 ring-[#08090d]" />
          </button>

          <div
            className="w-8 h-8 rounded-full bg-gradient-to-tr from-zinc-700 to-zinc-500 border border-white/20 flex items-center justify-center text-xs font-bold text-white shadow-sm cursor-pointer select-none hover:ring-2 hover:ring-[#e50914]/60 transition-all"
            title={`Active User: ${userId}`}
          >
            {userId.slice(-1).toUpperCase() || "S"}
          </div>
        </div>
      </div>
    </header>
  );
}
