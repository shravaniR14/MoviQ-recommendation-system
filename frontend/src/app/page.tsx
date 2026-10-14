"use client";

import React, { useState, useEffect } from "react";
import { Navbar } from "../components/Navbar";
import { Sidebar } from "../components/Sidebar";
import { Hero } from "../components/Hero";
import { AiRecommendations } from "../components/AiRecommendations";
import { BecauseYouWatched } from "../components/BecauseYouWatched";
import { PopularMovies } from "../components/PopularMovies";
import { AiAssistantDrawer } from "../components/AiAssistantDrawer";
import { RecentSearches } from "../components/RecentSearches";
import { QuickFilters } from "../components/QuickFilters";
import { MovieDetailsModal } from "../components/MovieDetailsModal";
import {
  Movie,
  RecentSearchItem,
} from "../lib/types";
import {
  fetchPersonalizedRecommendations,
  fetchSimilarMovies,
  fetchPopularMovies,
  recordInteraction,
  searchMovies,
} from "../lib/api";

export default function MoviqHomePage() {
  // State
  const [userId] = useState("user_42");
  const [currentTab, setCurrentTab] = useState("home");
  const [isAiOpen, setIsAiOpen] = useState(true);

  // Movie Catalogs
  const [featuredMovies, setFeaturedMovies] = useState<Movie[]>([]);
  const [personalizedMovies, setPersonalizedMovies] = useState<Movie[]>([]);
  const [userMaturity, setUserMaturity] = useState<"cold" | "nascent" | "warm">("nascent");
  const [lastWatchedMovie, setLastWatchedMovie] = useState<{
    tmdb_id: number;
    title: string;
  }>({
    tmdb_id: 27205,
    title: "Inception",
  });
  const [similarMovies, setSimilarMovies] = useState<Movie[]>([]);
  const [popularMovies, setPopularMovies] = useState<Movie[]>([]);

  // Filters
  const [activeGenre, setActiveGenre] = useState("All");
  const [selectedLanguage, setSelectedLanguage] = useState("All");
  const [sortBy, setSortBy] = useState("popularity");

  // User State & History
  const [myList, setMyList] = useState<Movie[]>([]);
  const [watchHistory, setWatchHistory] = useState<Movie[]>([]);
  const [userRatings, setUserRatings] = useState<Record<number, number>>({});
  const [recentSearches, setRecentSearches] = useState<RecentSearchItem[]>([
    {
      query: "Inception",
      timestamp: 1728712000000,
      thumbnail: "https://image.tmdb.org/t/p/w185/o2280Pz6uVvS5sWw7g8U3d0c3hX.jpg",
    },
    {
      query: "Interstellar",
      timestamp: 1728711000000,
      thumbnail: "https://image.tmdb.org/t/p/w185/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    },
    {
      query: "Sci-fi movies",
      timestamp: 1728710000000,
    },
    {
      query: "Marvel movies",
      timestamp: 1728709000000,
    },
    {
      query: "Dark Knight",
      timestamp: 1728708000000,
    },
  ]);

  // Modal
  const [selectedModalMovie, setSelectedModalMovie] = useState<Movie | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchViewActive, setSearchViewActive] = useState(false);
  const [searchResults, setSearchResults] = useState<Movie[]>([]);
  const [activeSearchQuery, setActiveSearchQuery] = useState("");

  // Initial Data Fetching on mount
  useEffect(() => {
    let ignore = false;
    async function loadInitial() {
      try {
        const popularData = await fetchPopularMovies(undefined, undefined, 20);
        if (!ignore && popularData?.movies?.length) {
          setFeaturedMovies(popularData.movies.slice(0, 5));
          setPopularMovies(popularData.movies.slice(4, 20));
        }

        const personalData = await fetchPersonalizedRecommendations(userId, 10);
        if (!ignore && personalData?.recommendations?.length) {
          setPersonalizedMovies(personalData.recommendations);
          setUserMaturity(personalData.maturity);
        }

        const similarData = await fetchSimilarMovies(lastWatchedMovie.tmdb_id, 8);
        if (!ignore && similarData?.similar_movies?.length) {
          setSimilarMovies(similarData.similar_movies);
        }
      } catch (err) {
        console.error("Error loading Moviq data:", err);
      }
    }

    loadInitial();
    return () => {
      ignore = true;
    };
  }, [userId, lastWatchedMovie.tmdb_id]);

  // Handle Genre or Filter change on Popular Movies
  useEffect(() => {
    let ignore = false;
    async function filterPopular() {
      try {
        const data = await fetchPopularMovies(
          activeGenre !== "All" ? activeGenre : undefined,
          selectedLanguage !== "All" ? selectedLanguage : undefined,
          16
        );
        if (!ignore && data?.movies) {
          setPopularMovies(data.movies);
        }
      } catch (err) {
        console.error("Error filtering popular movies:", err);
      }
    }

    filterPopular();
    return () => {
      ignore = true;
    };
  }, [activeGenre, selectedLanguage]);

  // Interaction Handlers
  const handleWatchNow = async (movie: Movie) => {
    // 1. Add to local history
    setWatchHistory((prev) => {
      const filtered = prev.filter((m) => m.tmdb_id !== movie.tmdb_id);
      return [movie, ...filtered];
    });

    // 2. Update last watched movie
    setLastWatchedMovie({
      tmdb_id: movie.tmdb_id,
      title: movie.title,
    });

    // 3. Record watch interaction to ML backend
    try {
      await recordInteraction(userId, "watch", movie.tmdb_id);

      // 4. Update similar movies carousel dynamically
      const sims = await fetchSimilarMovies(movie.tmdb_id, 8);
      if (sims && sims.similar_movies) {
        setSimilarMovies(sims.similar_movies);
      }

      // 5. Refresh personalized recommendations
      const recs = await fetchPersonalizedRecommendations(userId, 10);
      if (recs && recs.recommendations) {
        setPersonalizedMovies(recs.recommendations);
        setUserMaturity(recs.maturity);
      }
    } catch (err) {
      console.error("Error recording watch interaction:", err);
    }

    // Open detail modal to preview
    setSelectedModalMovie(movie);
    setIsModalOpen(true);
  };

  const handleRateMovie = async (movie: Movie, rating: number) => {
    setUserRatings((prev) => ({ ...prev, [movie.tmdb_id]: rating }));

    try {
      const res = await recordInteraction(userId, "rating", movie.tmdb_id, rating);
      if (res && (res.maturity === "cold" || res.maturity === "nascent" || res.maturity === "warm")) {
        setUserMaturity(res.maturity);
      }

      // Refresh personalized recommendations with updated taste profile
      const recs = await fetchPersonalizedRecommendations(userId, 10);
      if (recs && recs.recommendations) {
        setPersonalizedMovies(recs.recommendations);
      }
    } catch (err) {
      console.error("Error recording rating interaction:", err);
    }
  };

  const handleToggleMyList = (movie: Movie) => {
    setMyList((prev) => {
      const exists = prev.some((m) => m.tmdb_id === movie.tmdb_id);
      if (exists) {
        return prev.filter((m) => m.tmdb_id !== movie.tmdb_id);
      } else {
        return [...prev, movie];
      }
    });
  };

  const isMovieInList = (tmdbId: number) => {
    return myList.some((m) => m.tmdb_id === tmdbId);
  };

  const handleSearchSubmit = async (query: string) => {
    if (!query.trim()) return;

    setActiveSearchQuery(query);
    setSearchViewActive(true);

    // Add to recent searches
    setRecentSearches((prev) => {
      const filtered = prev.filter((s) => s.query.toLowerCase() !== query.toLowerCase());
      return [{ query, timestamp: Date.now() }, ...filtered.slice(0, 5)];
    });

    try {
      await recordInteraction(userId, "search", undefined, undefined, query);
      const res = await searchMovies(query, userId, 16);
      if (res && res.results) {
        setSearchResults(res.results);
      }
    } catch (err) {
      console.error("Search execution failed:", err);
    }
  };

  const handleSelectMovie = (movie: Movie) => {
    setSelectedModalMovie(movie);
    setIsModalOpen(true);
  };

  const handleResetFilters = () => {
    setSelectedLanguage("All");
    setActiveGenre("All");
    setSortBy("popularity");
  };

  return (
    <div className="min-h-screen bg-[#08090d] text-zinc-100 flex flex-col font-sans selection:bg-[#e50914] selection:text-white">
      {/* Top Navigation Bar */}
      <Navbar
        currentTab={currentTab}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          setSearchViewActive(false);
        }}
        userId={userId}
        onSelectMovie={handleSelectMovie}
        onSearchSubmit={handleSearchSubmit}
        toggleAiAssistant={() => setIsAiOpen((prev) => !prev)}
        isAiOpen={isAiOpen}
      />

      {/* Main Container */}
      <div className="flex-1 max-w-[1920px] w-full mx-auto px-4 lg:px-8 py-5 flex gap-6">
        {/* Left Sidebar */}
        <Sidebar
          currentTab={currentTab}
          onTabChange={(tab) => {
            setCurrentTab(tab);
            setSearchViewActive(false);
          }}
          selectedGenre={activeGenre}
          onGenreSelect={(genre) => {
            setActiveGenre(genre);
            setSearchViewActive(false);
          }}
          myListCount={myList.length}
        />

        {/* Center Main Stage */}
        <main className="flex-1 min-w-0 flex flex-col gap-6">
          {/* Top category pill above hero */}
          <div className="flex items-center justify-between">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/[0.06] border border-white/[0.08] text-xs font-semibold text-zinc-200">
              <span>All Movies</span>
            </div>
            {searchViewActive && (
              <button
                onClick={() => setSearchViewActive(false)}
                className="text-xs text-[#e50914] hover:underline font-medium"
              >
                ✕ Clear Search View
              </button>
            )}
          </div>

          {/* Conditional Views */}
          {searchViewActive ? (
            /* Search Results Stage */
            <section className="flex flex-col gap-4 py-2">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-white">
                  Search Results for &ldquo;{activeSearchQuery}&rdquo;
                </h2>
                <span className="text-xs text-zinc-400">
                  {searchResults.length} ML candidates ranked
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {searchResults.map((movie) => (
                  <div
                    key={movie.tmdb_id}
                    onClick={() => handleSelectMovie(movie)}
                    className="group relative flex flex-col rounded-2xl overflow-hidden bg-[#12141e] border border-white/[0.06] hover:border-[#e50914] transition-all duration-300 hover:shadow-[0_0_20px_rgba(229,9,20,0.4)] cursor-pointer"
                  >
                    <div className="aspect-[2/3] w-full bg-zinc-900">
                      <img
                        src={movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"}
                        alt={movie.title}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                      />
                    </div>
                    <div className="p-3">
                      <h3 className="text-sm font-semibold text-white truncate">
                        {movie.title}
                      </h3>
                      <div className="text-xs text-zinc-400 mt-1 flex justify-between">
                        <span>{movie.release_year || "Unknown"}</span>
                        {movie.hybrid_score && (
                          <span className="text-[#e50914] font-bold">
                            {Math.round(movie.hybrid_score * 100)}% Match
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ) : currentTab === "mylist" ? (
            /* My List View */
            <section className="flex flex-col gap-4 py-2">
              <h2 className="text-2xl font-bold text-white">My Watchlist ({myList.length})</h2>
              {myList.length === 0 ? (
                <div className="p-12 text-center text-zinc-500 bg-[#12141e] rounded-3xl border border-white/[0.06]">
                  Your watchlist is empty. Click &ldquo;+ Add to List&rdquo; on any movie to save it here!
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {myList.map((movie) => (
                    <div
                      key={movie.tmdb_id}
                      onClick={() => handleSelectMovie(movie)}
                      className="group relative flex flex-col rounded-2xl overflow-hidden bg-[#12141e] border border-white/[0.06] hover:border-[#e50914] transition-all cursor-pointer"
                    >
                      <div className="aspect-[2/3] w-full bg-zinc-900">
                        <img
                          src={movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"}
                          alt={movie.title}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-3">
                        <h3 className="text-sm font-semibold text-white truncate">{movie.title}</h3>
                        <span className="text-xs text-zinc-400">{movie.release_year}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          ) : currentTab === "history" ? (
            /* History View */
            <section className="flex flex-col gap-4 py-2">
              <h2 className="text-2xl font-bold text-white">
                Watch History ({watchHistory.length})
              </h2>
              {watchHistory.length === 0 ? (
                <div className="p-12 text-center text-zinc-500 bg-[#12141e] rounded-3xl border border-white/[0.06]">
                  No watched movies recorded yet. Click &ldquo;Watch Now&rdquo; on any movie to build your ML taste profile!
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {watchHistory.map((movie) => (
                    <div
                      key={movie.tmdb_id}
                      onClick={() => handleSelectMovie(movie)}
                      className="group relative flex flex-col rounded-2xl overflow-hidden bg-[#12141e] border border-white/[0.06] hover:border-[#e50914] transition-all cursor-pointer"
                    >
                      <div className="aspect-[2/3] w-full bg-zinc-900">
                        <img
                          src={movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"}
                          alt={movie.title}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-3">
                        <h3 className="text-sm font-semibold text-white truncate">{movie.title}</h3>
                        <span className="text-xs text-zinc-400">{movie.release_year}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          ) : (
            /* Standard Cinematic Home Feed */
            <>
              {/* Hero Section */}
              <Hero
                movies={featuredMovies}
                onWatchNow={handleWatchNow}
                onToggleMyList={handleToggleMyList}
                isMovieInList={isMovieInList}
                onSelectMovie={handleSelectMovie}
              />

              {/* AI Recommendations For You */}
              <AiRecommendations
                movies={personalizedMovies}
                maturity={userMaturity}
                onSelectMovie={handleSelectMovie}
                onWatchNow={handleWatchNow}
              />

              {/* Because You Watched Section */}
              <BecauseYouWatched
                targetTitle={lastWatchedMovie.title}
                movies={similarMovies}
                onSelectMovie={handleSelectMovie}
                onWatchNow={handleWatchNow}
              />

              {/* Popular Movies Discovery Grid */}
              <PopularMovies
                movies={popularMovies}
                activeGenre={activeGenre}
                onGenreChange={setActiveGenre}
                onSelectMovie={handleSelectMovie}
                onToggleMyList={handleToggleMyList}
                isMovieInList={isMovieInList}
              />
            </>
          )}
        </main>

        {/* Right Sidebar: AI Assistant Drawer, Recent Searches, and Quick Filters */}
        <aside className="w-80 lg:w-88 flex-shrink-0 hidden 2xl:flex flex-col gap-6 py-2 select-none">
          {/* AI Assistant Drawer */}
          {isAiOpen && (
            <AiAssistantDrawer
              userId={userId}
              onClose={() => setIsAiOpen(false)}
            />
          )}

          {/* Recent Searches Panel */}
          <RecentSearches
            searches={recentSearches}
            onSelectSearch={(q) => handleSearchSubmit(q)}
            onRemoveSearch={(q) =>
              setRecentSearches((prev) => prev.filter((s) => s.query !== q))
            }
            onClearAll={() => setRecentSearches([])}
          />

          {/* Quick Filters Panel */}
          <QuickFilters
            language={selectedLanguage}
            onLanguageChange={setSelectedLanguage}
            genre={activeGenre}
            onGenreChange={setActiveGenre}
            sortBy={sortBy}
            onSortByChange={setSortBy}
            onResetFilters={handleResetFilters}
          />
        </aside>
      </div>

      {/* Movie Details Modal */}
      <MovieDetailsModal
        movie={selectedModalMovie}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onWatchNow={handleWatchNow}
        onRateMovie={handleRateMovie}
        onToggleMyList={handleToggleMyList}
        isMovieInList={isMovieInList}
        userRating={selectedModalMovie ? userRatings[selectedModalMovie.tmdb_id] : undefined}
      />
    </div>
  );
}
