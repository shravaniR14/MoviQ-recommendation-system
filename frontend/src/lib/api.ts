import {
  Movie,
  PersonalizedResponse,
  SimilarMoviesResponse,
  HybridSearchResponse,
  PopularMoviesResponse,
  InteractionResponse,
  AskResponse,
} from "./types";

const API_BASE =
  typeof window !== "undefined"
    ? "http://127.0.0.1:8000"
    : process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function getTmdbPosterUrl(
  path: string | null | undefined,
  size: "w342" | "w500" | "original" = "w500"
): string {
  if (!path) {
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80";
  }
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/${size}${path.startsWith("/") ? "" : "/"}${path}`;
}

export function getTmdbBackdropUrl(path: string | null | undefined): string {
  if (!path) {
    return "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=1920&q=80";
  }
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/original${path.startsWith("/") ? "" : "/"}${path}`;
}

export async function fetchPersonalizedRecommendations(
  userId: string = "user_42",
  k: number = 10,
  language?: string,
  minRating?: number
): Promise<PersonalizedResponse> {
  const res = await fetch(`${API_BASE}/recommend/personalized`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      k,
      language: language || undefined,
      min_rating: minRating || undefined,
    }),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch personalized recommendations: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSimilarMovies(
  tmdbId: number,
  k: number = 10,
  language?: string
): Promise<SimilarMoviesResponse> {
  const url = new URL(`${API_BASE}/recommend/similar/${tmdbId}`);
  url.searchParams.set("k", k.toString());
  if (language) url.searchParams.set("language", language);

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch similar movies: ${res.statusText}`);
  }
  return res.json();
}

export async function searchMovies(
  query: string,
  userId?: string,
  k: number = 10,
  filters?: {
    language?: string;
    min_year?: number;
    max_year?: number;
    min_rating?: number;
    genre?: string;
  }
): Promise<HybridSearchResponse> {
  const body: Record<string, unknown> = {
    query,
    user_id: userId || undefined,
    k,
    ...filters,
  };

  const res = await fetch(`${API_BASE}/recommend/hybrid-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to execute search: ${res.statusText}`);
  }
  return res.json();
}

export async function recordInteraction(
  userId: string,
  interactionType: "rating" | "watch" | "search",
  tmdbId?: number,
  rating?: number,
  query?: string
): Promise<InteractionResponse> {
  const body: Record<string, unknown> = {
    user_id: userId,
    interaction_type: interactionType,
  };
  if (tmdbId !== undefined) body.tmdb_id = tmdbId;
  if (rating !== undefined) body.rating = rating;
  if (query !== undefined) body.query = query;

  const res = await fetch(`${API_BASE}/user/interaction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error(`Failed to record interaction: ${res.statusText}`);
  }
  return res.json();
}

export async function askAssistant(
  question: string,
  language: string = "en",
  userId: string = "user_42"
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      language,
      user_id: userId,
    }),
    cache: "no-store",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `AI Assistant request failed with status ${res.status}`);
  }
  return res.json();
}

export async function fetchPopularMovies(
  genre?: string,
  language?: string,
  k: number = 16
): Promise<PopularMoviesResponse> {
  const url = new URL(`${API_BASE}/movies/popular`);
  url.searchParams.set("k", k.toString());
  if (genre && genre !== "All") url.searchParams.set("genre", genre);
  if (language && language !== "All") url.searchParams.set("language", language);

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch popular movies: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchMovieDetails(tmdbId: number): Promise<Movie> {
  const res = await fetch(`${API_BASE}/movies/${tmdbId}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch movie details: ${res.statusText}`);
  }
  return res.json();
}
