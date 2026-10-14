export interface Movie {
  tmdb_id: number;
  title: string;
  original_title?: string;
  release_year?: number | null;
  language?: string;
  genres?: string[];
  vote_average?: number;
  vote_count?: number;
  popularity?: number;
  poster_path?: string | null;
  backdrop_path?: string | null;
  overview?: string;
  tagline?: string;
  runtime?: number | null;
  cast?: string[];
  director?: string | null;
  hybrid_score?: number;
  cf_score?: number;
  content_score?: number;
  match_source?: string;
  hybrid_similarity?: number;
  content_similarity?: number;
  cf_similarity?: number;
  semantic_score?: number;
}

export interface PersonalizedResponse {
  user_id: string;
  maturity: "cold" | "nascent" | "warm";
  count: number;
  recommendations: Movie[];
}

export interface SimilarMoviesResponse {
  tmdb_id: number;
  target_title: string;
  count: number;
  similar_movies: Movie[];
}

export interface HybridSearchResponse {
  query: string;
  user_id?: string;
  count: number;
  results: Movie[];
}

export interface PopularMoviesResponse {
  count: number;
  movies: Movie[];
}

export interface InteractionResponse {
  status: string;
  user_id: string;
  interaction_recorded: string;
  maturity: string;
  total_ratings: number;
  total_watched: number;
  top_genres: string[];
  preferred_languages: string[];
}

export interface AskResponse {
  question: string;
  language: string;
  user_id: string;
  answer: string;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "assistant";
  text: string;
  timestamp: string;
}

export interface RecentSearchItem {
  query: string;
  timestamp: number;
  thumbnail?: string;
}
