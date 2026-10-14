"use client";

import React from "react";
import {
  Home,
  Film,
  Sparkles,
  Bookmark,
  Clock,
  Flame,
  Smile,
  Theater,
  Rocket,
  ShieldAlert,
  Ghost,
  Heart,
} from "lucide-react";

interface SidebarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  selectedGenre: string;
  onGenreSelect: (genre: string) => void;
  myListCount: number;
}

const GENRES = [
  { name: "Action", icon: Flame },
  { name: "Comedy", icon: Smile },
  { name: "Drama", icon: Theater },
  { name: "Sci-Fi", icon: Rocket },
  { name: "Thriller", icon: ShieldAlert },
  { name: "Horror", icon: Ghost },
  { name: "Romance", icon: Heart },
];

export function Sidebar({
  currentTab,
  onTabChange,
  selectedGenre,
  onGenreSelect,
  myListCount,
}: SidebarProps) {
  return (
    <aside className="w-60 flex-shrink-0 hidden xl:flex flex-col gap-6 py-4 pr-3 select-none">
      {/* Primary Navigation */}
      <div className="flex flex-col gap-1.5">
        <button
          onClick={() => onTabChange("home")}
          className={`flex items-center gap-3.5 px-4 py-3 rounded-2xl text-sm font-medium transition-all ${
            currentTab === "home"
              ? "bg-[#e50914] text-white font-semibold shadow-[0_0_18px_rgba(229,9,20,0.45)]"
              : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
          }`}
        >
          <Home className="w-4 h-4" />
          <span>Home</span>
        </button>

        <button
          onClick={() => onTabChange("movies")}
          className={`flex items-center gap-3.5 px-4 py-3 rounded-2xl text-sm font-medium transition-all ${
            currentTab === "movies"
              ? "bg-[#e50914] text-white font-semibold shadow-[0_0_18px_rgba(229,9,20,0.45)]"
              : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
          }`}
        >
          <Film className="w-4 h-4" />
          <span>Movies</span>
        </button>

        <button
          onClick={() => onTabChange("assistant")}
          className={`flex items-center gap-3.5 px-4 py-3 rounded-2xl text-sm font-medium transition-all ${
            currentTab === "assistant"
              ? "bg-[#e50914] text-white font-semibold shadow-[0_0_18px_rgba(229,9,20,0.45)]"
              : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
          }`}
        >
          <Sparkles className="w-4 h-4 text-[#ff434d]" />
          <span>AI Assistant</span>
        </button>

        <button
          onClick={() => onTabChange("mylist")}
          className={`flex items-center justify-between px-4 py-3 rounded-2xl text-sm font-medium transition-all ${
            currentTab === "mylist"
              ? "bg-[#e50914] text-white font-semibold shadow-[0_0_18px_rgba(229,9,20,0.45)]"
              : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
          }`}
        >
          <div className="flex items-center gap-3.5">
            <Bookmark className="w-4 h-4" />
            <span>My List</span>
          </div>
          {myListCount > 0 && (
            <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-white/20 text-white">
              {myListCount}
            </span>
          )}
        </button>

        <button
          onClick={() => onTabChange("history")}
          className={`flex items-center gap-3.5 px-4 py-3 rounded-2xl text-sm font-medium transition-all ${
            currentTab === "history"
              ? "bg-[#e50914] text-white font-semibold shadow-[0_0_18px_rgba(229,9,20,0.45)]"
              : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
          }`}
        >
          <Clock className="w-4 h-4" />
          <span>History</span>
        </button>
      </div>

      {/* Genres Section */}
      <div className="pt-2 border-t border-white/[0.06]">
        <div className="px-4 pb-3 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
          Genres
        </div>
        <div className="flex flex-col gap-1">
          {GENRES.map((g) => {
            const Icon = g.icon;
            const isSelected = selectedGenre === g.name;
            return (
              <button
                key={g.name}
                onClick={() => onGenreSelect(isSelected ? "All" : g.name)}
                className={`flex items-center gap-3.5 px-4 py-2.5 rounded-xl text-sm transition-all text-left ${
                  isSelected
                    ? "text-white bg-white/[0.08] font-semibold border-l-2 border-[#e50914]"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.03]"
                }`}
              >
                <Icon className={`w-4 h-4 ${isSelected ? "text-[#e50914]" : "text-zinc-500"}`} />
                <span>{g.name}</span>
              </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
