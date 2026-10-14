"use client";

import React from "react";
import { Clock, X, Search } from "lucide-react";
import { RecentSearchItem } from "../lib/types";

interface RecentSearchesProps {
  searches: RecentSearchItem[];
  onSelectSearch: (query: string) => void;
  onRemoveSearch: (query: string) => void;
  onClearAll: () => void;
}

export function RecentSearches({
  searches,
  onSelectSearch,
  onRemoveSearch,
  onClearAll,
}: RecentSearchesProps) {
  if (!searches || searches.length === 0) {
    return null;
  }

  return (
    <div className="w-full bg-[#0d0f17] border border-white/[0.08] rounded-3xl p-5 shadow-xl select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/[0.06] mb-3">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-zinc-400" />
          <h3 className="text-sm font-bold text-white tracking-wide">Recent Searches</h3>
        </div>

        <button
          onClick={onClearAll}
          className="text-[11px] font-semibold text-zinc-400 hover:text-[#e50914] transition-colors"
        >
          Clear All
        </button>
      </div>

      {/* Searches List */}
      <div className="flex flex-col gap-2">
        {searches.map((item) => (
          <div
            key={item.query}
            onClick={() => onSelectSearch(item.query)}
            className="group flex items-center justify-between p-2 rounded-xl hover:bg-white/[0.06] cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-8 h-10 rounded-md overflow-hidden bg-zinc-800 flex-shrink-0 border border-white/10 flex items-center justify-center">
                {item.thumbnail ? (
                  <img
                    src={item.thumbnail}
                    alt={item.query}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <Search className="w-3.5 h-3.5 text-zinc-500" />
                )}
              </div>
              <span className="text-xs font-medium text-zinc-200 truncate group-hover:text-white transition-colors">
                {item.query}
              </span>
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemoveSearch(item.query);
              }}
              className="p-1 text-zinc-500 hover:text-zinc-200 transition-colors"
              aria-label={`Remove ${item.query} from history`}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
