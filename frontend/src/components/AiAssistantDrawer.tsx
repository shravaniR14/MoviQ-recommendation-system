"use client";

import React, { useState, useRef, useEffect } from "react";
import { Sparkles, RotateCcw, X, Send } from "lucide-react";
import { askAssistant } from "../lib/api";
import { ChatMessage } from "../lib/types";

interface AiAssistantDrawerProps {
  userId: string;
  onClose: () => void;
}

const INITIAL_SUGGESTIONS = [
  "Show me mind-bending movies",
  "Good Marathi movies",
  "I want a light comedy",
  "Recommend something like Inception",
];

let msgCounter = 0;
function createMessageId(prefix: string) {
  msgCounter += 1;
  return `${prefix}_${msgCounter}_${Math.random().toString(36).slice(2, 6)}`;
}

function getCurrentTime() {
  const now = new Date();
  return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
}

export function AiAssistantDrawer({
  userId,
  onClose,
}: AiAssistantDrawerProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = {
      id: createMessageId("user"),
      sender: "user",
      text,
      timestamp: getCurrentTime(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputValue("");
    setIsLoading(true);

    try {
      const res = await askAssistant(text, "en", userId);
      const botMsg: ChatMessage = {
        id: createMessageId("bot"),
        sender: "assistant",
        text: res.answer,
        timestamp: getCurrentTime(),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err: unknown) {
      console.error("Assistant error:", err);
      const errorMessage =
        err instanceof Error
          ? err.message
          : "I'm temporarily having trouble connecting to the AI model. Let me recommend some top titles from our ML database directly!";
      const errMsg: ChatMessage = {
        id: createMessageId("err"),
        sender: "assistant",
        text: errorMessage,
        timestamp: getCurrentTime(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
  };

  return (
    <div className="w-full bg-[#0d0f17] border border-white/[0.08] rounded-3xl p-5 flex flex-col shadow-xl select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3.5 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-[#e50914] flex items-center justify-center shadow-[0_0_10px_rgba(229,9,20,0.6)]">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="text-base font-bold text-white tracking-wide">MOVIQ AI</span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleReset}
            className="p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-white/[0.05] transition-colors"
            title="Reset conversation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-white/[0.05] transition-colors"
            title="Close Assistant"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto max-h-[380px] my-3 pr-1 space-y-3.5 text-xs no-scrollbar">
        {/* Initial Greeting */}
        <div className="bg-[#151824] border border-white/[0.06] rounded-2xl p-4 text-zinc-200 leading-relaxed shadow-sm">
          <div className="text-sm font-semibold text-white mb-1.5 flex items-center gap-1.5">
            <span>👋 Hi! I&apos;m Moviq AI</span>
          </div>
          <p className="text-zinc-300 mb-2">
            Your personal movie assistant. Tell me what kind of movies you like and I&apos;ll
            recommend the best ones for you!
          </p>

          {/* Quick Suggestion Chips */}
          <div className="flex flex-col gap-1.5 pt-2">
            {INITIAL_SUGGESTIONS.map((suggestion, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(suggestion)}
                className="text-left px-3 py-2 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] hover:border-[#e50914]/40 border border-white/[0.04] text-zinc-300 hover:text-white transition-all text-[11px] font-medium"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>

        {/* Message History */}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${
              msg.sender === "user" ? "items-end" : "items-start"
            }`}
          >
            <div
              className={`max-w-[90%] rounded-2xl px-3.5 py-2.5 leading-relaxed text-xs ${
                msg.sender === "user"
                  ? "bg-[#e50914] text-white shadow-[0_2px_12px_rgba(229,9,20,0.3)] rounded-br-none"
                  : "bg-[#151824] border border-white/[0.06] text-zinc-200 rounded-bl-none"
              }`}
            >
              {msg.text}
            </div>
            <span className="text-[10px] text-zinc-500 mt-1 px-1">{msg.timestamp}</span>
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 p-3 bg-[#151824] rounded-2xl border border-white/[0.06] w-fit">
            <div className="w-2 h-2 rounded-full bg-[#e50914] animate-bounce" />
            <div className="w-2 h-2 rounded-full bg-[#e50914] animate-bounce [animation-delay:0.2s]" />
            <div className="w-2 h-2 rounded-full bg-[#e50914] animate-bounce [animation-delay:0.4s]" />
            <span className="text-[11px] text-zinc-400 ml-1">Analyzing taste & ML models...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Field */}
      <div className="relative flex items-center pt-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask anything about movies..."
          disabled={isLoading}
          className="w-full bg-[#151824] text-xs text-white placeholder-zinc-500 pl-4 pr-11 py-3 rounded-full border border-white/[0.08] focus:border-[#e50914] focus:outline-none focus:ring-1 focus:ring-[#e50914]/50 transition-all shadow-inner"
        />
        <button
          onClick={() => handleSend()}
          disabled={!inputValue.trim() || isLoading}
          className="absolute right-1.5 w-8 h-8 rounded-full bg-[#e50914] hover:bg-[#ff1e27] disabled:opacity-40 disabled:hover:bg-[#e50914] text-white flex items-center justify-center transition-all shadow-[0_0_10px_rgba(229,9,20,0.5)]"
          aria-label="Send message"
        >
          <Send className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
