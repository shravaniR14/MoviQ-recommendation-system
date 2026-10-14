import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MOVIQ | AI Movie Discovery & Hybrid Recommendation",
  description: "Cinematic movie discovery powered by ML hybrid recommendation, vector search, and Gemini AI assistant.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark bg-[#08090d] text-zinc-100 antialiased">
      <body className="min-h-screen bg-[#08090d] text-zinc-100 selection:bg-red-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
