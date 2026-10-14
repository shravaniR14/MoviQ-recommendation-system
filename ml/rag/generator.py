import os
import logging
from google.genai.errors import APIError

from dotenv import load_dotenv
from google import genai

from ml.rag.context_builder import MovieContextBuilder


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )


class MoviqAssistant:

    def __init__(self, hybrid_recommender=None):
        self.client = genai.Client(api_key=API_KEY)
        self.context_builder = MovieContextBuilder(hybrid_recommender=hybrid_recommender)

    def ask(self, question, language="hi", user_id=None):
        movies = self.context_builder.retrieve(
            query=question,
            top_k=5,
            language=language,
            user_id=user_id,
        )

        if not movies:
            return "I couldn't find enough relevant movie information."

        context = "\n\n".join(
            f"Title: {movie['title']}\n"
            f"Year: {movie['year']}\n"
            f"Genres: {movie['genres']}\n"
            f"Rating: {movie['rating']}\n"
            f"Overview: {movie['overview']}"
            for movie in movies
        )

        prompt = f"""
You are Moviq, an AI movie discovery assistant.

Answer using only the movie information in the context.
Do not invent movie facts or claim unsupported details.
If the context is insufficient, say so.
Recommend relevant movies and briefly explain why.

Movie context:
{context}

User question:
{question}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return response.text or "I couldn't generate an answer."
        except Exception as exc:
            # Log detailed error without exposing credentials
            logger = logging.getLogger("moviq.rag")
            logger.error(f"Error during Gemini generation: {exc!r}")
            logger.exception("Full traceback for Gemini generation failure")
            # Fallback: use retrieved movie titles as a simple answer
            titles = [m.get("title") for m in movies if m.get("title")]
            if titles:
                return (
                    "I couldn't reach the AI service, but here are some relevant movies: "
                    + ", ".join(titles)
                    + "."
                )
            return "I couldn't reach the AI service and no relevant movies were found."
        


def main():
    assistant = MoviqAssistant()

    question = "Suggest Hindi movies about friendship and college life."

    print("\nMOVIQ AI ASSISTANT")
    print("=" * 60)
    print("Question:", question)
    print("\nAnswer:\n")
    print(assistant.ask(question, language="hi"))


if __name__ == "__main__":
    main()