import difflib
import re

from groq import BadRequestError, Groq

try:
    from src.config import GROQ_API_KEY, GROQ_MODEL
    from src.recommender import MovieRecommender
except ImportError:
    from config import GROQ_API_KEY, GROQ_MODEL
    from recommender import MovieRecommender


GENRE_SYNONYMS = {
    "sci fi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "romcom": "Romance",
    "romantic": "Romance",
    "funny": "Comedy",
    "comedy": "Comedy",
    "thriller": "Thriller",
    "suspense": "Thriller",
    "action": "Action",
    "crime": "Crime",
    "mystery": "Mystery",
    "horror": "Horror",
    "fantasy": "Fantasy",
    "animation": "Animation",
    "family": "Children",
    "kids": "Children",
    "drama": "Drama",
    "adventure": "Adventure",
    "war": "War",
    "musical": "Musical",
    "documentary": "Documentary",
}


def _extract_genres(text, known_genres):
    if not text:
        return []
    lowered = text.lower()
    genres = [genre for genre in known_genres if genre.lower() in lowered]
    for key, mapped in GENRE_SYNONYMS.items():
        if key in lowered and mapped in known_genres:
            genres.append(mapped)
    return sorted(set(genres))


def _extract_title_hint(text, known_titles):
    if not text:
        return None
    match = re.search(r"(?:like|love|similar to)\s+([\w\s:'-]{3,})", text, re.I)
    if match:
        candidate = match.group(1).strip()
        closest = difflib.get_close_matches(candidate, known_titles, n=1, cutoff=0.6)
        return closest[0] if closest else candidate

    lowered = text.lower()
    for title in known_titles:
        if title.lower() in lowered:
            return title
    return None


def _build_fallback_response(recommendations, explanations):
    lines = ["Here are some picks you might enjoy:"]
    for title in recommendations:
        reason = explanations.get(title, "popular pick")
        lines.append(f"- {title} ({reason})")
    return "\n".join(lines)


def chatbot_recommendation(user_input, user_id=1, recommender=None, memory=None):
    recommender = recommender or MovieRecommender()
    known_genres = sorted(
        {g for g in recommender.movie_metadata["genres"].dropna().str.split("|").explode()}
    )
    known_titles = recommender.movie_metadata["title"].dropna().tolist()
    genres = _extract_genres(user_input, known_genres)
    seed_title = _extract_title_hint(user_input, known_titles)

    if memory is not None:
        if genres:
            memory["genres"] = genres
        if seed_title:
            memory["seed_title"] = seed_title
        genres = memory.get("genres", genres)
        seed_title = memory.get("seed_title", seed_title)

    recommendations, reasons = recommender.hybrid_recommendations(
        user_id=user_id,
        seed_title=seed_title,
        genres=genres,
        n_movies=6,
    )
    explanations = recommender.explain_recommendations(recommendations, reasons)

    if not GROQ_API_KEY:
        return {
            "response": _build_fallback_response(recommendations, explanations),
            "recommendations": recommendations,
            "explanations": explanations,
        }

    client = Groq(api_key=GROQ_API_KEY)
    movie_list = ", ".join(recommendations)
    prompt = (
        "You are a movie expert assistant. "
        "Explain recommendations in a warm, confident tone. "
        "Keep it concise, avoid spoilers.\n\n"
        f"User says: {user_input}\n"
        f"Recommendations: {movie_list}\n"
        f"Reasons: {explanations}\n"
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return {
            "response": response.choices[0].message.content,
            "recommendations": recommendations,
            "explanations": explanations,
        }
    except BadRequestError:
        return {
            "response": _build_fallback_response(recommendations, explanations),
            "recommendations": recommendations,
            "explanations": explanations,
        }
