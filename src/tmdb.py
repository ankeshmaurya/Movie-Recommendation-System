import certifi
import requests
from requests.exceptions import RequestException

from .config import TMDB_API_KEY

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _is_enabled():
    return bool(TMDB_API_KEY)


def search_movie(title, year=None):
    if not _is_enabled():
        return None
    try:
        # First try with year if provided
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = year
        response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=params,
            timeout=10,
            verify=certifi.where(),
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]
        
        # Retry without year if first search failed or found nothing
        if year:
            params = {"api_key": TMDB_API_KEY, "query": title}
            response = requests.get(
                f"{TMDB_BASE_URL}/search/movie",
                params=params,
                timeout=10,
                verify=certifi.where(),
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return results[0] if results else None
        return None
    except RequestException:
        return None


def get_poster_url(title, year=None):
    result = search_movie(title, year)
    if not result:
        return None
    path = result.get("poster_path")
    if not path:
        return None
    return f"{TMDB_IMAGE_BASE}{path}"


def get_trending(limit=12):
    if not _is_enabled():
        return []
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/trending/movie/week",
            params={"api_key": TMDB_API_KEY},
            timeout=10,
            verify=certifi.where(),
        )
        if response.status_code != 200:
            return []
        results = response.json().get("results", [])
        return results[:limit]
    except RequestException:
        return []
