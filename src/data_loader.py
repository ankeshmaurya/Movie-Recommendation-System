import re
import pandas as pd

from .config import DATA_DIR


def _extract_year(title):
    match = re.search(r"\((\d{4})\)$", str(title))
    return int(match.group(1)) if match else None


def load_movies(path=None):
    file_path = path or (DATA_DIR / "movies.csv")
    movies = pd.read_csv(file_path)
    movies["year"] = movies["title"].apply(_extract_year)
    movies["genres_list"] = movies["genres"].fillna("").str.split("|")
    return movies


def load_ratings(path=None):
    file_path = path or (DATA_DIR / "ratings.csv")
    return pd.read_csv(file_path)


def load_merged(movies=None, ratings=None):
    movies = movies if movies is not None else load_movies()
    ratings = ratings if ratings is not None else load_ratings()
    merged = ratings.merge(movies, on="movieId", how="left")
    return merged
