from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import MODEL_DIR


class MovieRecommender:
    def __init__(self, artifacts_path=None):
        artifacts_path = artifacts_path or (MODEL_DIR / "artifacts.joblib")
        if not Path(artifacts_path).exists():
            raise FileNotFoundError(
                "Missing model artifacts. Run `python src/train.py` first."
            )
        artifacts = joblib.load(artifacts_path)
        self.user_movie_matrix = artifacts["user_movie_matrix"]
        self.movie_user_matrix = artifacts["movie_user_matrix"]
        self.user_neighbor_indices = artifacts["user_neighbor_indices"]
        self.user_neighbor_scores = artifacts["user_neighbor_scores"]
        self.movie_neighbor_indices = artifacts["movie_neighbor_indices"]
        self.movie_neighbor_scores = artifacts["movie_neighbor_scores"]
        self.user_clusters = artifacts["user_clusters"]
        self.cluster_profiles = artifacts["cluster_profiles"]
        self.movie_metadata = artifacts["movie_metadata"]
        self.popularity = artifacts["popularity"]
        self.user_activity = artifacts["user_activity"]
        self.pca_embeddings = artifacts["pca_embeddings"]

        self.user_index = self.user_movie_matrix.index
        self.movie_index = self.movie_user_matrix.index
        self.movie_to_idx = {title: idx for idx, title in enumerate(self.movie_index)}

    def get_popular_movies(self, n_movies=10):
        return self.popularity["title"].head(n_movies).tolist()

    def recommend_for_user(self, user_id, n_movies=10, k_sim=20):
        if user_id not in self.user_index:
            return self.get_popular_movies(n_movies)

        user_pos = self.user_index.get_loc(user_id)
        neighbor_ids = self.user_neighbor_indices[user_pos]
        neighbor_scores = self.user_neighbor_scores[user_pos]
        neighbor_ids = neighbor_ids[1 : k_sim + 1]
        neighbor_scores = neighbor_scores[1 : k_sim + 1]
        weighted_scores = self.user_movie_matrix.iloc[neighbor_ids].T.dot(
            neighbor_scores
        )
        scores = pd.Series(weighted_scores, index=self.user_movie_matrix.columns)
        watched = self.user_movie_matrix.loc[user_id]
        watched_titles = watched[watched > 0].index
        scores = scores.drop(watched_titles, errors="ignore")
        scores = scores.sort_values(ascending=False)
        if scores.empty:
            return self.get_popular_movies(n_movies)
        return scores.head(n_movies).index.tolist()

    def recommend_similar_movies(self, title, n_movies=10):
        if title not in self.movie_to_idx:
            return self.get_popular_movies(n_movies)
        movie_pos = self.movie_to_idx[title]
        neighbor_ids = self.movie_neighbor_indices[movie_pos][1 : n_movies + 1]
        return [self.movie_index[idx] for idx in neighbor_ids]

    def recommend_from_cluster(self, user_id, n_movies=10):
        if user_id not in self.user_index:
            return self.get_popular_movies(n_movies)
        cluster_id = int(self.user_clusters[self.user_index.get_loc(user_id)])
        cluster_movies = self.cluster_profiles.get(cluster_id, {}).get("top_movies", [])
        if not cluster_movies:
            return self.get_popular_movies(n_movies)
        return cluster_movies[:n_movies]

    def recommend_by_genres(self, genres, n_movies=10):
        if not genres:
            return self.get_popular_movies(n_movies)
        genre_pattern = "|".join([genre.strip() for genre in genres])
        matches = self.movie_metadata[
            self.movie_metadata["genres"].fillna("").str.contains(
                genre_pattern, case=False, regex=True
            )
        ]
        ranked = matches.sort_values(["rating_count", "rating_mean"], ascending=False)
        if ranked.empty:
            return self.get_popular_movies(n_movies)
        return ranked["title"].head(n_movies).tolist()

    def hybrid_recommendations(self, user_id, seed_title=None, genres=None, n_movies=10):
        sources = {
            "user": self.recommend_for_user(user_id, n_movies=15),
            "cluster": self.recommend_from_cluster(user_id, n_movies=15),
        }
        if seed_title:
            sources["similar"] = self.recommend_similar_movies(seed_title, n_movies=15)
        if genres:
            sources["genre"] = self.recommend_by_genres(genres, n_movies=15)

        scores = {}
        reasons = {}
        for source, items in sources.items():
            for rank, title in enumerate(items, start=1):
                scores[title] = scores.get(title, 0) + (1 / rank)
                reasons.setdefault(title, set()).add(source)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [title for title, _ in ranked][:n_movies]
        return recommendations, reasons

    def explain_recommendations(self, recommendations, reasons):
        explanations = {}
        for title in recommendations:
            source_tags = reasons.get(title, set())
            reason_text = []
            if "user" in source_tags:
                reason_text.append("similar users enjoyed it")
            if "cluster" in source_tags:
                reason_text.append("it trends in your taste cluster")
            if "similar" in source_tags:
                reason_text.append("it matches your selected movie")
            if "genre" in source_tags:
                reason_text.append("it aligns with your genres")
            explanations[title] = ", ".join(reason_text) if reason_text else "popular pick"
        return explanations


def recommend_movies(user_id, n_movies=5):
    recommender = MovieRecommender()
    return recommender.recommend_for_user(user_id, n_movies=n_movies)
