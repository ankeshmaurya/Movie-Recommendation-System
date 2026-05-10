from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import DATA_DIR, MODEL_DIR, DEFAULT_N_CLUSTERS
from src.data_loader import load_merged, load_movies
from src.feature_engineering import (
    build_movie_user_matrix,
    build_user_movie_matrix,
    compute_pca_embeddings,
    compute_popularity,
    compute_topk_similarities,
    compute_user_activity,
)


def train(n_clusters=DEFAULT_N_CLUSTERS):
    if not (DATA_DIR / "movies.csv").exists() or not (DATA_DIR / "ratings.csv").exists():
        raise FileNotFoundError(
            "Missing MovieLens files. Add movies.csv and ratings.csv to the data/ folder."
        )

    data = load_merged()
    movies = load_movies()

    user_movie_matrix = build_user_movie_matrix(data)
    movie_user_matrix = build_movie_user_matrix(user_movie_matrix)
    neighbor_artifacts = compute_topk_similarities(
        user_movie_matrix, movie_user_matrix, k=50
    )

    scaler = StandardScaler()
    scaled_users = scaler.fit_transform(user_movie_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_users)

    pca_embeddings, pca, pca_scaler = compute_pca_embeddings(user_movie_matrix)

    popularity = compute_popularity(data)
    user_activity = compute_user_activity(data)

    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        cluster_users = user_movie_matrix.index[clusters == cluster_id]
        cluster_data = data[data["userId"].isin(cluster_users)]
        top_movies = (
            cluster_data.groupby("title")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .index.tolist()
        )
        top_genres = (
            cluster_data["genres"].fillna("").str.split("|").explode().value_counts()
            .head(10)
            .index.tolist()
        )
        cluster_profiles[int(cluster_id)] = {
            "top_movies": top_movies,
            "top_genres": top_genres,
        }

    movie_metadata = movies.merge(popularity, on="title", how="left")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "user_movie_matrix": user_movie_matrix,
        "movie_user_matrix": movie_user_matrix,
        "user_neighbor_indices": neighbor_artifacts["user_indices"],
        "user_neighbor_scores": neighbor_artifacts["user_scores"],
        "movie_neighbor_indices": neighbor_artifacts["movie_indices"],
        "movie_neighbor_scores": neighbor_artifacts["movie_scores"],
        "user_clusters": clusters,
        "cluster_profiles": cluster_profiles,
        "movie_metadata": movie_metadata,
        "popularity": popularity,
        "user_activity": user_activity,
        "pca_embeddings": pca_embeddings,
        "pca": pca,
        "pca_scaler": pca_scaler,
        "kmeans": kmeans,
        "scaler": scaler,
    }

    joblib.dump(artifacts, MODEL_DIR / "artifacts.joblib")
    return artifacts


if __name__ == "__main__":
    train()
    print("Training completed. Artifacts saved to model/artifacts.joblib")
