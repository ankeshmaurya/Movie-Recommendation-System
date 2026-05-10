import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def build_user_movie_matrix(data):
    return data.pivot_table(index="userId", columns="title", values="rating").fillna(0)


def build_movie_user_matrix(user_movie_matrix):
    return user_movie_matrix.T


def compute_topk_similarities(user_movie_matrix, movie_user_matrix, k=50):
    user_model = NearestNeighbors(metric="cosine", algorithm="brute")
    user_model.fit(user_movie_matrix)
    user_distances, user_indices = user_model.kneighbors(
        user_movie_matrix, n_neighbors=min(k + 1, user_movie_matrix.shape[0])
    )

    movie_model = NearestNeighbors(metric="cosine", algorithm="brute")
    movie_model.fit(movie_user_matrix)
    movie_distances, movie_indices = movie_model.kneighbors(
        movie_user_matrix, n_neighbors=min(k + 1, movie_user_matrix.shape[0])
    )

    user_scores = 1 - user_distances
    movie_scores = 1 - movie_distances

    return {
        "user_indices": user_indices,
        "user_scores": user_scores,
        "movie_indices": movie_indices,
        "movie_scores": movie_scores,
    }


def compute_popularity(data):
    popularity = (
        data.groupby("title")
        .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"))
        .sort_values(["rating_count", "rating_mean"], ascending=False)
        .reset_index()
    )
    return popularity


def compute_user_activity(data):
    return (
        data.groupby("userId")
        .agg(ratings_count=("rating", "count"), rating_mean=("rating", "mean"))
        .reset_index()
    )


def compute_pca_embeddings(user_movie_matrix, n_components=2):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(user_movie_matrix)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
    return reduced, pca, scaler
