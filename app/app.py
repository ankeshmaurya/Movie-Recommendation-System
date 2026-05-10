from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.chatbot import chatbot_recommendation
from src.config import DATA_DIR, MODEL_DIR
from src.recommender import MovieRecommender
from src.tmdb import get_poster_url, get_trending

st.set_page_config(page_title="CineMind AI", layout="wide")

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Manrope:wght@400;600&display=swap');

:root {
    --bg: #f7f5f2;
    --bg-accent: #f1eee8;
    --accent: #e4572e;
    --ink: #1b1b1f;
    --muted: #5c5c66;
    --card: #ffffff;
    --shadow: rgba(24, 24, 33, 0.12);
}

html, body, [class*="css"]  {
    font-family: 'Manrope', sans-serif;
    background: radial-gradient(circle at top left, #ffffff, var(--bg));
    color: var(--ink);
}

.stApp {
    background: radial-gradient(circle at top left, #ffffff, var(--bg));
    color: var(--ink);
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.35rem;
}

.subtitle {
    color: var(--muted);
    margin-bottom: 1.5rem;
}

.card {
    background: var(--card);
    border-radius: 20px;
    padding: 1rem;
    border: 1px solid rgba(27, 27, 31, 0.06);
    box-shadow: 0 12px 30px var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 34px rgba(24, 24, 33, 0.18);
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    margin-top: 1.4rem;
    color: var(--ink);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, var(--bg-accent));
    border-right: 1px solid rgba(27, 27, 31, 0.08);
}

section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

.stButton > button {
    background: var(--accent);
    color: #ffffff;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1.4rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    border-bottom: 1px solid rgba(27, 27, 31, 0.08);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: var(--accent) !important;
}

.stTabs [data-baseweb="tab-border"] {
    background: rgba(27, 27, 31, 0.08) !important;
}

.stTextArea textarea {
    background: #ffffff;
    color: var(--ink);
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff;
    color: var(--ink);
}

.stNumberInput input {
    background: #ffffff;
    color: var(--ink);
}

.stNumberInput button {
    background: #ffffff;
    color: var(--ink);
    border: 1px solid rgba(27, 27, 31, 0.12);
}

.stSlider [data-baseweb="slider"] > div {
    background: var(--accent);
}

.stSlider [role="slider"] {
    background: #ffffff !important;
    border: 1px solid rgba(27, 27, 31, 0.12) !important;
}

.stSlider [data-baseweb="slider"] div[role="presentation"] {
    background: rgba(27, 27, 31, 0.12) !important;
}

.stSlider [data-baseweb="slider"] > div {
    background: var(--accent);
}

.stCaption {
    color: var(--muted);
}
</style>
""",
        unsafe_allow_html=True,
)


@st.cache_resource
def load_recommender():
    return MovieRecommender()


@st.cache_data
def load_data():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    if not movies_path.exists() or not ratings_path.exists():
        return None, None
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def render_movie_cards(titles, metadata, columns=5):
    cols = st.columns(columns)
    for idx, title in enumerate(titles):
        with cols[idx % columns]:
            meta_row = metadata[metadata["title"] == title]
            year = None
            if not meta_row.empty:
                year = meta_row.iloc[0].get("year")
            poster = get_poster_url(title, year)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.markdown(
                    "<div style='height:220px;background:#ece9e2;border-radius:14px;'></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**{title}**")
            if year:
                st.caption(f"{year}")
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="title">CineMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-first movie discovery powered by clustering, similarity, and conversational intelligence.</div>',
    unsafe_allow_html=True,
)

if not (MODEL_DIR / "artifacts.joblib").exists():
    st.error("Model artifacts missing. Run `python src/train.py` after adding MovieLens data.")
    st.stop()

recommender = load_recommender()

with st.sidebar:
    st.markdown("**Viewer Profile**")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    n_recs = st.slider("Recommendations", min_value=4, max_value=12, value=6)
    st.markdown("**Settings**")
    show_trending = st.toggle("Show TMDB Trending", value=True)
    st.caption("Add TMDB_API_KEY in .env for posters and trending.")

tab_explore, tab_chat, tab_analytics = st.tabs(["Explore", "Chatbot", "Analytics"])

with tab_explore:
    st.markdown("<div class='section-title'>Trending Now</div>", unsafe_allow_html=True)
    trending = get_trending(limit=10) if show_trending else []
    if trending:
        titles = [movie["title"] for movie in trending]
        temp_meta = pd.DataFrame(
            {
                "title": titles,
                "year": [movie.get("release_date", "")[:4] for movie in trending],
            }
        )
        render_movie_cards(titles, temp_meta, columns=5)
    else:
        popular = recommender.get_popular_movies(n_movies=10)
        render_movie_cards(popular, recommender.movie_metadata, columns=5)

    st.markdown("<div class='section-title'>Personalized Picks</div>", unsafe_allow_html=True)
    if st.button("Generate Picks"):
        personal = recommender.recommend_for_user(user_id, n_movies=n_recs)
        render_movie_cards(personal, recommender.movie_metadata, columns=3)

    st.markdown("<div class='section-title'>Movie Similarity Finder</div>", unsafe_allow_html=True)
    movie_choices = recommender.popularity["title"].head(400).tolist()
    seed_title = st.selectbox("Pick a movie", movie_choices)
    if seed_title:
        similar = recommender.recommend_similar_movies(seed_title, n_movies=n_recs)
        render_movie_cards(similar, recommender.movie_metadata, columns=3)

with tab_chat:
    st.markdown("<div class='section-title'>Ask the CineMind Assistant</div>", unsafe_allow_html=True)
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = {}

    user_input = st.text_area(
        "Tell me what you feel like watching",
        placeholder="I want a suspenseful sci-fi like Interstellar",
    )
    if st.button("Ask Assistant") and user_input:
        result = chatbot_recommendation(
            user_input,
            user_id=user_id,
            recommender=recommender,
            memory=st.session_state.chat_memory,
        )
        st.markdown(result["response"])
        render_movie_cards(result["recommendations"], recommender.movie_metadata, columns=3)

with tab_analytics:
    st.markdown("<div class='section-title'>Behavior & Clusters</div>", unsafe_allow_html=True)
    movies, ratings = load_data()
    if ratings is None:
        st.info("Add MovieLens files to data/ to unlock analytics charts.")
    else:
        rating_fig = px.histogram(ratings, x="rating", nbins=20, title="Ratings Distribution")
        st.plotly_chart(rating_fig, use_container_width=True)

        genre_counts = (
            movies["genres"].str.split("|").explode().value_counts().reset_index()
        )
        genre_counts.columns = ["genre", "count"]
        genre_fig = px.bar(genre_counts.head(15), x="genre", y="count", title="Top Genres")
        st.plotly_chart(genre_fig, use_container_width=True)

        activity_fig = px.bar(
            recommender.user_activity.sort_values("ratings_count", ascending=False).head(25),
            x="userId",
            y="ratings_count",
            title="Most Active Users",
        )
        st.plotly_chart(activity_fig, use_container_width=True)

        popular_fig = px.bar(
            recommender.popularity.head(15), x="title", y="rating_count", title="Most Rated Movies"
        )
        st.plotly_chart(popular_fig, use_container_width=True)

        pca_df = pd.DataFrame(
            recommender.pca_embeddings, columns=["PC1", "PC2"]
        )
        pca_df["cluster"] = recommender.user_clusters
        pca_fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="cluster",
            title="User Clusters (PCA)",
        )
        st.plotly_chart(pca_fig, use_container_width=True)
