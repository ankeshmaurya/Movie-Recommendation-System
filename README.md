# CineMind AI - SIH-Grade Movie Recommendation Platform

An end-to-end, production-ready movie recommendation system that combines unsupervised learning, collaborative filtering, hybrid ranking, and a conversational AI assistant. Built for SIH-level demos with a scalable architecture, advanced EDA, and a Netflix-inspired Streamlit UI.

## Highlights
- User clustering with KMeans for segmentation
- User-user collaborative filtering and movie similarity with cosine distance
- Hybrid recommendation logic with explanations and cold-start handling
- Conversational assistant with memory and AI-generated narratives
- TMDB integration for posters and trending movies
- Advanced EDA with PCA, heatmaps, popularity, and activity analytics

## Project Structure
```
app/
	app.py
data/
	movies.csv
	ratings.csv
model/
	artifacts.joblib
notebooks/
	EDA.ipynb
	Colab_MovieRec.ipynb
src/
	chatbot.py
	config.py
	data_loader.py
	feature_engineering.py
	recommender.py
	tmdb.py
	train.py
requirements.txt
.env.example
```

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file (optional but recommended for posters and AI replies):
```
GROQ_API_KEY=your_groq_key
TMDB_API_KEY=your_tmdb_key
```

## Dataset (MovieLens)
Place `movies.csv` and `ratings.csv` from MovieLens in the `data/` folder.

## Train Models
```bash
python src/train.py
```

This produces `model/artifacts.joblib` containing:
- user-movie matrix
- top-K neighbor indices and scores
- clusters and PCA embeddings
- popularity and activity analytics

## Run the App
```bash
streamlit run app/app.py
```

## Deployment
### Streamlit Cloud
1. Push to GitHub
2. Set `STREAMLIT_APP_FILE=app/app.py`
3. Add secrets for `GROQ_API_KEY` and `TMDB_API_KEY`

### Hugging Face Spaces
```
sdk: streamlit
app_file: app/app.py
python_version: 3.10
```
Add secrets in the Space settings.

### Render
Use a `Python` web service:
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`

## Notes
- For cold-start users, the system defaults to trending or most popular titles.
- Posters and trending require a TMDB key.
- AI chatbot responses require a GROQ key.
- For large MovieLens data, keep Streamlit analytics to summary charts.
