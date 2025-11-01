import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()
tfidf = model_data["tfidf"]
tfidf_matrix = model_data["tfidf_matrix"]
indices = model_data["indices"]
movies = model_data["movies"]

# -----------------------------
# TMDB Poster Fetching Helper
# -----------------------------
def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    try:
        api_key = "YOUR_TMDB_API_KEY"  # 🔑 replace with your TMDB key
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/300x450?text=No+Image"
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

# -----------------------------
# Recommendation Function
# -----------------------------
def get_recommendations(title, top_n=10):
    """Return top N movie titles similar to the given title."""
    if title not in indices:
        return []

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    movie_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()

    sorted_indices = sim_scores.argsort()[::-1]
    valid_indices = [i for i in sorted_indices if i != idx and i < len(movies)]
    top_indices = valid_indices[:top_n]

    recommended_movies = movies.iloc[top_indices]
    return recommended_movies

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎬 Personalized Movie Recommender", layout="wide")
st.title("🎥 Personalized Movie Recommendation System")
st.write("Type a movie name below to get personalized recommendations!")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select or type a movie title:", movie_list)

if st.button("Get Recommendations"):
    with st.spinner("Finding similar movies..."):
        recommendations = get_recommendations(selected_movie)

        if isinstance(recommendations, str) or len(recommendations) == 0:
            st.error("❌ Movie not found or no similar movies available.")
        else:
            cols = st.columns(5)
            for i, (index, row) in enumerate(recommendations.iterrows()):
                with cols[i % 5]:
                    poster_url = fetch_poster(row['id'])
                    st.image(poster_url, width=150)
                    st.caption(row['title'])
