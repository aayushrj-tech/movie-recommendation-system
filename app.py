

# Imports & setup
import streamlit as st
import pandas as pd
import pickle
import requests
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity




load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"



st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load data & vectors
df = pd.read_csv("data/movies_cleaned.csv")

with open("data/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("data/movie_vectors.pkl", "rb") as f:
    movie_vectors = pickle.load(f)

# helper functions

@st.cache_data(show_spinner=False, ttl=86400)  # 24 hours
def fetch_movie_poster(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY}

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.exceptions.RequestException:
        pass

    return None


def recommend_any_movie(movie_name, top_n=5):
    # Safety check (even though UI restricts input)
    if movie_name not in df["title"].values:
        return []

    # Get index of the selected movie
    idx = df[df["title"] == movie_name].index[0]

    # Compute cosine similarity
    scores = cosine_similarity(
        movie_vectors[idx],
        movie_vectors
    ).flatten()

    # Sort by similarity (descending)
    indices = scores.argsort()[::-1]

    # Remove the movie itself
    indices = [i for i in indices if i != idx]

    # Return top-N recommendations
    return df.iloc[indices[:top_n]]





# ui header
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one")

# ui controls
movie_list = sorted(df["title"].unique())

selected_movie = st.selectbox(
    "Select a movie",
    options=[""] + movie_list
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=15,
    value=5
)

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# Button + Spinner
if st.button("Recommend"):
    if selected_movie == "":
        st.warning("Please select a movie")
    else:
        with st.spinner("🎬 Finding similar movies..."):
            st.session_state.recommendations = recommend_any_movie(
                selected_movie, top_n
            )


# Poster cache


recommendations = st.session_state.recommendations

if recommendations is None:
    st.stop()


if len(recommendations) == 0:
    st.error("No recommendations found")
    st.stop()

st.caption("Showing posters for top 5 recommendations")

# card layouts
max_posters = 5
cols = st.columns(5)

for i, row in enumerate(recommendations.itertuples()):
    if i >= max_posters:
        break

    title = row.title

    # cache poster manually
    with cols[i]:
        poster = fetch_movie_poster(row.id)

        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.write("🖼️ Poster not available")



        st.markdown(f"### {title}")
        st.markdown(f"⭐ Rating: {row.vote_average}")


if len(recommendations) > max_posters:
    st.subheader("More recommendations")
    for row in recommendations.iloc[max_posters:].itertuples():
        st.write(f"• {row.title} (⭐ {row.vote_average})")
