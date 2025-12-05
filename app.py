# app.py
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommender using genres, keywords and overview (CountVectorizer + Cosine Similarity)")

@st.cache_data(show_spinner=False)
def load_data(csv_path="tmdb_5000_movies.csv"):
    df = pd.read_csv(csv_path)
    # Keep only useful columns; handle possible column name differences
    if 'id' in df.columns and 'movie_id' not in df.columns:
        df = df.rename(columns={'id':'movie_id'})
    needed = []
    for col in ['movie_id','title','overview','genres','keywords']:
        if col in df.columns:
            needed.append(col)
    df = df[needed]
    # fill missing overview
    if 'overview' in df.columns:
        df['overview'] = df['overview'].fillna('')
    else:
        df['overview'] = ''
    return df

def parse_json_column(text):
    """Convert JSON-like list-of-dicts string to list of names."""
    if pd.isna(text):
        return []
    try:
        L = [d['name'] for d in ast.literal_eval(text)]
        return L
    except Exception:
        # If it's already a list (rare) or malformed, try safe fallback
        if isinstance(text, list):
            return text
        return []

@st.cache_data(show_spinner=True)
def prepare_movies(df):
    # Convert genres & keywords
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(parse_json_column)
    else:
        df['genres'] = [[] for _ in range(len(df))]
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(parse_json_column)
    else:
        df['keywords'] = [[] for _ in range(len(df))]

    # Create tags column
    df['tags'] = df['overview'] + " " + df['genres'].apply(lambda x: " ".join(x)) + " " + df['keywords'].apply(lambda x: " ".join(x))
    # Lowercase titles for matching
    df['title_lower'] = df['title'].str.lower()
    return df

@st.cache_data(show_spinner=True)
def build_similarity(df, max_features=5000):
    cv = CountVectorizer(max_features=max_features, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    sim = cosine_similarity(vectors)
    return sim

@st.cache_data(show_spinner=False)
def get_recommendations(title, df, similarity_matrix, topn=10):
    title_lower = title.lower()
    if title_lower not in df['title_lower'].values:
        return None
    idx = df[df['title_lower'] == title_lower].index[0]
    distances = similarity_matrix[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:topn+1]
    recommendations = []
    for i in movie_list:
        recommendations.append((df.iloc[i[0]]['title'], float(i[1])))
    return recommendations

# Sidebar: load file / options
st.sidebar.header("Settings")
csv_path = st.sidebar.text_input("CSV path (relative)", value="tmdb_5000_movies.csv")
max_features = st.sidebar.slider("CountVectorizer max features", min_value=1000, max_value=20000, value=5000, step=1000)
topn = st.sidebar.slider("Number of recommendations", min_value=3, max_value=20, value=5)

# Load & prepare
with st.spinner("Loading dataset..."):
    try:
        movies_df = load_data(csv_path)
    except FileNotFoundError:
        st.error(f"CSV not found at: {csv_path}. Put 'tmdb_5000_movies.csv' in the app folder or change the path.")
        st.stop()
    movies_df = prepare_movies(movies_df)

# Build similarity (only once cached)
with st.spinner("Computing similarity... (happens once)"):
    similarity = build_similarity(movies_df, max_features=max_features)

# Search box
st.subheader("Search a movie")
movie_input = st.text_input("Enter a movie title (example: Avatar)", value="Avatar")

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        recs = get_recommendations(movie_input, movies_df, similarity, topn=topn)
        if recs is None:
            st.warning("Movie not found in the dataset. Try a different title or check spelling.")
        else:
            st.success(f"Top {topn} recommendations for '{movie_input}':")
            for i, (title, score) in enumerate(recs, start=1):
                st.write(f"**{i}. {title}**  — similarity {score:.3f}")

# Optional: sample random movie & show its tags
if st.checkbox("Show sample movie & tags"):
    sample = movies_df.sample(1).iloc[0]
    st.write("**Title:**", sample['title'])
    st.write("**Overview:**", sample['overview'] if sample['overview'] else "—")
    st.write("**Genres:**", ", ".join(sample['genres']))
    st.write("**Keywords:**", ", ".join(sample['keywords']))

st.markdown("---")
st.write("Built with CountVectorizer + Cosine Similarity. Want posters or TMDB images? Provide your TMDB API key and I can add poster lookup.")
