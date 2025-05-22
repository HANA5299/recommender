import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    links = pd.read_csv('links.csv')
    return movies, ratings, links

movies, ratings, links = load_data()

# Preprocessing
links.dropna(subset=['tmdbId'], inplace=True)
links['tmdbId'] = links['tmdbId'].astype(int)
movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_scores(movie_id):
    idx = movies[movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_ids = [movies.iloc[i[0]]['movieId'] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    return dict(zip(movie_ids, norm_scores))

# Collaborative Filtering model
@st.cache_resource
def train_model():
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model

model = train_model()

def get_collab_scores(user_id):
    all_movie_ids = movies['movieId'].unique()
    rated_ids = ratings[ratings['userId'] == user_id]['movieId'].values
    unseen_ids = [mid for mid in all_movie_ids if mid not in rated_ids]
    predictions = [model.predict(user_id, mid) for mid in unseen_ids]
    movie_ids = [int(p.iid) for p in predictions]
    scores = [p.est for p in predictions]
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    return dict(zip(movie_ids, norm_scores))

def hybrid_recommend(user_id, liked_movie_id, alpha=0.5, top_n=10):
    collab = get_collab_scores(user_id)
    content = get_content_scores(liked_movie_id)
    combined_scores = {}
    for mid in collab:
        c_score = content.get(mid, 0)
        hybrid_score = alpha * collab[mid] + (1 - alpha) * c_score
        combined_scores[mid] = hybrid_score
    top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_movie_ids = [mid for mid, _ in top_movies]
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()

# -------- Streamlit UI --------

st.title("Hybrid Movie Recommender System")

user_ids = ratings['userId'].unique()
user_id = st.selectbox("Select User ID", sorted(user_ids))

movie_titles = movies['title'].unique()
liked_movie_title = st.selectbox("Select a Movie You Like", sorted(movie_titles))
liked_movie_id = movies[movies['title'] == liked_movie_title]['movieId'].values[0]

alpha = st.slider("Hybrid Alpha (0 = Content-Based, 1 = Collaborative)", 0.0, 1.0, 0.5, 0.1)

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = hybrid_recommend(user_id, liked_movie_id, alpha=alpha)
        st.success("Here are your movie recommendations:")
        for i, title in enumerate(recommendations, 1):
            st.write(f"{i}. {title}")
