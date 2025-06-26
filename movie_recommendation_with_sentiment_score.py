import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import streamlit as st

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
reviews = pd.read_csv("reviews.csv")  # should contain 'movieId' and 'review'

#  Remove duplicate titles to avoid reindex issues
movies = movies.drop_duplicates(subset='title')

# Merge movie and rating data
movie_data = pd.merge(ratings, movies, on='movieId')
movie_data.dropna(inplace=True)

# Create user-item rating matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Collaborative filtering similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Genre-based similarity
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])
genre_similarity = cosine_similarity(genre_matrix)
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['title'], columns=movies['title'])

# Sentiment analysis on reviews
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity


reviews['sentiment'] = reviews['review'].astype(str).apply(get_sentiment)
avg_sentiment = reviews.groupby('movieId')['sentiment'].mean().reset_index()
avg_sentiment.columns = ['movieId', 'sentiment_score']

# Merge sentiment with movie titles
movie_sentiment = pd.merge(movies[['movieId', 'title']], avg_sentiment, on='movieId', how='left')
movie_sentiment.set_index('title', inplace=True)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movie_similarity_df or movie_title not in genre_similarity_df:
        return ["Movie not found"]

    collab_scores = movie_similarity_df[movie_title]
    genre_scores = genre_similarity_df[movie_title]
    base_scores = (collab_scores + genre_scores) / 2

    # Normalize sentiment scores
    sentiment_scores = movie_sentiment['sentiment_score'].fillna(0)
    if sentiment_scores.max() != sentiment_scores.min():
        sentiment_scores = (sentiment_scores - sentiment_scores.min()) / (sentiment_scores.max() - sentiment_scores.min())
    else:
        sentiment_scores[:] = 0

    # Reindex safely with unique titles
    sentiment_scores = sentiment_scores.reindex(base_scores.index).fillna(0)

    final_scores = (0.7 * base_scores) + (0.3 * sentiment_scores)
    recommendations = final_scores.sort_values(ascending=False)[1:6]
    return list(recommendations.index)

# Streamlit UI
# To run: `streamlit run movie_recommendation_with_sentiment_score.py`
st.title("🎬 Movie Recommendation System with Sentiment Filtering")

movie_list = movies['title'].unique()
selected_movie = st.selectbox("Select a movie you like:", ["🔽 Select a movie"] + list(movie_list))

if selected_movie != "🔽 Select a movie" and st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.markdown(f"🎥 You liked the movie **{selected_movie}**.")
    st.markdown("🔝 **Top 5 recommendations for you (with sentiment scores):**")
    for i, rec in enumerate(recommendations):
        score = movie_sentiment.loc[rec]['sentiment_score'] if rec in movie_sentiment.index else 0
        st.write(f"{i+1}. {rec} (Sentiment Score: {score:.2f})")
