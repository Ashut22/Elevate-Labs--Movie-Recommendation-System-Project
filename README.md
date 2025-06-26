# Elevate-Labs--.Movie-Recommendation-System-Project
# Movie Recommendation System (With and Without Sentiment Analysis)

This project includes two versions of a movie recommendation system:

1. **Without Sentiment Analysis**: Uses collaborative filtering and genre-based content filtering.
2. **With Sentiment Analysis**: Enhances recommendations by incorporating sentiment polarity of user reviews.

## Features

- Recommends top 5 movies similar to the selected movie.
- Two modes:
  - **Basic**: Collaborative + genre-based filtering.
  - **Advanced**: Adds review sentiment scoring to improve personalization.
- Interactive interface using Streamlit.

## Project Files

- `movie_recommendation_without_sentiment_score.py` – Recommender using only ratings and genres.
- `movie_recommendation_with_sentiment_score.py` – Recommender using ratings, genres, and sentiment scores.
- `generate_reviews_from_ratings.ipynb` – Generates synthetic reviews from numeric ratings for sentiment analysis.
- `movie_recommendation_without_sentiment_score.ipynb` – Notebook version of the basic recommendation model.

## How It Works

1. Loads datasets: `movies.csv`, `ratings.csv`, and `reviews.csv`.
2. Computes cosine similarity:
   - Between user ratings for collaborative filtering.
   - Between movie genres for content-based filtering.
3. (Optional) Applies sentiment analysis using TextBlob on reviews.
4. Combines scores to rank and recommend the top 5 similar movies.

## How to Run

1. Ensure all CSV files (`movies.csv`, `ratings.csv`, and `reviews.csv`) are in the same directory as the Python files.
2. Open your terminal or command prompt.
3. Navigate to the project directory.
4. Run the desired version using the following commands:

## Notes

- The **sentiment-based model** may take slightly longer to run because it involves merging additional data and applying normalization.
- Sentiment analysis helps make the recommendations more aligned with user preferences based on review tone.

```bash
# To run the basic version (without sentiment analysis):
streamlit run movie_recommendation_without_sentiment_score.py

# To run the advanced version (with sentiment analysis):
streamlit run movie_recommendation_with_sentiment_score.py
