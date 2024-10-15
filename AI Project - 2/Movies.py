import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Load movie dataset
movies = pd.read_csv("C:/Users/hp/Desktop/AI Projects/Ai Project-2/movies.csv", encoding='ISO-8859-1')

# Handle missing values in the 'genres' column
movies['genres'] = movies['genres'].fillna('')

# Vectorize the 'genres' column
cv = CountVectorizer()
vectorized_data = cv.fit_transform(movies['genres'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(vectorized_data)

# Streamlit UI
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter a movie title")

# Movie recommendation function
def recommend_movie(movie_name):
    try:
        idx = movies[movies['title'].str.lower() == movie_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:6]]
        return movies['title'].iloc[movie_indices]
    except IndexError:
        return ["Movie not found. Please try another title."]

# Display recommendations
if st.button('Recommend'):
    recommendations = recommend_movie(movie_name)
    st.write("Recommended Movies:")
    for rec in recommendations:
        st.write(rec)
