import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

title_basics = pd.read_csv("data/title.basics.tsv/title.basics.tsv", sep="\t", na_values="\\N")
title_ratings = pd.read_csv("data/title.ratings.tsv/title.ratings.tsv", sep="\t", na_values="\\N")

movies = pd.merge(title_basics, title_ratings, on="tconst")

movies = movies.dropna(subset=["primaryTitle", "averageRating", "genres"])

movies = movies[movies["titleType"] == "movie"]

movies.reset_index(drop=True, inplace=True)

print(movies.head())

tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genres"])

nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
nn_model.fit(genre_matrix)

def recommend_movies(movie_title, n_recommendations=5):
    movie_index = movies[movies["primaryTitle"] == movie_title].index[0]
    
    movie_vector = genre_matrix[movie_index]
    
    distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)
    
    recommended_indices = indices[0][1:]
    
    recommended_movies = movies.iloc[recommended_indices][["primaryTitle", "averageRating", "genres"]]
    return recommended_movies

print(recommend_movies("The Matrix", n_recommendations=5))
