import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

def load_data(title_basics_path, title_ratings_path):
    title_basics = pd.read_csv(title_basics_path, sep="\t", na_values="\\N")
    title_ratings = pd.read_csv(title_ratings_path, sep="\t", na_values="\\N")
    movies = pd.merge(title_basics, title_ratings, on="tconst")
    movies = movies.dropna(subset=["primaryTitle", "averageRating", "genres"])
    movies = movies[movies["titleType"] == "movie"]
    movies.reset_index(drop=True, inplace=True)
    return movies

def create_genre_matrix(movies):
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(movies["genres"])
    return genre_matrix, tfidf

def build_nn_model(genre_matrix, n_neighbors=10):
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn_model.fit(genre_matrix)
    return nn_model

def split_data(movies, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(movies, test_size=test_size, random_state=random_state)
    return train_data, test_data
