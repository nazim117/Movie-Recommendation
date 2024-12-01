import pandas as pd

def recommend_movies(movie_title, movies, genre_matrix, nn_model, n_recommendations=5):
    try:
        movie_pos = movies[movies["primaryTitle"] == movie_title].index[0]
        movie_row = movies.index.get_loc(movie_pos)
    except IndexError:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    movie_vector = genre_matrix[movie_row]

    distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=min(n_recommendations + 1, len(movies)))
    
    recommended_indices = indices[0][1:]
    valid_indices = [movies.index[i] for i in recommended_indices if i < len(movies)]

    if len(valid_indices) < n_recommendations:
        print(f"Warning: Only {len(valid_indices)} recommendations available for '{movie_title}'.")

    recommended_movies = movies.loc[valid_indices, ["primaryTitle", "averageRating", "genres"]]
    return recommended_movies