import pandas as pd
import joblib
from model import recommend_movies

def evaluate_model(csv_path, tfidf_path, model_path, movie_title, n_recommendations=5):
    try:
        test_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {csv_path}. Please ensure the file exists.")
        return

    try:
        tfidf = joblib.load(tfidf_path)
    except FileNotFoundError:
        print(f"Error: TF-IDF vectorizer file not found at {tfidf_path}. Please train the model first.")
        return

    try:
        nn_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    genre_matrix = tfidf.transform(test_data["genres"])

    print(f"\nRecommendations for '{movie_title}':")
    try:
        recommendations = recommend_movies(movie_title, test_data, genre_matrix, nn_model, n_recommendations)
        print(recommendations)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    csv_path = "output/test_data.csv"
    tfidf_path = "output/tfidf.pkl"
    model_path = "output/nn_model.pkl"

    movie_title = input("Enter the movie title: ")
    n_recommendations = int(input("Enter the number of recommendations: "))

    evaluate_model(csv_path, tfidf_path, model_path, movie_title, n_recommendations)
