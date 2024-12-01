from data_processing import load_data, create_genre_matrix, build_nn_model, split_data
import joblib
import os

def train_and_save_model(data_paths, output_dir):
    title_basics_path, title_ratings_path = data_paths

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading data from {title_basics_path} and {title_ratings_path}...")
    movies = load_data(title_basics_path, title_ratings_path)
    print(f"Data loaded successfully. Total movies: {len(movies)}")
    
    train_data, test_data = split_data(movies)
    print(f"Data split into training ({len(train_data)}) and testing ({len(test_data)}).")
    
    print("Creating genre matrix for training data...")
    genre_matrix, tfidf = create_genre_matrix(train_data)
    print("Genre matrix created.")
    
    print("Building nearest neighbors model...")
    nn_model = build_nn_model(genre_matrix, n_neighbors=50)
    print("Model built successfully.")
    
    test_data_csv_path = os.path.join(output_dir, "test_data.csv")
    test_data.to_csv(test_data_csv_path, index=False)
    print(f"Test data saved as CSV at {test_data_csv_path}.")
    
    tfidf_pkl_path = os.path.join(output_dir, "tfidf.pkl")
    joblib.dump(tfidf, tfidf_pkl_path)
    print(f"TF-IDF vectorizer saved at {tfidf_pkl_path}.")
    
    nn_model_pkl_path = os.path.join(output_dir, "nn_model.pkl")
    joblib.dump(nn_model, nn_model_pkl_path)
    print(f"Nearest Neighbors model saved at {nn_model_pkl_path}.")
    
    print("Training and saving process completed successfully.")

if __name__ == '__main__':
    title_basics_path = "data/title.basics.tsv/title.basics.tsv"
    title_ratings_path = "data/title.ratings.tsv/title.ratings.tsv"
    output_dir = "output"
    
    train_and_save_model((title_basics_path, title_ratings_path), output_dir)
