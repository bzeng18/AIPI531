import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import seaborn as sns
from surprise import Dataset, Reader, KNNBasic


# Path to the movies data file
movies_file = "ml-100k/u.item"

# Read the data using pandas
movies_df = pd.read_csv(
    movies_file,
    sep="|",
    encoding="latin-1",
    names=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],
)


column_names=['user_id','movie_id','rating','timestamp']
ratings_df = pd.read_csv('ml-100k/u.data',sep='\t',names=column_names)


def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe containing 3 columns (userId, movie_id, rating)
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['user_id'].nunique()
    N = df['movie_id'].nunique()

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movie_id"]), list(range(N))))
    
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movie_id"])))
    
    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [movie_mapper[i] for i in df['movie_id']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings_df)



# evaluate the sparsity of X
def calculate_sparsity(X):
    """
    This function calculates sparsity percentage of a matrix.
    
    Args:
        X: rating matrix
    
    Returns:
        sparsity: a float number between 0 and 1 representing the sparsity percentage
    """
    sparsity = float(len(X.nonzero()[0]))
    sparsity /= (X.shape[0] * X.shape[1])
    sparsity *= 100
    return sparsity

calculate_sparsity(X)




def get_recommendations_user_based(target_user_id, top_k):
    """
    This function generates top_k movie recommendations for a target user.
    
    Args:
        target_user_id: The ID of the user for whom recommendations are to be generated.
        top_k: The number of recommendations to generate.
    
    Returns:
        recommended_items: A list of tuples. Each tuple contains the ID and title of a recommended movie.
    """
    # Load the dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

    # Build the user-based collaborative filtering model
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(k=10, sim_options=sim_options)

    # Train the model
    trainset = data.build_full_trainset()
    model.fit(trainset)

    # Get items rated by the target user
    target_user_items = trainset.ur[trainset.to_inner_uid(target_user_id)]
    # Get items not yet rated by the target user
    target_user_unrated_items = [item_id for item_id in trainset.all_items() if item_id not in target_user_items]

    predictions = []
    for item_id in target_user_unrated_items:
        if item_id in movies_df['movie_id'].values:
            movie_title = movies_df.loc[movies_df['movie_id'] == item_id, 'movie_title'].values[0]
            predicted_rating = model.predict(target_user_id, item_id).est
            predictions.append((item_id, movie_title, predicted_rating))

    # Sort the predictions by rating in descending order
    predictions.sort(key=lambda x: x[2], reverse=True)

    # Get the top recommended items with movie names
    recommended_items = [(item_id, movie_title) for item_id, movie_title, _ in predictions[:top_k]]
    
    return recommended_items



def get_recommendations_item_based(target_user_id, top_k):

    # Load the dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

    # Build the item-based collaborative filtering model
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(k=10, sim_options=sim_options)

    # Train the model
    trainset = data.build_full_trainset()
    model.fit(trainset)

    # Get recommendations for a target user
    target_user_id = 123
    target_user_items = trainset.ur[trainset.to_inner_uid(target_user_id)]
    target_user_unrated_items = [item_id for item_id in trainset.all_items() if item_id not in target_user_items]

    predictions = []
    for item_id in target_user_unrated_items:
        if item_id in movies_df['movie_id'].values:
            movie_title = movies_df.loc[movies_df['movie_id'] == item_id, 'movie_title'].values[0]
            predicted_rating = model.predict(target_user_id, item_id).est
            predictions.append((item_id, movie_title, predicted_rating))
        else:
            movie_title = "Unknown"  # Set a default movie title if the ID is not found
            predicted_rating = model.predict(target_user_id, item_id).est
            predictions.append((item_id, movie_title, predicted_rating))

    # Sort the predictions by rating in descending order
    predictions.sort(key=lambda x: x[2], reverse=True)

    # Get the top recommended items with movie names
    top_k = 10
    recommended_items = [(item_id, movie_title) for item_id, movie_title, _ in predictions[:top_k]]
    return recommended_items


def get_similar_movies(target_movie_name, k):
    """
    This function generates k similar movies for a target movie.
    
    Args:
        target_movie_name: The name of the movie for which similar movies are to be found.
        k: The number of similar movies to generate.
    
    Returns:
        similar_movies: A list of similar movies.
    """
    # Load the dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

    # Build the item-based collaborative filtering model
    sim_options = {'name': 'cosine', 'user_based': False}
    model = KNNBasic(k=10, sim_options=sim_options)

    # Train the model
    trainset = data.build_full_trainset()
    model.fit(trainset)

    # Get the movie ID for the target movie
    target_movie_id = movies_df.loc[movies_df['movie_title'] == target_movie_name, 'movie_id'].values[0]

    # Get the IDs of similar movies
    similar_movie_ids = model.get_neighbors(target_movie_id, k=k)

    # Get the names of similar movies
    similar_movies = [(movies_df.loc[movies_df['movie_id'] == movie_id, 'movie_title'].values[0]) for movie_id in similar_movie_ids]

    return similar_movies



import argparse

def main(method, target_user_id, top_k, target_movie_name=None):
    if method == "user_based":
        recommended_items = get_recommendations_user_based(target_user_id, top_k)
    elif method == "item_based":
        recommended_items = get_recommendations_item_based(target_user_id, top_k)
    elif method == "similar_movies":
        if target_movie_name:
            similar_movies = get_similar_movies(target_movie_name, top_k)
            return similar_movies
        else:
            raise ValueError("The 'similar_movies' method requires a target movie name.")
    else:
        raise ValueError("Invalid method. Expected 'user_based', 'item_based', or 'similar_movies'.")

    return recommended_items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate movie recommendations.")
    parser.add_argument("method", type=str, help="The method to use for generating recommendations. Can be 'user_based', 'item_based', or 'similar_movies'.")
    parser.add_argument("target_user_id", type=int, help="The ID of the user for whom recommendations are to be generated.")
    parser.add_argument("top_k", type=int, help="The number of recommendations to generate.")
    parser.add_argument("--target_movie_name", type=str, help="The name of the movie for which similar movies are to be found. Only used if method is 'similar_movies'.")

    args = parser.parse_args()

    

    if args.method == "similar_movies":
        similar_movies = main(args.method, args.target_user_id, args.top_k, args.target_movie_name)
        print(f"Movies similar to {args.target_movie_name}:")
        for movie in similar_movies:
            print(movie)
    else:
        recommended_items = main(args.method, args.target_user_id, args.top_k)
        print("Recommended movies:")
        for item_id, movie_title in recommended_items:
            print(f"{movie_title} (ID: {item_id})")