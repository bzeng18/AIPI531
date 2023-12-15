import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
def combine_titles(row):
    return ', '.join(row.index[row == 1])
def select_random_movie(row):
    movies = row['Combined Movies'].split(', ')
    if movies:
        target = np.random.choice(movies)
        movies.remove(target)
        updated_movies = ', '.join(movies)
        return pd.Series([updated_movies, len(movies)+1, target])
    return pd.Series([None, 0, None])

client = OpenAI(api_key = "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")
def chat(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt = prompt,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    return response.choices[0].text
def compare_movies_similarity(one_hot_df, target_movie, movie_list_str):
    if target_movie not in one_hot_df.columns:
        raise ValueError(f"Target movie '{target_movie}' not found in DataFrame")
    movie_list = movie_list_str.split(',')

    for movie in movie_list:
        if movie not in one_hot_df.columns:
            raise ValueError(f"Movie '{movie}' not found in DataFrame")
    movie_matrix = one_hot_df.T

    similarity_matrix = cosine_similarity(movie_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=movie_matrix.index, columns=movie_matrix.index)

    similarities = {movie: similarity_df.at[target_movie, movie] for movie in movie_list}

    return similarities
def create_movie_genre_dict(df):
    movie_genre_dict = {}

    for movie_title, row in df.iterrows():

        genres = [genre for genre, value in row.items() if value == 1]
 
        concatenated_genres = ', '.join(genres)

        movie_genre_dict[movie_title] = concatenated_genres

    return movie_genre_dict
