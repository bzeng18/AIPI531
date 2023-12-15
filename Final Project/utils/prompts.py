from utils import chat
def Zero_Shot_Simple(candidates, movies):
    prompt = """
        Candidate Set (candidate movies): {}. The movies I have watched (watched movies): {}.
        Can you recommend 10 movies from the Candidate Set similar to the movies I've watched (Format: [<- a candidate movie ->]).
    """.format(
        ", ".join(candidates),
        movies,
    )
    return chat(prompt)
def Zero_Shot_Complex(candidates, movies):
    prompt = """
        Candidate Set (candidate movies): {}. The movies I have watched (watched movies): {}.
        Step 1: What features are mst important to me when selecting movies(Summarize my preferences briefly)?
        Step 2: You will select the movies(at most 5 movies) that appeal to me the most from the movies I have watched, based on my personal preferences. Do not make recommandation yet.
        Step 3: Can you recommend 10 movies from the Candidate Set similar to the movies I've watched and selected in Step 2(Format: [no. a watched movie :<-a candidate movie ->])?
    """.format(
        ", ".join(candidates),
        movies,
    )
    return chat(prompt)
def Zero_Shot_Complex_Genre(movies_genre_dict, movies):
    prompt = """
        Candidate Set (candidate movies with genre): {}. The movies I have watched (watched movies): {}.
        Step 1: Find out the movie genres for all the movies I watched.
        Step 2: What features and movie genres are most important to me when selecting movies(Summarize my preferences)?
        Step 3: You will select the movies that appeal to me the most from the movies I have watched, based on my personal preferences and genres. Do not make recommandation yet.
        Step 4: Can you recommend 10 movies from the Candidate Set similar to the movies I've watched and selected in Step 3(Format: [no. a watched movie :<-a candidate movie ->])?
    """.format(
        ", ".join(movies_genre_dict),
        movies,
    )
    return chat(prompt)
