# AIPI 531 F23 Final Project - Item Recommendation using LLMs

## Overview

This repository contains the final project for AIPI 531 (Fall 2023), focusing on item recommendation using Language Model-based approaches. The primary objectives of the project are to teach Large Language Models (LLMs) for recommending items through prompt engineering and to compare their performance against a simple baseline recommender. The project leverages prompt engineering approaches outlined in [this reference](https://arxiv.org/pdf/2304.03153.pdf) while encouraging students to explore and improve upon these methods. The comparison is based on offline metrics, with an emphasis on creativity, completeness, and rigorous experimentation.

## References and Tools

- [Hugging Face open source LLMs](https://huggingface.co/blog/llama2)
- [OpenAI LLMs](https://openai.com/chatgpt)
- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/pdf/2005.11401.pdf)
- [Langchain](https://python.langchain.com/docs/get_started/introduction)
- [Different ideas for applying LLMs for product recommendations](https://arxiv.org/pdf/2303.14524.pdf) ([Paper 2](https://arxiv.org/pdf/2305.02182.pdf), [Paper 3](https://arxiv.org/pdf/2304.10149.pdf))

## Repository Organization

The repository is organized as follows:

1. **Baseline-recommender**: Contains the ideas of some baseline recommenders, content based approaches, with users and items features embedded.

2. **ml-100k**: Includes the MovieLens_100K dataset

3. **LLM-recommender**: Contains three notebooks for different dataset using LLM-based approach and prompt engineering for a recommender system.

4. **utils**: Some useful data preprocessing functions and OpenAI API endpoints.

## Data Preparation

u.data and u.item from the MovieLens_100K dataset are used. The Data is first filtered by the most 300 similar users and then filtered by the most 100 similar movies. Cosine similarity scores have been used to perform those two filters. Then the data is processed into three different dataset
1. **LLM.ipynb**: This notebook includes all ranks of data ignoring the ranks users gave to the movies. The target is randomly selected from the movies a user watched.
2. **LLM_rating.ipynb**: This notebook includes only high ranks of data that users gave scores of 4 or 5 to the movies. The dataset focused predicting users' favorite movies. The target is randomly selected from the movies a user watched.
3. **LLM_rating_latest.ipynb**: This notebook includes only high ranks of data that users gave scores of 4 or 5 to the movies. The dataset focused predicting users' favorite movies. The target is chosen to be the latest movie a user watched.


## Prompt Engineering

We experimented three different types of prompts along with our baseline prediction to evaluate the performace using number of hits as metrics. The prompts are applied on all three datasets. Here are the details:

1. Simple Prompt:
   ```
        Candidate Set (candidate movies): {}. The movies I have watched (watched movies): {}.
        Can you recommend 10 movies from the Candidate Set similar to the movies I've watched (Format: [<- a candidate movie ->]).
   ```
3. Prompt with three-steps which let the LLM think internally about users preferences then make decision:
   ```
        Candidate Set (candidate movies): {}. The movies I have watched (watched movies): {}.
        Step 1: What features are mst important to me when selecting movies(Summarize my preferences briefly)?
        Step 2: You will select the movies(at most 5 movies) that appeal to me the most from the movies I have watched, based on my personal preferences. Do not make recommandation yet.
        Step 3: Can you recommend 10 movies from the Candidate Set similar to the movies I've watched and selected in Step 2(Format: [no. a watched movie :<-a candidate movie ->])?
    ```
5. Prompt with four-steps and input genre for all candidate movies so that LLM has a better idea on how to classify movies:
   ```
        Candidate Set (candidate movies with genre): {}. The movies I have watched (watched movies): {}.
        Step 1: Find out the movie genres for all the movies I watched.
        Step 2: What features and movie genres are most important to me when selecting movies(Summarize my preferences)?
        Step 3: You will select the movies that appeal to me the most from the movies I have watched, based on my personal preferences and genres. Do not make recommandation yet.
        Step 4: Can you recommend 10 movies from the Candidate Set similar to the movies I've watched and selected in Step 3(Format: [no. a watched movie :<-a candidate movie ->])?
    ```
   
## How to Reproduce Results

### Baseline Recommender

In the Baseline Recommender.ipynb notebook, we have implemented a baseline recommender system using several techniques:

**Data Analysis:** We used pandas for data loading and preprocessing, and to perform exploratory data analysis (EDA) to understand the underlying patterns and trends in the data.

**Recommendation Techniques:** The baseline recommender system was built using simple statistical techniques. For user-based recommendations, we calculated the average rating given by each user and recommended movies that the user hasn't seen yet but have high average ratings. For item-based recommendations, we calculated the average rating for each movie and recommended the top-rated movies.

**Similarity Calculation:** To find similar movies, we used a technique called cosine similarity. This technique measures the cosine of the angle between two vectors, giving us a measure of how similar they are.

**Weighted Rating Calculation:** We also used a weighted rating formula to calculate a score for each movie. This score was based on the movie's average rating and the number of ratings it has received. The formula helps to give a more balanced recommendation by considering both the rating and popularity of the movie.

**Evaluation:** We evaluated the performance of our recommender system using RMSE, MAE. These metrics give us a measure of the effectiveness of our recommendations.

To execute the code and run specific tasks, use the following commands:

```bash
python3 utils/baseline_recommender.py user_based 123 10
python3 utils/baseline_recommender.py item_based 123 10
python3 utils/baseline_recommender.py similar_movies 123 10 --target_movie_name "Toy Story (1995)"
```

"
Replace the placeholder values (e.g., 123, 10, "Toy Story (1995)") with the appropriate inputs based on your requirements. Adjust the parameters and options as needed to perform user-based recommendation, item-based recommendation, or find similar movies based on the specified target movie.

Note: Ensure that the code files and dataset are appropriately configured in your working directory before running these commands.



### LLM Recommender

1. Clone the repository: ```git clone git@github.com:bzeng18/AIPI531.git```
2. Open Final Project folder
3. Install the libraries with right version: ```pip install -r requirements.txt```
4. Get an Open AI API Key from the official [website](https://platform.openai.com/api-keys) and replace it with mine in LLM notebooks. 
5. Each LLM notebook represents a specific task with a baseline prediction and 3 different prompt engineering approaches. Open each of them and run all the cells.

## Summary of Findings

|                                  | Randomly selected target | High-ranked data   | High-ranked data and latest target|
| -------------------------------- | -------------------------| ------------------ | --------------------------------- |
|Baseline(10 most popular)         | 10.67                  % | 8                % | 3.67                            % | 
|Simple Prompt                     | 7.33                   % | 14               % | 1.33                            % | 
|Three-steps Prompt                | 11                     % | 10               % | 2.67                            % | 
|Four-steps Prompt with Genre      | 20                     % | 19               % | 19                              % | 

## Conclusion

There are some insights we could gain from the findings:
1. The four-steps prompt with genre outperforms a lot in all three cases which means that LLM-based recommender could be improved by feeding some directions or classification to the LLM.
2. Three-steps prompt has a better performance overall but the differences are not obvious and unstable which indicates that the LLM model does not have a good ability in reasoning without data.
3. Overall, the LLM-based recommender is unstable and changing a few words in prompts would generate a totally different prediction result.

## Limitations 
1. There is a cost of using OpenAI API and it's time consuming to get a response with long inputs. As a result, only 300 users and 100 movies have been tested
2. A more specific prompt structure could be used in each task but in order to make them comparable, we decided to use general prompts.

## Future explorations
1. Other open source models
2. Use a stream of prompts rather than input a single one
3. More offline metrics such as similarity scores could be used

## MovieLens_100K Dataset Summary & Usage License

The primary dataset used in this project is MovieLens_100K. Additionally, the dataset's [resources](https://www.kaggle.com/datasets/fakhrealam0786/movielens-100k-dataset/data) is provided, and users are encouraged to adhere to the specified conditions. The MovieLens_100K dataset was collected by the GroupLens Research Project at the University of Minnesota. The dataset includes 100,000 ratings from 943 users on 1682 movies. Users provided ratings on a scale of 1-5, and each user rated at least 20 movies. Demographic information such as age, gender, occupation, and zip code is also included.

### Usage License Conditions

- Users must not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
- Acknowledge the use of the dataset in publications resulting from the use of the data set.
- Users may not redistribute the data without separate permission.
- The data may not be used for any commercial or revenue-bearing purposes without obtaining permission.

### Citation

[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

[2] Wang, L., & Lim, E.-P. (2023). Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv [Cs.IR]. Retrieved from http://arxiv.org/abs/2304.03153


*Note: This README serves as a guide for organizing the repository and providing essential information. It should be customized based on the actual content and outcomes of the project.*
