# AIPI 531 F23 Final Project - Item Recommendation using LLMs

## Overview

This repository contains the final project for AIPI 531 (Fall 2023), focusing on item recommendation using Language Model-based approaches. The primary objectives of the project are to teach Large Language Models (LLMs) for recommending items through prompt engineering and to compare their performance against a simple baseline recommender. The project leverages prompt engineering approaches outlined in [this reference](https://arxiv.org/pdf/2304.03153.pdf) while encouraging students to explore and improve upon these methods. The comparison is based on offline metrics, with an emphasis on creativity, completeness, and rigorous experimentation.

## Datasets

The primary dataset used in this project is MovieLens_100K. Additionally, the dataset's [SUMMARY & USAGE LICENSE](#summary--usage-license) is provided, and users are encouraged to adhere to the specified conditions.

## References and Tools

- [Hugging Face open source LLMs](https://huggingface.co/blog/llama2)
- [OpenAI LLMs](https://openai.com/chatgpt)
- [Prompt Engineering Guide](https://www.promptingguide.ai)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/pdf/2005.11401.pdf)
- [Langchain](https://python.langchain.com/docs/get_started/introduction)
- [Different ideas for applying LLMs for product recommendations](https://arxiv.org/pdf/2303.14524.pdf) ([Paper 2](https://arxiv.org/pdf/2305.02182.pdf), [Paper 3](https://arxiv.org/pdf/2304.10149.pdf))

## Repository Organization

The repository is organized as follows:

1. **Code**: Contains the implementation of LLM-based item recommenders, prompt engineering approaches, and baseline recommenders.

2. **Datasets**: Includes the MovieLens_100K dataset and documentation on how to incorporate additional datasets.

3. **Experiments**: Provides detailed information on the experiments conducted, including code, configurations, and results.

4. **Results**: Presents the results of the experiments, comparing the performance of LLMs and the baseline recommender based on various metrics.

5. **Docs**: Contains documentation for reproducing the results, running the code, and understanding the prompt engineering approaches.

## How to Reproduce Results

To reproduce the results, follow the steps outlined in the documentation provided in the **Docs** folder. Ensure that you have the necessary dependencies installed, and the datasets are set up appropriately.

## Summary of Findings

A brief summary of the key findings and insights derived from the experiments is provided in the **Results** section. It highlights the performance of LLMs compared to the baseline recommender and discusses any notable observations or improvements.

## Troubleshooting

For any questions or issues related to package installation or other technical difficulties, please contact the Teaching Assistant (TA) mentioned in the troubleshooting section. While the primary responsibility lies with the student, the TA is available to assist in resolving any installation issues.

## MovieLens_100K Dataset Summary & Usage License

The MovieLens_100K dataset was collected by the GroupLens Research Project at the University of Minnesota. The dataset includes 100,000 ratings from 943 users on 1682 movies. Users provided ratings on a scale of 1-5, and each user rated at least 20 movies. Demographic information such as age, gender, occupation, and zip code is also included.

### Usage License Conditions

- Users must not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
- Acknowledge the use of the dataset in publications resulting from the use of the data set.
- Users may not redistribute the data without separate permission.
- The data may not be used for any commercial or revenue-bearing purposes without obtaining permission.

### Citation

To acknowledge the use of the dataset in publications, please cite the following paper:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

For further questions or comments, please contact GroupLens <grouplens-info@cs.umn.edu>.

*Note: This README serves as a guide for organizing the repository and providing essential information. It should be customized based on the actual content and outcomes of the project.*