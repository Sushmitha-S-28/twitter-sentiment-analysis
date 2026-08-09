# Twitter Sentiment Analysis using NLP and Machine Learning

## Project Overview

This project implements a Twitter Sentiment Analysis system using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes the text of tweets and classifies them into three sentiment categories:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The project uses the TweetEval sentiment dataset, which is accessed directly from Hugging Face using the Python `datasets` library. Therefore, the dataset does not need to be manually downloaded or uploaded.

---

## Objectives

- Analyze sentiment in Twitter data.
- Preprocess and clean tweet text.
- Apply Natural Language Processing techniques.
- Convert text into numerical features using TF-IDF.
- Train a Machine Learning classification model.
- Classify tweets as Positive, Neutral, or Negative.
- Evaluate the performance of the trained model.
- Predict the sentiment of new tweets.

---

## Technologies Used

- Python
- Google Colab
- Natural Language Processing (NLP)
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Hugging Face Datasets
- TF-IDF
- Logistic Regression
- VADER Sentiment Analyzer
---

##  Dataset

The project uses the **TweetEval Sentiment Dataset**.

Dataset source:

https://huggingface.co/datasets/cardiffnlp/tweet_eval

The dataset contains three sentiment classes:

| Label | Sentiment |
|------:|-----------|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The dataset is automatically loaded from Hugging Face using:

```python
from datasets import load_dataset

dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

## workflow

Twitter Dataset
       ↓
Data Loading
       ↓
Data Exploration
       ↓
Text Preprocessing
       ↓
Remove URLs and Mentions
       ↓
Remove Punctuation
       ↓
Stopword Removal
       ↓
Lemmatization
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression
       ↓
Sentiment Prediction
       ↓
Model Evaluation
       ↓
Positive / Neutral / Negative
