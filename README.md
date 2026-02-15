# SST-2 Sentiment Benchmarks

Benchmarking multiple NLP approaches for binary sentiment classification
on the Stanford Sentiment Treebank (SST-2), comparing classical
vectorization methods with neural sequence models.

## Motivation

Sentiment analysis is a core NLP task used in finance, product
analytics, and social media monitoring. This project explores how model
complexity affects performance --- from simple bag-of-words
representations to deep learning architectures --- while keeping the
dataset constant.

The goal is not just accuracy, but understanding: - when simple models
are enough - when neural models actually help - tradeoffs between speed,
interpretability, and performance

## Models Implemented

### Classical ML

-   CountVectorizer + Logistic Regression
-   TF-IDF + Logistic Regression
-   Naive Bayes

### Embedding Based

-   Word2Vec + classifier

### Deep Learning

-   LSTM
-   BiLSTM
-   GRU

## Dataset

Stanford Sentiment Treebank (SST-2) Binary movie review sentiment
classification (positive / negative)

## Pipeline

1.  Text cleaning and preprocessing
2.  Tokenization
3.  Feature generation
4.  Model training
5.  Evaluation and comparison

## Evaluation Metrics

-   Accuracy
-   Validation loss
-   Model comparison across architectures

## Project Structure

sst2-sentiment-benchmarks/ │── sst2_sentiment_benchmarks.ipynb │──
requirements.txt │── README.md

## How to Run

1.  Clone the repo
2.  Install dependencies
3.  Launch notebook

## Key Takeaways

-   Classical models perform strongly on small text datasets
-   Neural networks improve contextual understanding but require more
    tuning
-   Model choice depends on data size and latency constraints

## Future Improvements

-   Transformer models (BERT / DistilBERT)
-   Hyperparameter optimization
-   Real-time inference API

## Tech Stack

Python, scikit-learn, TensorFlow/Keras, NLTK, Gensim, Pandas, NumPy
