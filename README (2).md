# Sentiment Analysis Web App

A deployable NLP inference service that classifies text sentiment using
a trained LSTM model and exposes predictions through both a Flask API
and a browser interface.

Users can enter text and receive real-time Positive or Negative
predictions along with model confidence.

## Features

-   Interactive web interface for live predictions
-   REST API endpoint for programmatic access
-   LSTM deep learning sequence model
-   Tokenization and sequence padding pipeline
-   Confidence scoring
-   JSON response format

## How It Works

1.  User enters text in the webpage or sends a POST request
2.  Text is tokenized using the trained tokenizer
3.  Input is padded to match training sequence length
4.  The LSTM model performs inference
5.  Prediction and confidence score are returned

## API Endpoint

POST /predict

Request: { "text": "This movie was surprisingly good" }

Response: { "input": "This movie was surprisingly good", "prediction":
"Positive", "confidence": 0.9421 }

## Project Structure

sentiment-analysis-webapp/ │── API.py │── index.html │──
sst2_sentiment_models.ipynb

## How to Run

1.  Clone the repository git clone
    https://github.com/YOUR_USERNAME/sentiment-analysis-webapp.git cd
    sentiment-analysis-webapp

2.  Install dependencies pip install -r requirements.txt

3.  Run the application python API.py

4.  Open browser http://127.0.0.1:5000

## Purpose

This project demonstrates building a full ML inference pipeline: model →
preprocessing → API → user interface
