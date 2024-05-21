# Sentiment Analysis on Twitter Data

## Overview

This project aims to perform sentiment analysis on Twitter data using Python. The sentiment analysis is conducted to classify tweets into positive or negative categories based on the sentiment expressed in the text.

## Dataset

The dataset used for this analysis is sourced from Kaggle, containing approximately 1,048,573 entries. Each entry represents a tweet along with its sentiment label. Sentiments are represented as follows:
- **Positive**: Labelled as `1`
- **Negative**: Labelled as `0`

## Preprocessing

- **Cleaning**: The tweets are preprocessed by removing links, converting text to lowercase, removing punctuation, and removing stopwords.
- **Stemming**: Text data is further processed using stemming to reduce words to their root form.

## Model Training

- **Splitting Data**: The dataset is split into training and test sets using a 80-20 ratio.
- **Feature Extraction**: Text data is vectorized using TF-IDF Vectorizer.
- **Model**: Logistic Regression is utilized as the classification model.
- **Evaluation**: The model achieved a training accuracy of 85.9% and a test accuracy of 83.4%.

## Prediction Script

A script is provided to make predictions on new text inputs:

- **Input**: Users can enter their text input.
- **Preprocessing**: The input is preprocessed using the same methods as during training (removing links, punctuation, stopwords, and stemming).
- **Vectorizing**: The preprocessed text is vectorized using the pre-trained TF-IDF vectorizer.
- **Prediction**: The preprocessed and vectorized text is fed into the pre-trained logistic regression model for sentiment prediction.
- **Output**: The predicted sentiment (positive or negative) is displayed.

## Files

- **training.csv**: Original dataset containing Twitter data and sentiment labels.
- **SentimentAnalysis.ipynb**: Jupyter Notebook file used for training the model.
- **test_model.py**: Python script for using the trained model and providing output based on user input.
- **trained_model.sav**: Serialized file containing the trained logistic regression model.
- **vectorizer.sav**: Serialized file containing the TF-IDF vectorizer used for feature extraction.

## Dependencies

- pandas
- nltk
- scikit-learn

## Installation

To run this project, ensure you have the following dependencies installed:

```
pip3 install pandas nltk scikit-learn
```

## Usage

1. **Training the Model**:
   - Run `SentimentAnalysis.ipynb` Jupyter Notebook file first to train the model and save the trained model (`trained_model.sav`) and TF-IDF vectorizer (`vectorizer.sav`).

2. **Making Predictions**:
   - After training the model, you can use `test_model.py` script to make predictions on new text inputs.
   - Follow the instructions provided in the script to interact with the trained model and input your text for sentiment prediction.

## Demo

- **Positive and Negative Comment Output**: ![Screenshot](https://github.com/Shrooooyes/Sentiment-Analysis/assets/112112961/8b3b1dd4-97b5-4172-8471-b3482e1c441f)


## Credits

- **Author**: [Shreyash](https://github.com/Shrooooyes/)
- **Date**: 2024-05-23
