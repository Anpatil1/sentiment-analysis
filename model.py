import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from textblob import TextBlob

import pickle

# Import cleaned datasets
df_train = pd.read_csv('engineering colleges_reviews.csv')
df_test = pd.read_csv('engineering colleges_reviews.csv')

# Assuming the columns are named "college_name," "Review," "Sentiment," and "Keyword"
X_train_text = df_train['Review'] + ' ' + df_train['Keyword']  # Combine 'Review' and 'Keyword'
X_test_text = df_test['Review'] + ' ' + df_test['Keyword']  # Combine 'Review' and 'Keyword'

X_train = df_train[['college_name', 'Review', 'Keyword']]  # Retain 'college_name' for reference
X_test = df_test[['college_name', 'Review', 'Keyword']]  # Retain 'college_name' for reference

y_train = df_train['Sentiment']
y_test = df_test['Sentiment']

# Fit a count vectorizer
vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorizer.fit(X_train_text)
X_train_cv = vectorizer.transform(X_train_text)
X_test_cv = vectorizer.transform(X_test_text)

# Fit a Logistic Regression Model
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train_cv, y_train)

# Get predictions and check accuracies
lr_train_preds = lr.predict(X_train_cv)
lr_preds = lr.predict(X_test_cv)
lr_train_acc = accuracy_score(y_train, lr_train_preds)
lr_test_acc = accuracy_score(y_test, lr_preds)
print("Training Accuracy:", lr_train_acc)
print("Testing Accuracy:", lr_test_acc)

# Pickle the model and vectorizer for use in the app
pickle.dump(lr, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vector.pkl', 'wb'))


# Add TextBlob sentiment analysis function
def analyze_sentiment_with_textblob(text):
    analysis = TextBlob(text)
    # Return the sentiment polarity score
    return analysis.sentiment.polarity


# Example usage:
sample_text = "I love this product! It's amazing."
sentiment_score = analyze_sentiment_with_textblob(sample_text)
print("TextBlob Sentiment Score:", sentiment_score)
