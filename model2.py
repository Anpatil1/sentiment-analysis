# Importing necessary libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sentiment_result.csv')

# Handle missing values in the 'Review' and 'sentiment' columns
df = df.dropna(subset=['Review', 'sentiment'])

# Assuming you have a 'sentiment' column in your dataset
y = df['sentiment']

# Split the dataset into features (X) and target variable (y)
X = df['Review']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the reviews using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectorized, y_train)

# Train Logistic Regression (LR) model
lr_model = LogisticRegression()
lr_model.fit(X_train_vectorized, y_train)

# Save models and vectorizer
with open('svm_model.pkl', 'wb') as svm_model_file:
    pickle.dump(svm_model, svm_model_file)

with open('lr_model.pkl', 'wb') as lr_model_file:
    pickle.dump(lr_model, lr_model_file)

with open('vectorizer.pkl', 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)

# Evaluate SVM model
svm_predictions = svm_model.predict(X_test_vectorized)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Evaluate LR model
lr_predictions = lr_model.predict(X_test_vectorized)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions, average='weighted')
lr_recall = recall_score(y_test, lr_predictions, average='weighted')
lr_f1 = f1_score(y_test, lr_predictions, average='weighted')

# Print evaluation metrics
print("SVM Model Evaluation:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1 Score: {svm_f1:.4f}\n")

print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1 Score: {lr_f1:.4f}")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix for SVM model
plot_confusion_matrix(y_test, svm_predictions, 'SVM Confusion Matrix')

# Plot confusion matrix for Logistic Regression model
plot_confusion_matrix(y_test, lr_predictions, 'Logistic Regression Confusion Matrix')
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

feature_names = vectorizer.get_feature_names_out()
print("Number of Features:", len(feature_names))
print("Sample Features:", feature_names[:10])


print("Actual vs. Predicted for SVM:")
print(pd.DataFrame({'Actual': y_test, 'Predicted': svm_predictions}).head(10))

print("\nActual vs. Predicted for Logistic Regression:")
print(pd.DataFrame({'Actual': y_test, 'Predicted': lr_predictions}).head(10))


print("Training Set Distribution:")
print(y_train.value_counts())

print("\nTesting Set Distribution:")
print(y_test.value_counts())
