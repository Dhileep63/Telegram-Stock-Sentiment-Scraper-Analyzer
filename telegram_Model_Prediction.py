# Install necessary libraries (if not already installed)
!pip install pandas
!pip install numpy
!pip install scikit-learn

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('telegram_stock_data.csv')  # Replace with the correct path to your CSV file

# Clean the column names (remove any leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check if 'text' column is present
if 'text' not in df.columns:
    print("Error: 'text' column is missing.")
else:
    # Clean the text data
    def clean_text(text):
        if not text:  # Check if the text is None or empty
            return ""
        text = str(text)  # Ensure text is a string
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\S+|#\S+", "", text)  # Remove mentions and hashtags
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special characters
        return text.lower()

    df['clean_text'] = df['text'].apply(clean_text)

    # Assign dummy labels for classification (e.g., whether the message is about trading or not)
    df['label'] = df['clean_text'].apply(lambda x: 1 if "trading" in x or "Stock" in x else 0)

    # Feature: Process the text data using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']  # Labels for classification

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a machine learning model (Random Forest in this case)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv('cleaned_telegram_stock_data.csv', index=False)

    print("Text classification and evaluation completed successfully.")


'''
Output 
Accuracy: 0.99
Precision: 1.00
Recall: 0.87
F1 Score: 0.93
Text classification and evaluation completed successfully.
'''