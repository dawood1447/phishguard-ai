import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib
import os

from model import extract_features

def train():
    print("Loading massive dataset...")
    data_path = os.path.join('dataset', 'urls.csv')
    df = pd.read_csv(data_path)

    print("--- Class Distribution Before Balancing ---")
    print(df['label'].value_counts())

    # --- YOUR FIX: Balancing the dataset ---
    df_phishing = df[df['label'] == 'phishing']
    df_legit = df[df['label'] == 'legitimate']

    # Downsample the majority class to match the minority class
    if len(df_legit) > len(df_phishing):
        df_legit = df_legit.sample(len(df_phishing), random_state=42)
    else:
        df_phishing = df_phishing.sample(len(df_legit), random_state=42)

    df = pd.concat([df_phishing, df_legit])
    # Shuffle the combined dataset so it's not all phishing then all legit
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n--- Class Distribution After Balancing ---")
    print(df['label'].value_counts())

    X_urls = df['url']
    y = df['label']

    print("\nVectorizing URLs with TF-IDF...")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_urls)

    print("Extracting improved custom features...")
    X_custom = np.array([extract_features(url) for url in X_urls])

    print("Combining features...")
    X_combined = hstack([X_tfidf, X_custom])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    print("Training balanced Logistic Regression model...")
    # YOUR FIX: Increased max_iter and added class_weight
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Evaluating model accuracy...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    print("Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Training complete! Artifacts saved.")

if __name__ == "__main__":
    train()