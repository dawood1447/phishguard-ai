import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib
import os

from model import extract_features, clean_url

def train():
    data_path = os.path.join('dataset', 'urls.csv')
    df = pd.read_csv(data_path)

    df_phishing = df[df['label'] == 'phishing']
    df_legit = df[df['label'] == 'legitimate']

    if len(df_legit) > len(df_phishing):
        df_legit = df_legit.sample(len(df_phishing), random_state=42)
    else:
        df_phishing = df_phishing.sample(len(df_legit), random_state=42)

    df = pd.concat([df_phishing, df_legit]).sample(frac=1, random_state=42).reset_index(drop=True)

    df['cleaned_url'] = df['url'].apply(clean_url)

    X_orig = df['url']
    X_clean = df['cleaned_url']
    y = df['label']

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_clean)

    # Updated to pass both original and cleaned URLs
    X_custom = np.array([extract_features(orig, clean) for orig, clean in zip(X_orig, X_clean)])

    X_combined = hstack([X_tfidf, X_custom])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    train()