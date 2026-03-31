import joblib
import numpy as np
from scipy.sparse import hstack
import re

try:
    classifier = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    classifier = None
    vectorizer = None

WHITELIST = set([
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", 
    "amazon.com", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "microsoft.com", "nhentai.net", "whatsapp.com",
    "chatgpt.com", "openai.com", "netflix.com", "apple.com", "yahoo.com"
])

def clean_url(url):
    url = re.sub(r'^https?://', '', str(url))
    url = url.replace('www.', '')
    return url

def extract_features(original_url, cleaned_url):
    return [
        len(cleaned_url),
        cleaned_url.count('.'),
        int('@' in cleaned_url),
        int('-' in cleaned_url),
        int('?' in cleaned_url),
        len(cleaned_url.split('/')),
        int(str(original_url).startswith("https")) 
    ]

def predict_phishing(url):
    if not classifier or not vectorizer:
        return "Error", 0.0

    cleaned_url = clean_url(url)
    base_domain = cleaned_url.split('/')[0]

    is_whitelisted = False
    
    if base_domain in WHITELIST or cleaned_url in WHITELIST:
        is_whitelisted = True
    else:
        for trusted_domain in WHITELIST:
            if base_domain.endswith("." + trusted_domain):
                is_whitelisted = True
                break

    if is_whitelisted:
        return "legitimate", 100.00

    tfidf_features = vectorizer.transform([cleaned_url])
    custom_features = np.array([extract_features(url, cleaned_url)])
    
    combined_features = hstack([tfidf_features, custom_features])
    prediction = classifier.predict(combined_features)[0]
    probabilities = classifier.predict_proba(combined_features)[0]
    
    class_index = np.where(classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index] * 100
    
    return prediction, confidence