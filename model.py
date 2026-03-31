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

# ---> UPGRADE: Expanded Whitelist <---
WHITELIST = set([
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", 
    "amazon.com", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "microsoft.com", "nhentai.net", "whatsapp.com",
    "chatgpt.com", "openai.com", "netflix.com", "apple.com", "yahoo.com"
])

def clean_url(url):
    """Removes http, https, and www so the AI doesn't get biased."""
    url = re.sub(r'^https?://', '', str(url))
    url = url.replace('www.', '')
    return url

def extract_features(url):
    """Extracts custom numerical features from the cleaned URL string."""
    return [
        len(url),                 
        url.count('.'),           
        int('@' in url),          
        int('-' in url),          
        int('?' in url),          
        len(url.split('/'))       
    ]

def predict_phishing(url):
    """Hybrid check: Smart Whitelist first, ML second."""
    if not classifier or not vectorizer:
        return "Error", 0.0

    cleaned_url = clean_url(url)
    base_domain = cleaned_url.split('/')[0]

    # ---> UPGRADE: Smart Subdomain Checker <---
    is_whitelisted = False
    
    # 1. Check for an exact match
    if base_domain in WHITELIST or cleaned_url in WHITELIST:
        is_whitelisted = True
    else:
        # 2. Check if it's a subdomain of a trusted site (e.g., gemini.google.com)
        for trusted_domain in WHITELIST:
            if base_domain.endswith("." + trusted_domain):
                is_whitelisted = True
                break

    if is_whitelisted:
        return "legitimate", 100.00

    # THE MACHINE LEARNING CHECK
    tfidf_features = vectorizer.transform([cleaned_url])
    custom_features = np.array([extract_features(cleaned_url)])
    
    combined_features = hstack([tfidf_features, custom_features])
    prediction = classifier.predict(combined_features)[0]
    probabilities = classifier.predict_proba(combined_features)[0]
    
    class_index = np.where(classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index] * 100
    
    return prediction, confidence