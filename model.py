import joblib
import numpy as np
from scipy.sparse import hstack
import re

# Load the saved model and vectorizer
try:
    classifier = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    classifier = None
    vectorizer = None

# --- HYBRID ENGINE: The Whitelist ---
# Enterprise security tools always bypass ML for the most common safe sites.
WHITELIST = set([
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", 
    "amazon.com", "twitter.com", "instagram.com", "linkedin.com",
    "github.com", "microsoft.com", "nhentai.net"
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
    """Hybrid check: Whitelist first, ML second."""
    if not classifier or not vectorizer:
        return "Error", 0.0

    cleaned_url = clean_url(url)

    # 1. THE WHITELIST CHECK (Bypass AI)
    # Extract the base domain (e.g., getting 'google.com' out of 'google.com/search')
    base_domain = cleaned_url.split('/')[0]
    
    if base_domain in WHITELIST or cleaned_url in WHITELIST:
        return "legitimate", 100.00 # 100% confidence it's a globally trusted site

    # 2. THE MACHINE LEARNING CHECK
    # If the site isn't famous, let your trained AI analyze it
    tfidf_features = vectorizer.transform([cleaned_url])
    custom_features = np.array([extract_features(cleaned_url)])
    
    combined_features = hstack([tfidf_features, custom_features])
    prediction = classifier.predict(combined_features)[0]
    probabilities = classifier.predict_proba(combined_features)[0]
    
    class_index = np.where(classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index] * 100
    
    return prediction, confidence