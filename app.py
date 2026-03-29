from flask import Flask, render_template, request
from model import predict_phishing

app = Flask(__name__)

# Global list to store the last 5 searches
search_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        
        if url:
            # Backend URL Validation
            if not url.startswith("http://") and not url.startswith("https://"):
                return render_template(
                    "index.html", 
                    error="Invalid format. Please include http:// or https://", 
                    url_input=url,
                    history=search_history
                )

            # Get AI Prediction
            prediction, confidence = predict_phishing(url)
            conf_str = f"{confidence:.2f}"
            
            # 🔥 BIG UPGRADE: Explainable AI Logic (Threat Indicators)
            reasons = []
            if prediction == "phishing":
                if "@" in url:
                    reasons.append("Contains '@' symbol (often used to mask real domains).")
                if "-" in url:
                    reasons.append("Contains suspicious hyphen usage (common in fake brand sites).")
                if len(url) > 75:
                    reasons.append("URL is unusually long (used to hide malicious parameters).")
                if "http://" in url and "https://" not in url:
                    reasons.append("Missing HTTPS (connection is unencrypted and insecure).")

            # 🔥 BIG FEATURE: Add to History
            # We store it as a dictionary so the UI can easily read it
            new_entry = {"url": url, "result": prediction, "confidence": conf_str}
            
            # Insert at the beginning of the list (so newest is on top)
            search_history.insert(0, new_entry)
            
            # Keep only the last 5 entries
            if len(search_history) > 5:
                search_history.pop()

            return render_template(
                "index.html", 
                result=prediction, 
                confidence=conf_str,
                url_input=url,
                reasons=reasons,
                history=search_history
            )
            
    # GET Request (First time loading page)
    return render_template("index.html", result=None, history=search_history)

if __name__ == "__main__":
    app.run(debug=True, port=5000)