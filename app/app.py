from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

# Load stopwords
STOP_WORDS = set(stopwords.words("english"))

def clean_text(s):
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# Load model (FIXED PATH)
model = pickle.load(open(r"C:\Users\lkart\Desktop\internship\model\fake_news_pipeline.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["news"]
        cleaned = clean_text(text)

        prob = model.predict_proba([cleaned])[0]
        pred = model.predict([cleaned])[0]

        prediction = "Fake News" if pred == 0 else "Real News"
        confidence = round(max(prob) * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)