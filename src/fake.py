import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from preprocess import clean_text

# Load dataset
fake = pd.read_csv(r"C:\Users\lkart\Desktop\internship\data\Fake.csv")
true = pd.read_csv(r"C:\Users\lkart\Desktop\internship\data\True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true]).reset_index(drop=True)

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open(r"C:\Users\lkart\Desktop\internship\model\fake_news_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully!")