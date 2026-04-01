import string
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(tokens)