import pickle
import re

# Load single pipeline (contains vectorizer + model both)
pipeline = pickle.load(open("training/model.pkl", "rb"))

# ── Same cleaning function used during training ───────────────────────────────
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    replacements = {
        r'\bu\b'        : 'you',
        r'\bur\b'       : 'your',
        r'\br\b'        : 'are',
        r'\bdev\b'      : 'developer',
        r'\bapps?\b'    : 'app',
        r'\bwebsite\b'  : 'website',
        r'\bweb\s?app\b': 'webapp',
        r'\basap\b'     : 'urgent',
        r'\bbtw\b'      : 'by the way',
        r'\bngl\b'      : '',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    pred    = pipeline.predict([cleaned])[0]
    prob    = pipeline.predict_proba([cleaned])[0][1]
    return pred, prob, cleaned

# ── Test cases ────────────────────────────────────────────────────────────────
test_messages = [
    "need help",
    "we are not looking for a requirement",
    "anyone can engineer a model",
    "can someone build my confidence in system design interviews",
    "anyone hiring flutter developers remotely",
    "looking for feedback on my portfolio website",
    "looking for a mentor who can guide me in nodejs",
    "looking for a job as a nodejs developer",
    "hiring a react developer for a 3 month project",
    "need a dev to migrate our app from php to nodejs",
    "need someone to integrate payment gateway in our existing app",
    "tech cofounder needed for our fintech idea",
    "dm me if u do mobile stuff",
    "small gig available, backend work",
    "need a tech guy for our small business",
    "anyone know a good course for system design",
    "finally understood how promises work lol",
    "i finished the project finally after 2 weeks"
]

