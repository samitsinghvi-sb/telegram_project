import json
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ── 1. Load Data ──────────────────────────────────────────────────────────────
texts, labels = [], []

with open("data.jsonl", "r") as f:
    for line in f:
        item = json.loads(line.strip())
        texts.append(item["text"])
        labels.append(item["label"])

print(f"Total samples      : {len(texts)}")
print(f"Requirements  (1)  : {sum(labels)}")
print(f"Non-requirements(0): {labels.count(0)}")

# ── 2. Text Cleaning ──────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower().strip()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize common chat abbreviations
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

    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

texts_clean = [clean_text(t) for t in texts]

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels       # ensures both splits have same class ratio
)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# ── 4. Handle Class Imbalance ─────────────────────────────────────────────────
classes     = np.array([0, 1])
weights     = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"\nClass weights: {class_weight_dict}")

# ── 5. Vectorizer ─────────────────────────────────────────────────────────────
# Word n-grams  : captures phrases like "need developer", "build app"
# Char n-grams  : captures typos like "develper", "bild", "websit"
# Transformer weights: word features weighted 2x more than char features
#                      since semantic meaning matters more than typo correction

word_vect = TfidfVectorizer(
    ngram_range     = (1, 3),     # unigrams + bigrams + trigrams
    min_df          = 1,          # include rare words (small dataset)
    max_df          = 0.95,       # ignore words in 95%+ of docs (too common)
    sublinear_tf    = True,       # log normalization dampens high freq words
    max_features    = 15000,
)

char_vect = TfidfVectorizer(
    analyzer        = 'char_wb',  # char_wb > char: respects word boundaries
    ngram_range     = (2, 5),     # (2,5) catches more typo variations than (3,5)
    min_df          = 1,
    max_df          = 0.95,
    sublinear_tf    = True,
    max_features    = 20000,
)

vectorizer = FeatureUnion([
    ('word', word_vect),
    ('char', char_vect),
], transformer_weights={
    'word': 2.0,                  # semantic meaning > char patterns
    'char': 1.0,
})

# ── 6. Full Pipeline ──────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('features', vectorizer),
    ('clf', LogisticRegression(
        C                = 3.0,   # slightly higher C = less regularization
                                  # works better when data is clean & focused
        class_weight     = class_weight_dict,
        max_iter         = 1000,
        solver           = 'lbfgs',
        random_state     = 42,
    ))
])

# ── 7. Cross Validation (before final fit) ───────────────────────────────────
print("\n── Cross Validation (5-fold) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print(f"F1 per fold : {[round(s, 3) for s in cv_scores]}")
print(f"Mean F1     : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# If std > 0.08, your dataset is too small or too noisy — worth knowing early
if cv_scores.std() > 0.08:
    print("⚠️  High variance across folds — consider collecting more data")

# ── 8. Final Training ─────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── 9. Evaluation ─────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("\n── Test Set Results ──")
print(classification_report(
    y_test, y_pred,
    target_names=['Not Requirement', 'Requirement']
))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                  Predicted: No    Predicted: Yes")
print(f"Actual: No        {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual: Yes       {cm[1][0]:<18} {cm[1][1]}")

# ── 10. Explainability — top words driving each class ────────────────────────
print("\n── Top Features Driving REQUIREMENT prediction ──")
clf         = pipeline.named_steps['clf']
feat_names  = pipeline.named_steps['features'].get_feature_names_out()
coef        = clf.coef_[0]

top_req     = np.argsort(coef)[-20:][::-1]
top_not_req = np.argsort(coef)[:20]

print("\n→ Words pushing toward REQUIREMENT:")
for i in top_req:
    print(f"   {feat_names[i]:<30} {coef[i]:.4f}")

print("\n→ Words pushing toward NOT REQUIREMENT:")
for i in top_not_req:
    print(f"   {feat_names[i]:<30} {coef[i]:.4f}")

# ── 11. Save ──────────────────────────────────────────────────────────────────
pickle.dump(pipeline, open("model.pkl", "wb"))
print("\nModel saved as single pipeline ✅")
print("Load with: pipeline = pickle.load(open('model.pkl', 'rb'))")