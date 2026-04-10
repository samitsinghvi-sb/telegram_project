import torch
import re
from transformers import pipeline

# ── Setup ─────────────────────────────────────────────────────────────────────
# Using zero-shot classification — no training needed
# This lets you fairly compare DistilBERT vs your LR model on same test cases
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"  # lightweight, good for this task
)

LABELS     = ["project requirement", "not a project requirement"]
THRESHOLD  = 0.75   # confidence needed to call it a requirement

# ── Same cleaner as your LR script ───────────────────────────────────────────
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
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text):
    cleaned = clean_text(text)
    result  = classifier(cleaned, LABELS)
    req_score = result['scores'][result['labels'].index("project requirement")]
    pred      = 1 if req_score >= THRESHOLD else 0
    return pred, round(req_score, 4)

# ── Test cases — your 4 categories ───────────────────────────────────────────
test_cases = [
    # (text, true_label, category)

    # ✅ Looks like req, IS req
    ("we need a developer to build a ecommerce website for our startup",    1, "looks_like__is_req"),
    ("looking for someone to create a mobile app for our business",         1, "looks_like__is_req"),
    ("need a nodejs backend developer urgently, dm me",                     1, "looks_like__is_req"),
    ("we want to build a custom CRM, budget is 50k",                        1, "looks_like__is_req"),
    ("hiring a react developer for a 3 month project",                      1, "looks_like__is_req"),
    ("need someone to integrate payment gateway in our existing app",        1, "looks_like__is_req"),
    ("looking for a full stack dev to build our SaaS platform",             1, "looks_like__is_req"),
    ("we need an android app for our logistics company",                    1, "looks_like__is_req"),
    ("need a wordpress developer for a restaurant website",                 1, "looks_like__is_req"),
    ("our company needs a custom erp system",                               1, "looks_like__is_req"),

    # ❌ Looks like req, NOT req
    ("looking for a job as a nodejs developer",                             0, "looks_like__not_req"),
    ("i can build websites, anyone need my services",                       0, "looks_like__not_req"),
    ("need help understanding how react hooks work",                        0, "looks_like__not_req"),
    ("anyone hiring flutter developers remotely",                           0, "looks_like__not_req"),
    ("looking for a mentor who can guide me in nodejs",                     0, "looks_like__not_req"),
    ("need a good laptop recommendation for development",                   0, "looks_like__not_req"),
    ("looking for a react tutorial for beginners",                          0, "looks_like__not_req"),
    ("need advice on which database to use for my project",                 0, "looks_like__not_req"),
    ("looking for feedback on my portfolio website",                        0, "looks_like__not_req"),
    ("i want to learn how to build mobile apps",                            0, "looks_like__not_req"),

    # ⚠️ Doesn't look like req, IS req
    ("is someone capable enough to build an optimised webapp",              1, "hidden__is_req"),
    ("dm me if u do mobile stuff",                                          1, "hidden__is_req"),
    ("our startup needs tech help",                                         1, "hidden__is_req"),
    ("bild us smthing for inventory mgmt asap",                             1, "hidden__is_req"),
    ("we hav a idea and need smone to make it real",                        1, "hidden__is_req"),
    ("anyone free for a quick project this month",                          1, "hidden__is_req"),
    ("tech cofounder needed for our fintech idea",                          1, "hidden__is_req"),
    ("who here does devops stuff, we might have work",                      1, "hidden__is_req"),
    ("got a project, need someone reliable",                                1, "hidden__is_req"),
    ("we have budget, need someone to start asap",                          1, "hidden__is_req"),

    # 💬 Doesn't look like req, NOT req
    ("lol this is hilarious",                                               0, "hidden__not_req"),
    ("good morning everyone",                                               0, "hidden__not_req"),
    ("did anyone watch the IPL match yesterday",                            0, "hidden__not_req"),
    ("which framework do you guys prefer for backend",                      0, "hidden__not_req"),
    ("just deployed my first app, feeling great",                           0, "hidden__not_req"),
    ("hey guys whats up",                                                   0, "hidden__not_req"),
    ("anyone know a good course for system design",                         0, "hidden__not_req"),
    ("this nodejs bug is driving me crazy",                                 0, "hidden__not_req"),
    ("what do you guys think about bun js",                                 0, "hidden__not_req"),
    ("my client keeps changing requirements every day",                     0, "hidden__not_req"),
]

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running predictions...\n")

results_by_category = {}
all_preds, all_labels = [], []
failures = []

for text, true_label, category in test_cases:
    pred, conf = predict(text)
    correct    = pred == true_label

    all_preds.append(pred)
    all_labels.append(true_label)

    if category not in results_by_category:
        results_by_category[category] = {"correct": 0, "total": 0}

    results_by_category[category]["total"]   += 1
    results_by_category[category]["correct"] += int(correct)

    if not correct:
        failures.append((text, true_label, pred, conf))

# ── Category breakdown ────────────────────────────────────────────────────────
category_display = {
    "looks_like__is_req"  : "✅  Looks like req    → IS req     (should catch these)",
    "looks_like__not_req" : "❌  Looks like req    → NOT req    (false positive risk)",
    "hidden__is_req"      : "⚠️   Doesn't look like → IS req     (hardest to catch)",
    "hidden__not_req"     : "💬  Doesn't look like → NOT req    (should ignore these)",
}

print("=" * 65)
print("DISTILBERT — RESULTS BY CATEGORY")
print("=" * 65)

total_correct = 0
total_all     = 0

for cat, display in category_display.items():
    r      = results_by_category.get(cat, {"correct": 0, "total": 0})
    acc    = r["correct"] / r["total"] * 100 if r["total"] else 0
    status = "✓" if acc >= 90 else "~" if acc >= 70 else "✗"
    print(f"\n[{status}] {display}")
    print(f"     {r['correct']}/{r['total']} correct  ({acc:.0f}%)")
    total_correct += r["correct"]
    total_all     += r["total"]

print(f"\n{'='*65}")
print(f"OVERALL ACCURACY : {total_correct}/{total_all} ({total_correct/total_all*100:.1f}%)")

# ── Failures ──────────────────────────────────────────────────────────────────
if failures:
    print(f"\n{'='*65}")
    print("FAILURES")
    print(f"{'='*65}")
    for text, true, pred, conf in failures:
        print(f"\n  Text     : {text}")
        print(f"  Expected : {'Requirement' if true == 1 else 'Not Requirement'}")
        print(f"  Got      : {'Requirement' if pred == 1 else 'Not Requirement'}")
        print(f"  Conf     : {conf:.4f}")