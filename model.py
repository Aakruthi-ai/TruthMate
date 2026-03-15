import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

print("🚀 TruthMate — Dual Model Training")
print("="*50)

# ---- Text cleaner ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================================================================
# MODEL 1: NEWS
# ================================================================
print("\n📰 Training NEWS model...")

# Load fake and true news — label them manually
fake_df = pd.read_csv("Fake.csv")       # ← change filename if different
true_df = pd.read_csv("True.csv")       # ← change filename if different

fake_df["label"] = 0   # fake = 0
true_df["label"] = 1   # real = 1

# Combine title + text as input
fake_df["text"] = fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)
true_df["text"] = true_df["title"].astype(str) + " " + true_df["text"].astype(str)

news_df = pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]], axis=0)
news_df = news_df.dropna().reset_index(drop=True)
news_df["text"] = news_df["text"].apply(clean_text)

print(f"   Fake news: {len(fake_df):,}  |  Real news: {len(true_df):,}")
print(f"   Total: {len(news_df):,} | Labels: {news_df['label'].value_counts().to_dict()}")

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    news_df["text"], news_df["label"],
    test_size=0.2, random_state=42, stratify=news_df["label"]
)

print("   🔢 Vectorizing news...")
news_vec = TfidfVectorizer(
    stop_words="english", max_features=40000,
    ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95
)
X_train_nv = news_vec.fit_transform(X_train_n)
X_test_nv  = news_vec.transform(X_test_n)

print("   🤖 Training news model...")
news_model = LogisticRegression(
    max_iter=1000, class_weight="balanced",
    C=5, solver="saga", n_jobs=-1
)
news_model.fit(X_train_nv, y_train_n)

y_pred_n = news_model.predict(X_test_nv)
acc_n = round(accuracy_score(y_test_n, y_pred_n) * 100, 2)
print(f"\n   ✅ News Model Accuracy: {acc_n}%")
print(classification_report(y_test_n, y_pred_n, target_names=["Fake", "Real"]))

joblib.dump(news_model, "news_model.pkl")
joblib.dump(news_vec,   "news_vectorizer.pkl")
print("   💾 Saved: news_model.pkl, news_vectorizer.pkl")

# ================================================================
# MODEL 2: JOBS
# ================================================================
print("\n💼 Training JOBS model...")

jobs_df = pd.read_csv("fake_job_postings.csv")   # ← change filename if different

# Combine all text fields
jobs_df["text"] = (
    jobs_df["title"].astype(str) + " " +
    jobs_df["company_profile"].astype(str) + " " +
    jobs_df["description"].astype(str) + " " +
    jobs_df["requirements"].astype(str) + " " +
    jobs_df["benefits"].astype(str)
)

# fraudulent=1 means FAKE job → label=0
# fraudulent=0 means REAL job → label=1
jobs_df["label"] = jobs_df["fraudulent"].apply(lambda x: 0 if int(x) == 1 else 1)
jobs_df = jobs_df[["text", "label"]].dropna().reset_index(drop=True)
jobs_df["text"] = jobs_df["text"].apply(clean_text)

real_jobs = jobs_df[jobs_df["label"] == 1]
fake_jobs = jobs_df[jobs_df["label"] == 0]
print(f"   Real jobs: {len(real_jobs):,}  |  Fake jobs: {len(fake_jobs):,}")

# Oversample fake jobs to match real jobs
fake_jobs_up = resample(fake_jobs, replace=True, n_samples=len(real_jobs), random_state=42)
jobs_balanced = pd.concat([real_jobs, fake_jobs_up]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   After balancing: {jobs_balanced['label'].value_counts().to_dict()}")

X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(
    jobs_balanced["text"], jobs_balanced["label"],
    test_size=0.2, random_state=42, stratify=jobs_balanced["label"]
)

print("   🔢 Vectorizing jobs...")
job_vec = TfidfVectorizer(
    stop_words="english", max_features=20000,
    ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95
)
X_train_jv = job_vec.fit_transform(X_train_j)
X_test_jv  = job_vec.transform(X_test_j)

print("   🤖 Training jobs model...")
job_model = LogisticRegression(
    max_iter=1000, class_weight="balanced",
    C=3, solver="saga", n_jobs=-1
)
job_model.fit(X_train_jv, y_train_j)

y_pred_j = job_model.predict(X_test_jv)
acc_j = round(accuracy_score(y_test_j, y_pred_j) * 100, 2)
print(f"\n   ✅ Jobs Model Accuracy: {acc_j}%")
print(classification_report(y_test_j, y_pred_j, target_names=["Fake", "Real"]))

joblib.dump(job_model, "job_model.pkl")
joblib.dump(job_vec,   "job_vectorizer.pkl")
print("   💾 Saved: job_model.pkl, job_vectorizer.pkl")

print("\n" + "="*50)
print(f"🎯 News Model: {acc_n}%  |  Jobs Model: {acc_j}%")
print("✅ Training complete! Now run app.py")