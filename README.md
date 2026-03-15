# 🧠 TruthMate — AI Fake Content Detector

An AI-powered web application that detects fake news articles and fraudulent job postings using a dual-layer verification system.

---

## How It Works

TruthMate combines two independent machine learning models with Groq AI (Llama 3.3 70B) to analyze content and return a transparent, explainable verdict.

- **News Model** — trained on 44,898 articles (Fake.csv + True.csv)
- **Jobs Model** — trained on 17,880 job postings (EMSCAD dataset)
- Both models use **Logistic Regression** with **TF-IDF vectorization** (bigrams, 40K features)
- Class imbalance in job data (17K real vs 866 fake) handled via oversampling
- **Accuracy: 99.5%** on held-out test sets

---

## Features

- 🟢🔴 Fake / Real verdict with confidence score
- 🎯 Trust Score combining ML (40%) + Groq AI (60%)
- 🔦 Suspicious sentence highlighting
- 🚩 Red flags and positive signals breakdown
- 😤 Emotional manipulation and source quality assessment
- 🔗 URL scraping with junk content detection
- 📊 Side-by-side results for news and job inputs

---

## Tech Stack

Python · Scikit-learn · TF-IDF · Logistic Regression · Gradio · Groq API · BeautifulSoup4 · Newspaper3k · Joblib


---

## Known Limitations

- URL scraping may fail for sites with strict bot protection — paste article text directly as a workaround
- Optimized for English-language content only


---
