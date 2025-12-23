# Sentiment Analysis on Movie Reviews (NLP Project)

## 📌 Project Overview
This project implements a sentiment analysis system using Natural Language Processing (NLP) and Machine Learning to classify movie reviews as **positive** or **negative**.

The model is trained on the IMDb Movie Reviews dataset and demonstrates a complete NLP pipeline from raw text preprocessing to model evaluation.

---

## 🎯 Objectives
- Clean and preprocess raw text data
- Convert text into numerical features using TF-IDF
- Train a Logistic Regression classifier
- Evaluate model performance using standard metrics
- Save the trained model for reuse

---

## 📂 Dataset
- **IMDb Movie Reviews Dataset**
- 50,000 reviews (balanced)
- Labels: `positive`, `negative`

---

## 🧠 Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## 🔄 Project Workflow
1. Load and explore dataset
2. Label encoding
3. Text preprocessing (lowercasing, stopword removal, lemmatization)
4. Train-test split
5. TF-IDF vectorization
6. Model training
7. Model evaluation
8. Model saving

---

## 📊 Model Performance
- Accuracy: ~88–90%
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## ▶️ How to Run
```bash
pip install pandas scikit-learn nltk joblib matplotlib
python sentiment_analysis.py
