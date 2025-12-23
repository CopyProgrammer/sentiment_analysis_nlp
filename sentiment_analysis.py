import pandas as pd

# Load dataset
df = pd.read_csv("imdb_reviews.csv")

# Show first five row
print(df.head())

# Show column names
print(df.columns)

# Dataset size
print("Rows:", df.shape[0])
print("Column:", df.shape[1])

# Convert sentiment to numeric labels
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print(df[["sentiment", "label"]].head())

# Check class distribution
print(df["label"].value_counts())

df = df.dropna(subset=["review","label"])
df["review"] = df["review"].astype(str)

import nltk
nltk.download("stopwords")
nltk.download("wordnet")

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub(r"[^a-z\s]", "", text) # remove punctuation/number
    words = text.split() # tokenize
    words = [w for w in words if w not in stop_words] # remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words] # lemmatize
    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

print(df[["review", "clean_review"]].head())

# Features (input)
X = df["clean_review"]

# Target (output)
y = df["label"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Train label distribution:")
print(y_train.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 1)
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

print("Train vector shape:", X_train_vec.shape)
print("Test vector shape:", X_test_vec.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

print("Model training completed!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict sentiment on test set
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

import matplotlib.pyplot as plt

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"


# Test examples
test_sentences = [
    "The movie was absolutely fantastic and enjoyable",
    "I hated this film, it was boring and too long",
    "Not bad, but could have been much better"
]

for sentence in test_sentences:
    print(f"Review: {sentence}")
    print("Predicted Sentiment:", predict_sentiment(sentence))
    print("-" * 50)


import joblib

# Save trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
