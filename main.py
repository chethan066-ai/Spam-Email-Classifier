import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","message"]

# 2. Preprocess
X = df["message"]
y = df["label"].map({"ham":0, "spam":1})

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Predictions
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Test with custom input
sample = ["Congratulations! You've won $1000, click here to claim."]
sample_vec = vectorizer.transform(sample)
print("Custom Prediction:", "Spam" if model.predict(sample_vec)[0] else "Not Spam")