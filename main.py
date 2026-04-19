import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Step 1: Load dataset (subset for speed)
# -----------------------------
file_path = "data/training.1600000.processed.noemoticon.csv"
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(file_path, encoding='latin-1', names=columns)

# Use a smaller subset to prevent freezing
df = df.sample(50000, random_state=42)
print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# Step 2: Map sentiment labels
# -----------------------------
def map_sentiment(x):
    if x == 0:
        return 0  # negative
    elif x == 2:
        return 1  # neutral
    else:
        return 2  # positive

df['sentiment'] = df['target'].apply(map_sentiment)

# -----------------------------
# Step 3: Text cleaning (fast version)
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text_fast(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

df['cleaned'] = df['text'].apply(clean_text_fast)
print(df['cleaned'].head())

# -----------------------------
# Step 4: TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']
print("TF-IDF shape:", X.shape)

# -----------------------------
# Step 5: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Step 6: Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Step 8: Save model and vectorizer
# -----------------------------
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")