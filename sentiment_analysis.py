# sentiment_analysis.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Download NLTK stopwords (first time only)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# Load dataset
print("Loading IMDB dataset...")
df = pd.read_csv('data/imdb_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check sentiment distribution
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Text Preprocessing Function
def clean_text(text):
    """
    Clean text by removing special characters, HTML tags, and stopwords
    """
    # Remove HTML tags (IMDB reviews have these)
    text = re.sub(r'<.*?>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Apply cleaning (this might take a minute for 50k reviews)
print("\nCleaning text... (this may take 1-2 minutes)")
start_time = time.time()
df['cleaned_text'] = df['review'].apply(clean_text)
print(f"Cleaning completed in {time.time() - start_time:.2f} seconds")

# Show example of cleaning
print("\nExample of text cleaning:")
print(f"Original: {df['review'].iloc[0][:200]}...")
print(f"Cleaned: {df['cleaned_text'].iloc[0][:200]}...")

# Prepare data
X = df['cleaned_text']
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Convert to binary

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# TF-IDF Vectorization
print("\nVectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,  # Top 5000 most important words
    min_df=5,           # Word must appear in at least 5 documents
    max_df=0.8,         # Ignore words that appear in more than 80% of documents
    ngram_range=(1, 2)  # Use both single words and pairs of words
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Train Logistic Regression Model
print("\nTraining Logistic Regression model...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Predictions
print("\nMaking predictions on test set...")
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"{'='*60}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - IMDB Sentiment Analysis', fontsize=16, pad=20)
plt.ylabel('Actual Sentiment', fontsize=12)
plt.xlabel('Predicted Sentiment', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Feature Importance - Top words for each sentiment
print("\n" + "="*60)
print("TOP PREDICTIVE WORDS")
print("="*60)

# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top positive words
top_positive_indices = coefficients.argsort()[-10:][::-1]
print("\nTop 10 words indicating POSITIVE sentiment:")
for idx in top_positive_indices:
    print(f"  • {feature_names[idx]}: {coefficients[idx]:.3f}")

# Top negative words
top_negative_indices = coefficients.argsort()[:10]
print("\nTop 10 words indicating NEGATIVE sentiment:")
for idx in top_negative_indices:
    print(f"  • {feature_names[idx]}: {coefficients[idx]:.3f}")

# Test with custom movie reviews
print("\n" + "="*60)
print("TESTING WITH CUSTOM MOVIE REVIEWS")
print("="*60)

test_reviews = [
    "This movie was absolutely brilliant! The acting was superb and the plot kept me engaged throughout.",
    "Worst movie I've ever seen. Complete waste of time and money. Terrible acting.",
    "It was okay, nothing special. Some good moments but overall pretty average.",
    "Masterpiece! One of the best films of the decade. Highly recommend!",
    "Boring and predictable. I fell asleep halfway through."
]

for i, review in enumerate(test_reviews, 1):
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    
    print(f"\nReview {i}: '{review[:80]}...'")
    print(f"Prediction: {sentiment} (Confidence: {confidence:.1f}%)")

print("\n" + "="*60)
print("Analysis complete! Check 'confusion_matrix.png' for visualization.")
print("="*60)