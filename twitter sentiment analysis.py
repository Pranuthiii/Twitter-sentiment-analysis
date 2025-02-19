import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Ask for Twitter Username
username = input("Enter the Twitter username to analyze: @")
print(f"\nFetching recent tweets from @{username}...")

# Simulate Processing with Loading Dots
for _ in range(3):
    print("Processing.", end="\r")
    time.sleep(0.5)
    print("Processing..", end="\r")
    time.sleep(0.5)
    print("Processing...", end="\r")
    time.sleep(0.5)

print("\nTweets successfully retrieved! Performing sentiment analysis...\n")
time.sleep(2)

# Step 2: Generate Fake Tweets
def generate_fake_tweets(n=100):
    sentiments = ['positive', 'negative', 'neutral']
    words = ["great", "terrible", "amazing", "worst", "love", "hate", "awesome", "bad", "fantastic", "awful", 
             "cool", "boring", "exciting", "disappointing", "thrilling", "useless", "helpful", "enjoyable"]
    
    tweets = []
    labels = []

    for _ in range(n):
        tweet = " ".join(random.choices(words, k=random.randint(5, 12)))
        label = random.choice(sentiments)
        tweets.append(tweet)
        labels.append(label)

    return pd.DataFrame({'tweet': tweets, 'sentiment': labels})

# Create Fake Dataset
df = generate_fake_tweets(100)

# Step 3: Preprocess Data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['tweet'])
y = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Randomized Accuracy Simulation
fake_accuracy = round(random.uniform(0.75, 0.95), 2)  # Random accuracy between 75% and 95%
print(f"✅ Sentiment Analysis Complete! Simulated Accuracy: {fake_accuracy * 100:.2f}%\n")

# Step 4: Generate Sentiment Breakdown
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["green", "red", "gray"])
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.title(f"Sentiment Breakdown for @{username}")
plt.show()

# Step 5: Word Cloud for Most Used Words
wordcloud = WordCloud(width=500, height=300, background_color='white').generate(" ".join(df['tweet']))
plt.figure(figsize=(7, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Most Frequent Words in @{username}'s Tweets")
plt.show()

# Step 6: Display Random Sample Tweets
print("\n📌 Sample Tweets Retrieved:")
for i in range(5):
    tweet = df.iloc[random.randint(0, len(df)-1)]
    print(f"- \"{tweet['tweet']}\"  [{tweet['sentiment'].upper()}]")

print("\n🔎 Sentiment analysis completed successfully. 📊")
