import os
import re
import nltk
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from google_play_scraper import app, Sort, reviews_all

# Initialize NLTK
nltk.download(["punkt", "stopwords", "wordnet", "vader_lexicon"])
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()


def get_available_apps():
    """Find all app CSV files in the directory"""
    csv_files = glob("*.csv")
    available_apps = {}

    # Skip analysis results files
    skip_files = ["all_combined.csv", "analyzed_reviews.csv"]

    for i, filename in enumerate([f for f in csv_files if f not in skip_files], 1):
        app_name = filename.replace(".csv", "")
        available_apps[str(i)] = {"name": app_name, "filename": filename}

    return available_apps


def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)


def main():
    print("\nApp Review Sentiment Analysis Tool")
    print("----------------------------------")

    available_apps = get_available_apps()

    if not available_apps:
        print("No app CSV files found in directory.")
        print("Please ensure your CSV files are in the same folder as this script.")
        return

    print("\nAvailable apps for analysis:")
    for num, app in available_apps.items():
        print(f"{num}. {app['name']}")

    while True:
        choice = input("\nEnter the number of the app to analyze (or 'q' to quit): ")
        if choice.lower() == "q":
            return
        if choice in available_apps:
            selected_app = available_apps[choice]
            break
        print("Invalid selection. Please try again.")

    print(f"\nAnalyzing: {selected_app['name']}")

    try:
        df = pd.read_csv(selected_app["filename"])

        # Check for required columns (case insensitive)
        required = {"content", "score"}
        actual = {col.lower() for col in df.columns}

        if not required.issubset(actual):
            missing = required - actual
            print(f"Error: Missing required columns: {missing}")
            return

        # Standardize column names
        df = df.rename(
            columns={
                next(col for col in df.columns if col.lower() == "content"): "review",
                next(col for col in df.columns if col.lower() == "score"): "rating",
            }
        )

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Analysis pipeline
    print("\nPreprocessing reviews...")
    df["cleaned_review"] = df["review"].apply(preprocess_text)

    print("Analyzing sentiment...")
    df["sentiment_score"] = df["cleaned_review"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    df["sentiment"] = df["sentiment_score"].apply(
        lambda x: "positive" if x > 0.1 else ("negative" if x < -0.1 else "neutral")
    )

    # Results
    total = len(df)
    pos = len(df[df["sentiment"] == "positive"])
    neg = len(df[df["sentiment"] == "negative"])

    print("\n=== Results ===")
    print(f"Total reviews: {total}")
    print(f"Positive: {pos} ({pos/total:.1%})")
    print(f"Negative: {neg} ({neg/total:.1%})")
    print(f"Neutral: {total-pos-neg} ({(total-pos-neg)/total:.1%})")

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
    plt.title(f"Sentiment Analysis for {selected_app['name']}")
    plt.show()

    # Save results
    output_name = f"{selected_app['name'].replace(' ', '_')}_analysis.csv"
    df.to_csv(output_name, index=False)
    print(f"\nResults saved to: {output_name}")


if __name__ == "__main__":
    main()
