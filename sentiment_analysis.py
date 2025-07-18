import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from google_play_scraper import app, Sort, reviews
from transformers import pipeline
import torch
import time
from googlesearch import search
from datetime import datetime
import warnings
import os

# Set matplotlib to use non-interactive backend to prevent display issues
matplotlib.use("Agg")  # Use 'Agg' backend which doesn't require a display

# Suppress warnings
warnings.filterwarnings("ignore")

# Load spaCy model for preprocessing
print("Loading spaCy model for preprocessing...")
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("spaCy model loaded successfully!")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Trying to download the model...")
    try:
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("spaCy model downloaded and loaded successfully!")
    except:
        print("Failed to load spaCy model. Exiting.")
        exit(1)

# Initialize sentiment analysis pipeline
print("Loading sentiment analysis model...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading sentiment analysis model: {e}")
    print("Trying to use a smaller model...")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("Alternative model loaded successfully!")
    except:
        print("Failed to load sentiment analysis model. Exiting.")
        exit(1)


def search_apps(query, max_results=5):
    """Search for apps on Google Play"""
    results = []

    print(f"\nSearching for '{query}' on Google Play...")
    try:
        for url in search(f"{query} site:play.google.com", num_results=max_results):
            if "id=" in url:
                app_id = url.split("id=")[-1].split("&")[0]
            else:
                # Handle different URL formats
                app_id = (
                    url.split("/")[-1]
                    if url.endswith("/")
                    else url.split("/")[-1].split("?")[0]
                )

            app_name = app_id.replace("_", " ").replace(".", " ").title()
            results.append({"name": app_name, "id": app_id})
    except Exception as e:
        print(f"Search error: {e}")

    return results


def get_app_details(app_id):
    """Get app details from Google Play"""
    try:
        result = app(app_id)
        return {
            "title": result["title"],
            "description": result["description"],
            "score": result["score"],
            "installs": result["installs"],
            "icon": result["icon"],
            "id": app_id,
        }
    except Exception as e:
        print(f"Error fetching app details: {e}")
        return None


def fetch_reviews(app_id, count=200):
    """Fetch reviews from Google Play"""
    print(f"\nFetching {count} reviews for {app_id}...")
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < count:
        try:
            result, continuation_token = reviews(
                app_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=min(100, count - len(all_reviews)),
                continuation_token=continuation_token,
            )
            all_reviews.extend(result)

            if not continuation_token:
                break

            time.sleep(1)  # Be polite with requests
        except Exception as e:
            print(f"Error fetching reviews: {e}")
            break

    print(f"Retrieved {len(all_reviews)} reviews")
    return all_reviews[:count]


def preprocess_text(text):
    """Clean and preprocess text data using spaCy"""
    if pd.isna(text) or not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Create spaCy doc
    doc = nlp(text)

    # Extract tokens: lemmas, lowercase, no punctuation, no stop words, length > 2
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_stop and len(token.text) > 2
    ]

    return " ".join(tokens)


def analyze_sentiment_batch(texts, batch_size=32):
    """Analyze sentiment in batches"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            results.extend(sentiment_analyzer(batch))
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {e}")
            # Fill with neutral sentiment for failed batches
            results.extend([{"label": "NEUTRAL", "score": 0.5}] * len(batch))
    return results


def main():
    print("\n=== Google Play Review Sentiment Analyzer ===")
    print("Powered by Transformers and Google Play Scraper")

    # App search
    while True:
        query = input("\nEnter app name to search (or 'q' to quit): ").strip()
        if query.lower() == "q":
            return

        if not query:
            print("Please enter a valid search term")
            continue

        search_results = search_apps(query)

        if not search_results:
            print("No apps found. Try a different search term.")
            continue

        print("\nSearch Results:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result['name']} (ID: {result['id']})")

        choice = input(
            "\nEnter the number of the app to analyze (or 'b' to search again): "
        )
        if choice.lower() == "b":
            continue

        try:
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(search_results):
                selected_app = search_results[selected_index]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get app details
    app_details = get_app_details(selected_app["id"])
    if not app_details:
        print("Failed to get app details. Please try another app.")
        return

    print(f"\nSelected App: {app_details['title']}")
    print(f"Average Rating: {app_details['score']} ⭐")
    print(f"Installs: {app_details['installs']}")

    # Fetch reviews
    review_count_input = input(
        "\nHow many reviews to analyze? (default 200, max 1000): "
    ).strip()
    try:
        if review_count_input:
            review_count = int(review_count_input)
            review_count = max(50, min(review_count, 1000))
        else:
            review_count = 200
    except ValueError:
        review_count = 200
        print("Invalid input. Using default value of 200 reviews")

    reviews_data = fetch_reviews(selected_app["id"], review_count)

    if not reviews_data:
        print("No reviews found. Exiting.")
        return

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "review": r.get("content", ""),
                "rating": r.get("score", 0),
                "thumbs_up": r.get("thumbsUpCount", 0),
                "date": r.get("at", datetime.now()),
            }
            for r in reviews_data
        ]
    )

    # Preprocess reviews
    print("\nPreprocessing reviews with spaCy...")
    df["cleaned_review"] = df["review"].apply(preprocess_text)

    # Analyze sentiment
    print("Analyzing sentiment with transformer model...")
    sentiment_results = analyze_sentiment_batch(df["cleaned_review"].tolist())

    # Extract sentiment labels and scores
    df["sentiment_label"] = [result["label"] for result in sentiment_results]
    df["sentiment_score"] = [
        result["score"] if result["label"] == "POSITIVE" else -result["score"]
        for result in sentiment_results
    ]

    # Convert to standard labels
    df["sentiment"] = df["sentiment_label"].apply(
        lambda x: (
            "positive"
            if x == "POSITIVE"
            else "negative" if x == "NEGATIVE" else "neutral"
        )
    )

    # Calculate metrics
    total_reviews = len(df)
    positive = len(df[df["sentiment"] == "positive"])
    negative = len(df[df["sentiment"] == "negative"])
    neutral = len(df[df["sentiment"] == "neutral"])

    # Display results
    print("\n=== Analysis Results ===")
    print(f"App: {app_details['title']}")
    print(f"Total Reviews Analyzed: {total_reviews}")
    print(f"Positive Reviews: {positive} ({positive/total_reviews:.1%})")
    print(f"Negative Reviews: {negative} ({negative/total_reviews:.1%})")
    print(f"Neutral Reviews: {neutral} ({neutral/total_reviews:.1%})")

    # Show sample reviews
    if positive > 0:
        print("\n=== Top Positive Reviews ===")
        for i, review in enumerate(
            df.loc[df["sentiment"] == "positive"]
            .sort_values(by="sentiment_score", ascending=False)
            .head(3)["review"]
        ):
            print(f"{i+1}. {review[:150]}{'...' if len(review) > 150 else ''}")

    if negative > 0:
        print("\n=== Top Negative Reviews ===")
        for i, review in enumerate(
            df.loc[df["sentiment"] == "negative"]
            .sort_values(by="sentiment_score")
            .head(3)["review"]
        ):
            print(f"{i+1}. {review[:150]}{'...' if len(review) > 150 else ''}")

    # Visualization
    try:
        plt.figure(figsize=(14, 10))

        # Sentiment distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=df, x="sentiment", order=["positive", "neutral", "negative"])
        plt.title("Sentiment Distribution")
        plt.xlabel("")

        # Rating distribution
        plt.subplot(2, 2, 2)
        sns.countplot(data=df, x="rating")
        plt.title("Star Rating Distribution")
        plt.xlabel("Stars")

        # Sentiment vs rating
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x="rating", y="sentiment_score")
        plt.title("Sentiment Score by Rating")
        plt.xlabel("Star Rating")
        plt.ylabel("Sentiment Score")

        # Time series of sentiment
        if "date" in df and len(df["date"].unique()) > 1:
            plt.subplot(2, 2, 4)
            try:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                weekly = df.resample("W")["sentiment_score"].mean()
                if len(weekly) > 1:
                    weekly.plot()
                    plt.title("Weekly Average Sentiment")
                    plt.ylabel("Sentiment Score")
                    plt.xlabel("Date")
                df.reset_index(inplace=True)
            except Exception as e:
                print(f"Error creating time series plot: {e}")

        plt.tight_layout()
        plt.suptitle(
            f"Sentiment Analysis for {app_details['title']}", fontsize=16, y=1.02
        )

        # Save the plot as an image
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = (
            f"{app_details['title'].replace(' ', '_')}_sentiment_plot_{timestamp}.png"
        )
        plt.savefig(plot_filename)
        print(f"\nVisualization saved to: {plot_filename}")

        # Try to show the plot if possible
        try:
            plt.show()
        except:
            print(
                "Plot display not supported in this environment. Image file saved instead."
            )

    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        print("Skipping visualization step.")

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{app_details['title'].replace(' ', '_')}_sentiment_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
