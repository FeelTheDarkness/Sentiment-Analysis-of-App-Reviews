import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import transformers as tf
from transformers.pipelines import pipeline
from google_play_scraper import Sort, reviews_all

# --- Initialization ---
# Load a transformer-based sentiment analysis pipeline
# (This will download the model on first run)
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def preprocess_text(text):
    # Basic cleaning; deep cleaning not as necessary for Transformers
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_reviews(package_name, count=None):
    try:
        print(f"Fetching reviews for '{package_name}'...")
        result = reviews_all(
            package_name, lang="en", country="us", sort=Sort.NEWEST, count=count
        )
        print(f"Successfully fetched {len(result)} reviews.")
        return result
    except Exception as e:
        print(f"An error occurred while fetching reviews for '{package_name}': {e}")
        return []


def assign_sentiment(text):
    # The pipeline returns a list of dicts: [{'label': 'POSITIVE', 'score': 0.999...}]
    result = sentiment_pipeline(text)
    if isinstance(result, list):
        label = result[0]["label"].lower()  # 'positive' or 'negative'
        score = result[0]["score"]
    else:
        label = result["label"].lower()
        score = result["score"]
    if label not in ("positive", "negative"):
        label = "neutral"
    # Score is probability for the predicted class
    return pd.Series(
        [
            score if label == "positive" else -score if label == "negative" else 0.0,
            label,
        ]
    )


def main():
    print("\n--- App Review Sentiment Analysis Tool [Transformers] ---")
    print(
        "This tool fetches reviews from the Google Play Store and performs sentiment analysis with a transformer model."
    )
    print("---------------------------------------------------------")

    # --- Step 1: Get User Input ---
    package_name = input(
        "Enter the Google Play app package name (e.g., com.google.android.apps.maps): "
    ).strip()
    if not package_name:
        print("No package name provided. Exiting.")
        return

    count_input = input(
        "Enter the maximum number of reviews to fetch (or press Enter for all): "
    ).strip()
    count = int(count_input) if count_input.isdigit() else None

    # --- Step 2: Fetch reviews ---
    reviews_data = fetch_reviews(package_name, count)
    if not reviews_data:
        print(f"No reviews were found for '{package_name}'. Exiting.")
        return

    df = pd.DataFrame(reviews_data)

    if "content" not in df.columns or "score" not in df.columns:
        print(
            "The fetched review data is missing the expected 'content' or 'score' columns."
        )
        return

    df = df.rename(columns={"content": "review", "score": "rating"})
    df = df.dropna(subset=["review"])
    df = df.loc[df["review"].str.strip() != ""]

    print("\nPreprocessing reviews...")
    df["cleaned_review"] = df["review"].apply(preprocess_text)

    # --- Step 3: Sentiment Analysis ---
    print(
        "Analyzing sentiment with transformer model (this may take a while for large datasets)..."
    )
    # Batch the inference for better performance (default pipeline batch size is 32)
    sentiments = df["cleaned_review"].apply(assign_sentiment)
    df["sentiment_score"] = sentiments.iloc[:, 0]
    df["sentiment"] = sentiments.iloc[:, 1]

    # Normalize label for display (optionally, treat uncertain as 'neutral')
    # If you want a 'neutral' threshold, you could set: if abs(score) < 0.2: neutral
    # For simplicity, transformers mostly return just positive/negative

    # --- Step 4: Display Results ---
    total_reviews = len(df)
    positive_reviews = len(df[df["sentiment"] == "positive"])
    negative_reviews = len(df[df["sentiment"] == "negative"])
    neutral_reviews = total_reviews - positive_reviews - negative_reviews

    print("\n========== Analysis Results ==========")
    print(f"App Package: {package_name}")
    print(f"Total Reviews Analyzed: {total_reviews}")
    print(
        f"Positive Reviews: {positive_reviews} ({positive_reviews/total_reviews:.1%})"
    )
    print(
        f"Negative Reviews: {negative_reviews} ({negative_reviews/total_reviews:.1%})"
    )
    print(
        f"Neutral (low confidence): {neutral_reviews} ({neutral_reviews/total_reviews:.1%})"
    )
    print("======================================")

    # --- Step 5: Visualization ---
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df,
        x="sentiment",
        order=["positive", "neutral", "negative"],
        palette="viridis",
    )
    plt.title(f"Sentiment Analysis for '{package_name}' (Transformers)", fontsize=16)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.show()

    # --- Step 6: Save Results ---
    safe_app_name = re.sub(r"[\W_]+", "_", package_name)
    output_filename = f"{safe_app_name}_transformers_sentiment_analysis.csv"
    try:
        df.to_csv(output_filename, index=False)
        print(f"\nAnalysis complete. Results saved to: {output_filename}")
    except Exception as e:
        print(f"\nCould not save the results to a file. Error: {e}")


if __name__ == "__main__":
    main()
