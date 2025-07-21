# enhanced_app_analysis.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import spacy
from google_play_scraper import app as gp_app, reviews, Sort
from app_store_scraper import AppStore
from transformers.pipelines import pipeline
import torch
from collections import Counter
import warnings
import os

# --- INITIAL SETUP ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def initialize_models():
    """Loads and initializes the spaCy and Transformers models."""
    print("🚀 Initializing models...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        print("Spacy 'en_core_web_sm' model not found. Downloading...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    sentiment_analyzer = pipeline(  # type: ignore
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )
    print("✅ Models loaded successfully!")
    return nlp, sentiment_analyzer


# --- DATA SCRAPING ---


def scrape_google_play_reviews(app_id, review_count):
    """Scrapes reviews for a given app ID from the Google Play Store."""
    print(f"Fetching {review_count} reviews from Google Play for '{app_id}'...")
    try:
        reviews_list, _ = reviews(
            app_id, lang="en", country="us", count=review_count, sort=Sort.NEWEST
        )
        df = pd.DataFrame(reviews_list)
        df = df[["content", "score", "at"]].rename(columns={"content": "review", "score": "rating", "at": "date"})  # type: ignore
        df["source"] = "Google Play"
        print(f"✅ Found {len(df)} reviews.")
        return df
    except Exception as e:
        print(f"❌ Error scraping Google Play: {e}")
        return pd.DataFrame()


def scrape_apple_app_store_reviews(app_name, app_id, country_code, review_count):
    """Scrapes reviews for a given app name/ID from the Apple App Store."""
    print(f"Fetching {review_count} reviews from Apple App Store for '{app_name}'...")
    try:
        # Prioritize searching by app_id if provided, as it's more reliable
        if app_id:
            store = AppStore(country=country_code, app_name=app_name, app_id=app_id)
        else:
            store = AppStore(country=country_code, app_name=app_name)

        store.review(how_many=review_count)
        df = pd.DataFrame(store.reviews)

        if df.empty:
            print(
                f"⚠️ No reviews found for '{app_name}' in the '{country_code}' App Store."
            )
            return pd.DataFrame()

        df = df[["review", "rating", "date"]].copy()
        df["source"] = "App Store"
        print(f"✅ Found {len(df)} reviews.")
        return df
    except Exception as e:
        print(f"❌ Error scraping App Store: {e}")
        print(
            "💡 Tip: The app might have a different name in this country's App Store. Finding the numeric App ID from the app's URL is more reliable."
        )
        return pd.DataFrame()


# --- DATA PROCESSING & ANALYSIS ---


def analyze_sentiment(df, sentiment_analyzer):
    """Performs sentiment analysis on the review text."""
    print("🧠 Performing sentiment analysis...")
    if "review" not in df.columns or df.empty:
        return df

    df["review"] = df["review"].astype(str).fillna("")
    reviews_list = df["review"].tolist()

    results = []
    batch_size = 64
    for i in range(0, len(reviews_list), batch_size):
        batch = reviews_list[i : i + batch_size]
        results.extend(sentiment_analyzer(batch))

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [
        r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results
    ]
    return df


def extract_topics(text, nlp_model):
    """Extracts key noun phrases (topics), filtering for stop words, pronouns, and short words."""
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp_model(text.lower())
    topics = []
    stop_words = nlp_model.Defaults.stop_words

    for chunk in doc.noun_chunks:
        clean_chunk = chunk.text.strip()

        # Check if the root of the chunk is a pronoun or if the chunk is a stop word or too short
        is_pronoun = chunk.root.pos_ == "PRON"
        is_stop_word = clean_chunk in stop_words
        is_short = (
            len(clean_chunk) <= 2
        )  # Increased from 1 to 2 to filter out more noise

        if not is_pronoun and not is_stop_word and not is_short:
            topics.append(clean_chunk)

    return topics


def analyze_topics(df, nlp_model):
    """Analyzes topics for all reviews and adds them to the DataFrame."""
    print("🔍 Extracting topics from reviews...")
    if "review" not in df.columns or df.empty:
        df["topics"] = pd.Series([[] for _ in range(len(df))])
        return df

    df["topics"] = df["review"].apply(lambda x: extract_topics(x, nlp_model))
    return df


# --- VISUALIZATION ---


def create_interactive_dashboard(app_data, comparison_data=None):
    """Creates an interactive Plotly dashboard and saves it as an HTML file."""
    print("📊 Generating interactive dashboard...")
    is_comparison = comparison_data is not None

    rows, cols = 5, (2 if is_comparison else 1)
    subplot_titles = (
        "Sentiment Score & User Rating Distribution",
        "Sentiment Breakdown",
        "Weekly Average Sentiment",
        "Top Positive Topics",
        "Top Negative Topics",
    )
    if is_comparison:
        app1_name, app2_name = app_data["name"], comparison_data["name"]
        subplot_titles = (
            f"Sentiment Score - {app1_name}",
            f"Sentiment Score - {app2_name}",
            f"Sentiment Breakdown - {app1_name}",
            f"Sentiment Breakdown - {app2_name}",
            "Weekly Average Sentiment",
            None,
            f"Top Positive Topics - {app1_name}",
            f"Top Positive Topics - {app2_name}",
            f"Top Negative Topics - {app1_name}",
            f"Top Negative Topics - {app2_name}",
        )

    specs = (
        [
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "domain"}, {"type": "domain"}],
            [{"colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ]
        if is_comparison
        else [
            [{"type": "histogram"}],
            [{"type": "pie"}],
            [{"type": "xy"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
        ]
    )
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.08,
    )

    def plot_app_data(data, fig, col_num):
        df = data["df"]
        if df.empty:
            return

        fig.add_trace(
            go.Histogram(
                x=df["sentiment_score"], name="Sentiment Score", marker_color="#1f77b4"
            ),
            row=1,
            col=col_num,
        )
        if not is_comparison:
            fig.add_trace(
                go.Histogram(
                    x=df["rating"], name="User Rating", marker_color="#ff7f0e"
                ),
                row=1,
                col=1,
            )

        sentiment_counts = df["sentiment_label"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name="Sentiment",
            ),
            row=2,
            col=col_num,
        )

        df_ts = df.set_index("date").resample("W")["sentiment_score"].mean().dropna()
        fig.add_trace(
            go.Scatter(
                x=df_ts.index,
                y=df_ts.values,
                mode="lines+markers",
                name=f"{data['name']} Sentiment Trend",
            ),
            row=3,
            col=1,
        )

        if "topics" in df.columns:
            positive_topics = Counter(
                topic
                for _, row in df[df["sentiment_label"] == "POSITIVE"].iterrows()
                for topic in row["topics"]
            )
            negative_topics = Counter(
                topic
                for _, row in df[df["sentiment_label"] == "NEGATIVE"].iterrows()
                for topic in row["topics"]
            )
            pos_topics_df = pd.DataFrame(positive_topics.most_common(10), columns=["Topic", "Count"]).sort_values(by="Count")  # type: ignore
            neg_topics_df = pd.DataFrame(negative_topics.most_common(10), columns=["Topic", "Count"]).sort_values(by="Count")  # type: ignore
            fig.add_trace(
                go.Bar(
                    y=pos_topics_df["Topic"],
                    x=pos_topics_df["Count"],
                    orientation="h",
                    name="Positive",
                    marker_color="green",
                ),
                row=4,
                col=col_num,
            )
            fig.add_trace(
                go.Bar(
                    y=neg_topics_df["Topic"],
                    x=neg_topics_df["Count"],
                    orientation="h",
                    name="Negative",
                    marker_color="red",
                ),
                row=5,
                col=col_num,
            )

    plot_app_data(app_data, fig, 1)
    if is_comparison:
        plot_app_data(comparison_data, fig, 2)
        fig.update_layout(
            title_text=f"📊 App Analysis: {app_data['name']} vs. {comparison_data['name']}",
            height=1200,
            showlegend=True,
        )
    else:
        fig.update_layout(
            title_text=f"📊 App Analysis: {app_data['name']}",
            height=1600,
            showlegend=True,
        )

    filename = f"analysis_dashboard_{app_data['name'].replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"✅ Dashboard saved to: {filename}")


# --- MAIN EXECUTION ---
def main():
    """Main function to run the interactive analysis workflow."""
    nlp, sentiment_analyzer = initialize_models()

    print("\n--- App Analysis Configuration ---")
    analysis_mode = input("Select analysis mode (1: Single App, 2: Comparison): ")
    app_names = []
    if analysis_mode == "2":
        app_names.append(input("Enter name of App 1: "))
        app_names.append(input("Enter name of App 2: "))
    else:
        app_names.append(input("Enter the name of the app to analyze: "))

    app_data_dict = {}
    for app_name in app_names:
        print(f"\n--- Configuring for '{app_name}' ---")

        source_choice = input(
            f"Choose review source for '{app_name}' (1: Google Play, 2: Apple App Store, 3: Both): "
        )
        review_count = int(input("How many reviews to fetch per source? (e.g., 500): "))
        all_reviews = []
        if source_choice in ["1", "3"]:
            google_app_id = input(
                f"Enter Google Play App ID for '{app_name}' (e.g., 'com.google.android.gm'): "
            )
            all_reviews.append(scrape_google_play_reviews(google_app_id, review_count))
        if source_choice in ["2", "3"]:
            apple_country = input(
                f"Enter Apple App Store country code for '{app_name}' (e.g., 'us', 'ca', 'in'): "
            )
            # --- FIX: Asking for optional but recommended App ID ---
            apple_app_id = input(
                f"Enter Apple App ID for '{app_name}' (optional, but recommended - e.g., '284882215' for Facebook): "
            )
            all_reviews.append(
                scrape_apple_app_store_reviews(
                    app_name, apple_app_id, apple_country, review_count
                )
            )

        if not all_reviews:
            print(f"No sources selected for '{app_name}'. Skipping.")
            continue

        df = pd.concat(all_reviews, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        if df.empty:
            print(f"No reviews found for '{app_name}'. Skipping.")
            continue

        filter_choice = input("Apply advanced date range filter? (y/n): ").lower()
        if filter_choice == "y":
            try:
                start_date_str = input("Enter start date (YYYY-MM-DD): ")
                end_date_str = input("Enter end date (YYYY-MM-DD): ")

                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)

                df["date"] = df["date"].dt.tz_localize(None)
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                print(
                    f"✅ Filtered to {len(df)} reviews from {start_date_str} to {end_date_str}."
                )
            except ValueError:
                print(
                    "❌ Invalid date format. Please use YYYY-MM-DD. Skipping date filter."
                )

        df = analyze_sentiment(df, sentiment_analyzer)
        df = analyze_topics(df, nlp)
        app_data_dict[app_name] = {"name": app_name, "df": df}
        csv_filename = f"{app_name.replace(' ', '_')}_sentiment_analysis.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Detailed data for '{app_name}' saved to {csv_filename}")

    if len(app_data_dict) == 1:
        create_interactive_dashboard(list(app_data_dict.values())[0])
    elif len(app_data_dict) == 2:
        apps = list(app_data_dict.values())
        create_interactive_dashboard(apps[0], apps[1])

    print("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()
