# enhanced_app_analysis.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import sys
import os
import requests
import json
from datetime import datetime

# Add the virtual environment site-packages to the path to ensure spaCy uses the correct installation
venv_site_packages = "/home/parantapsinha/Parantap/Sentiment-Analysis-of-App-Reviews/.analysis/lib/python3.11/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import spacy
from google_play_scraper import app as gp_app, reviews, Sort
from transformers.pipelines import pipeline
import torch
from collections import Counter
import warnings

# --- INITIAL SETUP ---

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def initialize_models():
    """Loads and initializes the spaCy and Transformers models."""
    print("🚀 Initializing models...")

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        print("✅ spaCy model loaded successfully!")
    except OSError as e:
        print(f"❌ Failed to load spaCy model: {e}")
        print("Please install the model manually:")
        print("   python -m spacy download en_core_web_sm")
        return None, None

    sentiment_analyzer = pipeline(  # type: ignore
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
    )
    print("✅ Transformers model loaded successfully!")
    print("✅ All models loaded successfully!")
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
    """Scrapes reviews for a given app ID from the Apple App Store."""
    print(f"Fetching {review_count} reviews from Apple App Store for '{app_name}'...")

    # Method 1: Try using the iTunes RSS feed approach
    try:
        import requests
        import json
        from datetime import datetime

        # Construct the RSS feed URL for app reviews
        rss_url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"

        print(f"Trying iTunes RSS feed approach...")
        response = requests.get(rss_url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check if there are entries in the feed
            if "feed" in data and "entry" in data["feed"]:
                entries = data["feed"]["entry"]

                # The first entry is usually app info, skip it
                if isinstance(entries, list) and len(entries) > 1:
                    entries = entries[1:]  # Skip first entry
                elif not isinstance(entries, list):
                    entries = []

                reviews_list = []
                for entry in entries[:review_count]:  # Limit to requested count
                    try:
                        review_data = {
                            "review": entry.get("content", {}).get("label", ""),
                            "rating": int(entry.get("im:rating", {}).get("label", 0)),
                            "date": entry.get("updated", {}).get("label", pd.NaT),
                        }
                        if review_data["review"]:  # Only add if review has content
                            reviews_list.append(review_data)
                    except (KeyError, ValueError):
                        continue

                if reviews_list:
                    df = pd.DataFrame(reviews_list)
                    df["source"] = "App Store"
                    print(f"✅ Found {len(df)} reviews using RSS feed.")
                    return df
                else:
                    print("⚠️ No valid reviews found in RSS feed.")
            else:
                print("⚠️ No review entries found in RSS feed.")
        else:
            print(f"⚠️ RSS feed returned status code: {response.status_code}")

    except Exception as e:
        print(f"❌ RSS feed approach failed: {e}")

    # Method 2: Try the original app-store-scraper library
    try:
        from app_store_scraper import AppStore

        print("Trying app-store-scraper library...")

        # Try with different parameters
        store = AppStore(country=country_code, app_name=app_name, app_id=int(app_id))
        store.review(how_many=review_count)

        if store.reviews:
            reviews_list = []
            for r in store.reviews:
                review_data = {
                    "review": r.get("review", ""),
                    "rating": r.get("rating", 0),
                    "date": r.get("date", pd.NaT),
                }
                if review_data["review"]:  # Only add if review has content
                    reviews_list.append(review_data)

            if reviews_list:
                df = pd.DataFrame(reviews_list)
                df["source"] = "App Store"
                print(f"✅ Found {len(df)} reviews using app-store-scraper.")
                return df

    except Exception as e:
        print(f"❌ app-store-scraper failed: {e}")

    # Method 3: Try using requests with a different endpoint
    try:
        import requests

        print("Trying direct API approach...")

        # Try the lookup API first to verify the app exists
        lookup_url = (
            f"https://itunes.apple.com/lookup?id={app_id}&country={country_code}"
        )
        lookup_response = requests.get(lookup_url, timeout=10)

        if lookup_response.status_code == 200:
            lookup_data = lookup_response.json()
            if lookup_data.get("resultCount", 0) > 0:
                print(
                    f"✅ App found: {lookup_data['results'][0].get('trackName', 'Unknown')}"
                )

                # Unfortunately, the lookup API doesn't return reviews
                # But we can confirm the app exists
                print(
                    "⚠️ Direct review API is not available. Consider using web scraping tools."
                )
            else:
                print(f"⚠️ App with ID {app_id} not found in {country_code} store.")

    except Exception as e:
        print(f"❌ Direct API approach failed: {e}")

    # If all methods fail, return empty DataFrame with a helpful message
    print("\n❌ Unable to fetch App Store reviews. Possible reasons:")
    print("1. The app ID might be incorrect")
    print("2. The app might not be available in the specified country")
    print("3. Apple's API might have changed or be temporarily unavailable")
    print("\nSuggestions:")
    print("1. Verify the app ID is correct (you can find it in the App Store URL)")
    print("2. Try a different country code (e.g., 'us', 'gb', 'ca')")
    print("3. Consider using Google Play reviews if the app is available there")
    print("4. Try installing: pip install itunes-app-scraper")

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
        # Filter out empty reviews
        batch = [r for r in batch if r.strip()]
        if batch:
            results.extend(sentiment_analyzer(batch))

    # Handle cases where some reviews might have been filtered out
    if len(results) < len(reviews_list):
        # Fill in missing results with neutral sentiment
        full_results = []
        result_idx = 0
        for review in reviews_list:
            if review.strip():
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                full_results.append({"label": "NEUTRAL", "score": 0.5})
        results = full_results

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

    # <<< FIX: Adjust layout for single-app view to separate histograms >>>
    if is_comparison:
        rows, cols = 5, 2
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
        specs = [
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "domain"}, {"type": "domain"}],
            [{"colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ]
        fig_height = 1200
    else:
        rows, cols = 6, 1
        subplot_titles = (
            "Sentiment Score Distribution", # New Title
            "User Rating Distribution",     # New Title
            "Sentiment Breakdown",
            "Weekly Average Sentiment",
            "Top Positive Topics",
            "Top Negative Topics",
        )
        specs = [
            [{"type": "histogram"}],
            [{"type": "histogram"}],
            [{"type": "pie"}],
            [{"type": "xy"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
        ]
        fig_height = 1800 # Increased height for the extra plot

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.06, # Adjusted spacing
    )

    def plot_app_data(data, fig, col_num):
        df = data["df"]
        if df.empty:
            return

        # --- Plotting Logic for Single App View ---
        if not is_comparison:
            # <<< FIX: Add Sentiment Score histogram to its own row (row 1) >>>
            # Manually define bins to correctly represent the -1 to 1 score range.
            fig.add_trace(
                go.Histogram(
                    x=df["sentiment_score"],
                    name="Sentiment Score",
                    marker_color="#1f77b4",
                    xbins=dict(start=-1.0, end=1.0, size=0.1) # Corrects binning issue
                ),
                row=1,
                col=1,
            )
            # <<< FIX: Add User Rating histogram to its own row (row 2) >>>
            fig.add_trace(
                go.Histogram(
                    x=df["rating"],
                    name="User Rating",
                    marker_color="#ff7f0e"
                ),
                row=2,
                col=1,
            )
            # Adjust row indices for subsequent plots
            pie_row, scatter_row, pos_topic_row, neg_topic_row = 3, 4, 5, 6
        
        # --- Plotting Logic for Comparison View ---
        else:
            # Original logic for comparison view remains the same
            fig.add_trace(
                go.Histogram(
                    x=df["sentiment_score"],
                    name="Sentiment Score",
                    marker_color="#1f77b4",
                    xbins=dict(start=-1.0, end=1.0, size=0.1)
                ),
                row=1,
                col=col_num,
            )
            pie_row, scatter_row, pos_topic_row, neg_topic_row = 2, 3, 4, 5

        # --- Common plotting logic ---
        sentiment_counts = df["sentiment_label"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name="Sentiment",
            ),
            row=pie_row,
            col=col_num if is_comparison else 1,
        )

        df_ts = df.set_index("date").resample("W")["sentiment_score"].mean().dropna()
        fig.add_trace(
            go.Scatter(
                x=df_ts.index,
                y=df_ts.values,
                mode="lines+markers",
                name=f"{data['name']} Sentiment Trend",
            ),
            row=scatter_row,
            col=1, # This spans both columns in comparison mode
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
            pos_topics_df = pd.DataFrame(positive_topics.most_common(10), columns=["Topic", "Count"]).sort_values(by="Count")
            neg_topics_df = pd.DataFrame(negative_topics.most_common(10), columns=["Topic", "Count"]).sort_values(by="Count")
            fig.add_trace(
                go.Bar(
                    y=pos_topics_df["Topic"],
                    x=pos_topics_df["Count"],
                    orientation="h",
                    name="Positive",
                    marker_color="green",
                ),
                row=pos_topic_row,
                col=col_num if is_comparison else 1,
            )
            fig.add_trace(
                go.Bar(
                    y=neg_topics_df["Topic"],
                    x=neg_topics_df["Count"],
                    orientation="h",
                    name="Negative",
                    marker_color="red",
                ),
                row=neg_topic_row,
                col=col_num if is_comparison else 1,
            )

    plot_app_data(app_data, fig, 1)
    if is_comparison:
        plot_app_data(comparison_data, fig, 2)
        fig.update_layout(
            title_text=f"📊 App Analysis: {app_data['name']} vs. {comparison_data['name']}",
            height=fig_height,
            showlegend=True,
        )
    else:
        fig.update_layout(
            title_text=f"📊 App Analysis: {app_data['name']}",
            height=fig_height,
            showlegend=True,
        )

    filename = f"analysis_dashboard_{app_data['name'].replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"✅ Dashboard saved to: {filename}")


# --- MAIN EXECUTION ---
def main():
    """Main function to run the interactive analysis workflow."""
    nlp, sentiment_analyzer = initialize_models()

    # Check if models were loaded successfully
    if nlp is None or sentiment_analyzer is None:
        print("❌ Failed to initialize models. Exiting.")
        return

    print("\n--- App Analysis Configuration ---")
    analysis_mode = input(
        "Select analysis mode (1: Single App, 2: Comparison): "
    ).strip()
    app_names = []
    if analysis_mode == "2":
        app_names.append(input("Enter name of App 1: ").strip())
        app_names.append(input("Enter name of App 2: ").strip())
    else:
        app_names.append(input("Enter the name of the app to analyze: ").strip())

    app_data_dict = {}
    for app_name in app_names:
        print(f"\n--- Configuring for '{app_name}' ---")

        source_choice = input(
            f"Choose review source for '{app_name}' (1: Google Play, 2: Apple App Store, 3: Both): "
        ).strip()
        review_count = int(input("How many reviews to fetch per source? (e.g., 500): "))
        all_reviews = []
        if source_choice in ["1", "3"]:
            google_app_id = input(
                f"Enter Google Play App ID for '{app_name}' (e.g., 'com.google.android.gm'): "
            ).strip()
            all_reviews.append(scrape_google_play_reviews(google_app_id, review_count))
        if source_choice in ["2", "3"]:
            # Clean the country code input to remove extra spaces or quotes
            apple_country = (
                input(
                    f"Enter Apple App Store country code for '{app_name}' (e.g., 'us', 'ca', 'in'): "
                )
                .strip()
                .strip("'\"")
            )
            apple_app_id = input(
                f"Enter Apple App ID for '{app_name}' (optional, but recommended - e.g., '284882215' for Facebook): "
            ).strip()
            all_reviews.append(
                scrape_apple_app_store_reviews(
                    app_name, apple_app_id, apple_country, review_count
                )
            )

        # Check if all_reviews is empty or contains only empty DataFrames
        if not any(not df.empty for df in all_reviews):
            print(
                f"❌ No reviews were fetched for '{app_name}'. Skipping to the next app if available."
            )
            continue

        df = pd.concat([r for r in all_reviews if not r.empty], ignore_index=True)

        # Check if the concatenated DataFrame is empty
        if df.empty:
            print(
                f"❌ No reviews found for '{app_name}' after attempting to fetch. Cannot proceed with analysis for this app."
            )
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        # Add another check after date conversion, as some rows might be dropped
        if df.empty:
            print(f"No valid reviews with dates found for '{app_name}'. Skipping.")
            continue

        filter_choice = (
            input("Apply advanced date range filter? (y/n): ").lower().strip()
        )
        if filter_choice == "y":
            try:
                start_date_str = input("Enter start date (YYYY-MM-DD): ")
                end_date_str = input("Enter end date (YYYY-MM-DD): ")
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)

                # Ensure the 'date' column is timezone-naive for comparison
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                print(
                    f"✅ Filtered to {len(df)} reviews from {start_date_str} to {end_date_str}."
                )
            except (ValueError, KeyError):
                print(
                    "❌ Invalid date format or filtering error. Please use YYYY-MM-DD. Skipping date filter."
                )

        # Final check before analysis
        if df.empty:
            print(
                f"No reviews remain for '{app_name}' after filtering. Skipping analysis."
            )
            continue

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
