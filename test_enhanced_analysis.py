import unittest
import pandas as pd
from sentiment_analysis import (
    initialize_models,
    analyze_emotions,
    analyze_topics,
    analyze_aspect_sentiments,
    identify_bug_feature_requests,
)

class TestEnhancedAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize models once for all tests."""
        cls.nlp, cls.sentiment_analyzer, cls.emotion_analyzer, cls.absa_analyzer = initialize_models()
        if not all([cls.nlp, cls.sentiment_analyzer, cls.emotion_analyzer, cls.absa_analyzer]):
            raise unittest.SkipTest("Models could not be initialized.")

    def test_full_pipeline(self):
        """Test the full enhanced analysis pipeline on sample data."""
        # Sample data
        data = {
            "review": [
                "This app is amazing, the interface is so intuitive!",
                "It crashes all the time, a real bug-fest. Please fix this.",
                "I wish it had a dark mode feature.",
                "The customer service is terrible, but the app itself is okay.",
                ""
            ],
            "rating": [5, 1, 3, 2, 3],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
        }
        df = pd.DataFrame(data)

        # Run the analysis functions
        df = analyze_topics(df, self.nlp)
        df = analyze_emotions(df, self.emotion_analyzer)
        df = analyze_aspect_sentiments(df, self.absa_analyzer)
        df = identify_bug_feature_requests(df)

        # --- Assertions ---

        # Check that all new columns are present
        self.assertIn("topics", df.columns)
        self.assertIn("emotion", df.columns)
        self.assertIn("aspect_sentiments", df.columns)
        self.assertIn("category", df.columns)

        # Check topics
        self.assertIsInstance(df["topics"].iloc[0], list)
        self.assertTrue(len(df["topics"].iloc[0]) > 0)  # "amazing app", "intuitive interface"
        self.assertEqual(len(df["topics"].iloc[4]), 0) # Empty review should have no topics

        # Check emotions
        self.assertIsInstance(df["emotion"].iloc[0], str)
        self.assertEqual(df["emotion"].iloc[0], "joy")
        self.assertIn(df["emotion"].iloc[1], ["anger", "sadness"]) # could be either

        # Check aspect sentiments
        self.assertIsInstance(df["aspect_sentiments"].iloc[0], dict)
        self.assertTrue(len(df["aspect_sentiments"].iloc[0]) > 0)
        # Example check: aspect 'interface' should be positive
        self.assertIn('interface', df['aspect_sentiments'].iloc[0])
        self.assertEqual(df['aspect_sentiments'].iloc[0]['interface'], 'Positive')


        # Check categories
        self.assertEqual(df["category"].iloc[0], "review")
        self.assertEqual(df["category"].iloc[1], "bug_report")
        self.assertEqual(df["category"].iloc[2], "feature_request")
        self.assertEqual(df["category"].iloc[3], "review")


if __name__ == "__main__":
    unittest.main()
