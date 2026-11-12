"""Unit tests for preprocessing and text normalization functions."""

import unittest
import pandas as pd
from preprocess import clean_text, prepare_dataframe, verify_no_data_leakage, split_dataset


class TestPreprocessing(unittest.TestCase):
    """Test suite for data normalization and preprocessing."""

    def test_clean_text_null_and_empty(self):
        """Ensure None, NaN, and whitespace return empty string."""
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")
        self.assertEqual(clean_text(float("nan")), "")

    def test_preserves_casing_and_punctuation(self):
        """Crucial test: casing and exclamation marks must NOT be aggressively stripped."""
        text = "I REALLY HATE THIS! WHY ARE YOU HERE???"
        cleaned = clean_text(text)
        self.assertIn("REALLY HATE", cleaned)
        self.assertIn("!", cleaned)
        self.assertIn("?", cleaned)

    def test_preserves_emojis(self):
        """Emojis carry strong toxicity and sentiment signals and must be preserved."""
        text = "You are pathetic 🤬 🖕 😡"
        cleaned = clean_text(text)
        self.assertIn("🤬", cleaned)
        self.assertIn("🖕", cleaned)
        self.assertIn("😡", cleaned)

    def test_normalizes_urls(self):
        """URLs must be mapped to [URL] token."""
        text = "Check out http://spam-site.com/attack and https://malicious.org"
        cleaned = clean_text(text, normalize_urls=True)
        self.assertNotIn("http://spam-site.com", cleaned)
        self.assertIn("[URL]", cleaned)

    def test_normalizes_mentions(self):
        """User mentions should be normalized to [USER] token."""
        text = "Hey @john_doe you are terrible @jane-doe"
        cleaned = clean_text(text, normalize_mentions=True)
        self.assertNotIn("@john_doe", cleaned)
        self.assertIn("[USER]", cleaned)

    def test_strips_html_entities_and_tags(self):
        """HTML tags and HTML entities should be decoded/cleaned."""
        text = "<p>You are &amp; always will be an idiot.<br></p>"
        cleaned = clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<br>", cleaned)
        self.assertIn("&", cleaned)
        self.assertIn("always will be an idiot.", cleaned)

    def test_prepare_dataframe_deduplication(self):
        """Duplicate comments should be dropped."""
        data = {
            "comment_text": [
                "Hello world",
                "Hello world",
                "Different comment",
                None,
                "   ",
            ],
            "toxic": [0, 0, 1, 0, 0],
            "lang": ["en", "en", "en", "en", "en"],
        }
        df = pd.DataFrame(data)
        cleaned_df = prepare_dataframe(df, text_col="comment_text", label_col="toxic", drop_duplicates=True)
        self.assertEqual(len(cleaned_df), 2)
        self.assertIn("clean_text", cleaned_df.columns)
        self.assertIn("label", cleaned_df.columns)

    def test_leakage_detection(self):
        """Leakage verification should detect common samples."""
        train_texts = ["Comment A", "Comment B", "Comment C"]
        eval_texts = ["Comment B", "Comment D"]
        overlap_count = verify_no_data_leakage(train_texts, eval_texts, split_name="val")
        self.assertEqual(overlap_count, 1)

    def test_split_dataset_no_leakage(self):
        """Split dataset should stratify and remove any leaked rows between train and val/test."""
        data = {
            "clean_text": [f"Unique comment number {i}" for i in range(100)],
            "label": [0] * 80 + [1] * 20,
            "lang": ["en"] * 100,
        }
        df = pd.DataFrame(data)
        train_df, val_df, test_df = split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(val_df), 10)
        self.assertEqual(len(test_df), 10)

        # Confirm 0 overlap
        train_set = set(train_df["clean_text"])
        val_set = set(val_df["clean_text"])
        test_set = set(test_df["clean_text"])
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)


if __name__ == "__main__":
    unittest.main()
