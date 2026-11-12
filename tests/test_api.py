"""Integration tests for FastAPI dashboard endpoints."""

import io
import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestDashboardAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_serve_index(self):
        """GET / should serve the HTML dashboard."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"comments.", res.content)
        self.assertIn(b"Is it toxic?", res.content)
        self.assertIn(b"Toxic Comment Classifier", res.content)

    def test_predict_single_toxic(self):
        """POST /predict with toxic text should flag positive."""
        payload = {"text": "You are a complete idiot and a total loser."}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("baseline", data)
        self.assertIn("transformer", data)
        self.assertGreater(data["transformer"]["probability"], 0.5)
        self.assertEqual(data["transformer"]["verdict"], "Toxic")

    def test_predict_single_clean(self):
        """POST /predict with respectful comment should be clean."""
        payload = {"text": "This article provides very thoughtful and well-researched insights. Thank you."}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertLess(data["transformer"]["probability"], 0.5)
        self.assertEqual(data["transformer"]["verdict"], "Clean")

    def test_predict_empty_text(self):
        """POST /predict with empty text should return 400."""
        res = self.client.post("/predict", json={"text": "   "})
        self.assertEqual(res.status_code, 400)

    def test_get_results(self):
        """GET /results should return dynamic metrics without hardcoding."""
        res = self.client.get("/results")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("headline", data)
        self.assertIn("per_language", data)
        self.assertIn("test_table", data)
        self.assertIn("plots", data)
        self.assertIsNotNone(data["headline"]["transformer_f1"])

    def test_predict_batch_csv(self):
        """POST /predict-batch should process uploaded CSV."""
        csv_content = (
            "comment_text\n"
            "This is a respectful discussion.\n"
            "You are a terrible person and an idiot.\n"
        )
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
        res = self.client.post("/predict-batch", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
