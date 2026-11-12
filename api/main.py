"""FastAPI backend for Multilingual Toxic Comment Classifier.

Serves single prediction, batch prediction, empirical results,
and the static front-end assets.
"""

from contextlib import asynccontextmanager
import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import existing workspace code
from preprocess import clean_text
from src.baseline import BaselineToxicityClassifier
from src.transformer_model import XLMRobertaToxicityClassifier

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "xlm-roberta-toxic"

baseline_model: Optional[BaselineToxicityClassifier] = None
transformer_model: Optional[XLMRobertaToxicityClassifier] = None
device_name: str = "cpu"


def load_models():
    """Load baseline and XLM-RoBERTa models into memory once at startup."""
    global baseline_model, transformer_model, device_name

    # 1. Load Baseline Model
    if BASELINE_MODEL_PATH.exists():
        logger.info(f"Loading baseline model from {BASELINE_MODEL_PATH}")
        baseline_model = BaselineToxicityClassifier.load(BASELINE_MODEL_PATH)
    else:
        logger.warning(f"Baseline model not found at {BASELINE_MODEL_PATH}")

    # 2. Determine device for XLM-RoBERTa: MPS -> CPU
    if torch.backends.mps.is_available():
        chosen_device = torch.device("mps")
        device_name = "mps"
    else:
        chosen_device = torch.device("cpu")
        device_name = "cpu"

    if TRANSFORMER_MODEL_DIR.exists():
        logger.info(f"Loading XLM-RoBERTa from {TRANSFORMER_MODEL_DIR} on {device_name}")
        try:
            transformer_model = XLMRobertaToxicityClassifier(
                model_name_or_path=TRANSFORMER_MODEL_DIR,
                device=chosen_device,
            )
            transformer_model.load()
        except Exception as e:
            logger.warning(f"Failed loading on {device_name}, attempting CPU fallback: {e}")
            device_name = "cpu"
            transformer_model = XLMRobertaToxicityClassifier(
                model_name_or_path=TRANSFORMER_MODEL_DIR,
                device=torch.device("cpu"),
            )
            transformer_model.load()
    else:
        logger.warning(f"XLM-RoBERTa model directory not found at {TRANSFORMER_MODEL_DIR}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Multilingual Toxic Comment Classifier API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_single(req: PredictRequest) -> Dict[str, Any]:
    """Return toxicity probability and one-word verdict for both models."""
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    cleaned = clean_text(raw_text)
    input_text = cleaned if cleaned else raw_text

    result: Dict[str, Any] = {
        "text": raw_text,
        "cleaned_text": cleaned,
        "baseline": None,
        "transformer": None,
    }

    # Baseline prediction
    if baseline_model is not None:
        try:
            probs = baseline_model.predict_proba([input_text])[0]
            p_toxic = float(probs[1])
            result["baseline"] = {
                "probability": round(p_toxic, 4),
                "verdict": "Toxic" if p_toxic >= 0.5 else "Clean",
            }
        except Exception as e:
            logger.error(f"Baseline prediction error: {e}")
            result["baseline"] = {"error": str(e)}

    # XLM-RoBERTa prediction
    if transformer_model is not None:
        try:
            probs = transformer_model.predict_proba([input_text])[0]
            p_toxic = float(probs[1])
            result["transformer"] = {
                "probability": round(p_toxic, 4),
                "verdict": "Toxic" if p_toxic >= 0.5 else "Clean",
            }
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            result["transformer"] = {"error": str(e)}

    return result


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Classify multiple comments from an uploaded CSV with a comment_text column."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")

    content = await file.read()
    try:
        decoded = content.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Resolve text column
    target_col = None
    for cand in ["comment_text", "text", "comment", "clean_text", "body"]:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain a 'comment_text' column. Found columns: {list(df.columns)}",
        )

    # Process comments (limit to first 500 for responsive web UI)
    df_subset = df.head(500).copy()
    raw_texts = df_subset[target_col].fillna("").astype(str).tolist()
    cleaned_texts = [clean_text(t) or t for t in raw_texts]

    baseline_probs: List[Optional[float]] = []
    transformer_probs: List[Optional[float]] = []

    if baseline_model is not None:
        b_probs = baseline_model.predict_proba(cleaned_texts)[:, 1]
        baseline_probs = [float(p) for p in b_probs]
    else:
        baseline_probs = [None] * len(cleaned_texts)

    if transformer_model is not None:
        t_probs = transformer_model.predict_proba(cleaned_texts)[:, 1]
        transformer_probs = [float(p) for p in t_probs]
    else:
        transformer_probs = [None] * len(cleaned_texts)

    rows = []
    baseline_toxic_count = 0
    transformer_toxic_count = 0

    for idx, raw in enumerate(raw_texts):
        bp = baseline_probs[idx]
        tp = transformer_probs[idx]

        b_verdict = ("Toxic" if bp >= 0.5 else "Clean") if bp is not None else "N/A"
        t_verdict = ("Toxic" if tp >= 0.5 else "Clean") if tp is not None else "N/A"

        if bp is not None and bp >= 0.5:
            baseline_toxic_count += 1
        if tp is not None and tp >= 0.5:
            transformer_toxic_count += 1

        rows.append({
            "index": idx + 1,
            "text": raw,
            "baseline_prob": round(bp, 4) if bp is not None else None,
            "baseline_verdict": b_verdict,
            "transformer_prob": round(tp, 4) if tp is not None else None,
            "transformer_verdict": t_verdict,
        })

    return {
        "total": len(rows),
        "total_in_file": len(df),
        "baseline_toxic_count": baseline_toxic_count,
        "transformer_toxic_count": transformer_toxic_count,
        "rows": rows,
    }


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON from {path}: {e}")
    return None


@app.get("/results")
def get_evaluation_results() -> Dict[str, Any]:
    """Read metrics and plot references from results/test/ and results/validation/."""
    # Test directory
    test_dir = RESULTS_DIR / "test"
    val_dir = RESULTS_DIR / "validation"

    # Fallback to root results if subdirectories do not exist
    if not test_dir.exists():
        test_dir = RESULTS_DIR

    xlm_test_json = _read_json_file(test_dir / "xlm-roberta_metrics.json")
    base_test_json = _read_json_file(test_dir / "baseline_tf-idf_metrics.json")
    base_val_json = _read_json_file(val_dir / "baseline_tf-idf_metrics.json")
    xlm_val_json = _read_json_file(val_dir / "xlm-roberta_metrics.json")

    # If root has xlm-roberta metrics and test didn't
    if xlm_test_json is None and (RESULTS_DIR / "xlm-roberta_metrics.json").exists():
        xlm_test_json = _read_json_file(RESULTS_DIR / "xlm-roberta_metrics.json")
    if base_test_json is None and (RESULTS_DIR / "baseline_tf-idf_metrics.json").exists():
        base_test_json = _read_json_file(RESULTS_DIR / "baseline_tf-idf_metrics.json")

    # Headline values
    xlm_f1 = xlm_test_json.get("f1") if xlm_test_json else None
    base_f1 = base_test_json.get("f1") if base_test_json else None
    xlm_roc = xlm_test_json.get("roc_auc") if xlm_test_json else None

    # Total comments tested & languages count
    total_comments = 0
    if xlm_test_json and "support" in xlm_test_json:
        total_comments = xlm_test_json["support"].get("total", 0)
    elif base_test_json and "support" in base_test_json:
        total_comments = base_test_json["support"].get("total", 0)

    # Per-language comparison chart data
    per_language_map: Dict[str, Dict[str, Any]] = {}

    # Read from CSVs or JSON per_language
    xlm_lang_csv = test_dir / "xlm-roberta_per_language.csv"
    base_lang_csv = test_dir / "baseline_tf-idf_per_language.csv"

    if xlm_lang_csv.exists():
        try:
            df_xlm = pd.read_csv(xlm_lang_csv)
            for _, r in df_xlm.iterrows():
                lang = str(r["lang"]).upper()
                per_language_map.setdefault(lang, {})["lang"] = lang
                per_language_map[lang]["transformer_f1"] = round(float(r["f1"]), 4)
                per_language_map[lang]["sample_count"] = int(r["sample_count"])
        except Exception as e:
            logger.error(f"Error reading {xlm_lang_csv}: {e}")

    if base_lang_csv.exists():
        try:
            df_base = pd.read_csv(base_lang_csv)
            for _, r in df_base.iterrows():
                lang = str(r["lang"]).upper()
                per_language_map.setdefault(lang, {})["lang"] = lang
                per_language_map[lang]["baseline_f1"] = round(float(r["f1"]), 4)
                if "sample_count" not in per_language_map[lang]:
                    per_language_map[lang]["sample_count"] = int(r["sample_count"])
        except Exception as e:
            logger.error(f"Error reading {base_lang_csv}: {e}")

    # Fallback to json per_language if csv not present
    if not per_language_map and xlm_test_json and "per_language" in xlm_test_json:
        for item in xlm_test_json["per_language"]:
            lang = str(item["lang"]).upper()
            per_language_map.setdefault(lang, {})["lang"] = lang
            per_language_map[lang]["transformer_f1"] = round(float(item["f1"]), 4)
            per_language_map[lang]["sample_count"] = int(item["sample_count"])

    if base_test_json and "per_language" in base_test_json:
        for item in base_test_json["per_language"]:
            lang = str(item["lang"]).upper()
            per_language_map.setdefault(lang, {})["lang"] = lang
            per_language_map[lang]["baseline_f1"] = round(float(item["f1"]), 4)

    per_language_list = sorted(
        per_language_map.values(),
        key=lambda x: x.get("sample_count", 0),
        reverse=True,
    )

    languages_count = len(per_language_list) if per_language_list else 6

    # Comparison tables
    def build_summary_row(name: str, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not data:
            return None
        cm = data.get("confusion_matrix", {})
        return {
            "model": name,
            "accuracy": round(float(data.get("accuracy", 0)), 4),
            "precision": round(float(data.get("precision", 0)), 4),
            "recall": round(float(data.get("recall", 0)), 4),
            "f1": round(float(data.get("f1", 0)), 4),
            "macro_f1": round(float(data.get("macro_f1", 0)), 4),
            "roc_auc": round(float(data.get("roc_auc", 0)), 4) if data.get("roc_auc") is not None else None,
            "tn": cm.get("tn"),
            "fp": cm.get("fp"),
            "fn": cm.get("fn"),
            "tp": cm.get("tp"),
        }

    test_rows = []
    if base_test_json:
        test_rows.append(build_summary_row("Baseline (TF-IDF)", base_test_json))
    if xlm_test_json:
        test_rows.append(build_summary_row("XLM-RoBERTa", xlm_test_json))

    val_rows = []
    if base_val_json:
        val_rows.append(build_summary_row("Baseline (TF-IDF)", base_val_json))
    if xlm_val_json:
        val_rows.append(build_summary_row("XLM-RoBERTa", xlm_val_json))

    # Confusion matrix plots
    plots = {
        "test_xlm_cm": "/results-static/test/xlm-roberta_confusion_matrix.png"
        if (test_dir / "xlm-roberta_confusion_matrix.png").exists()
        else None,
        "test_baseline_cm": "/results-static/test/baseline_tf-idf_confusion_matrix.png"
        if (test_dir / "baseline_tf-idf_confusion_matrix.png").exists()
        else None,
        "val_baseline_cm": "/results-static/validation/baseline_tf-idf_confusion_matrix.png"
        if (val_dir / "baseline_tf-idf_confusion_matrix.png").exists()
        else None,
    }

    return {
        "headline": {
            "transformer_f1": round(xlm_f1, 4) if xlm_f1 is not None else None,
            "baseline_f1": round(base_f1, 4) if base_f1 is not None else None,
            "total_comments": total_comments,
            "languages_count": languages_count,
            "roc_auc": round(float(xlm_roc), 4) if xlm_roc is not None else None,
        },
        "per_language": per_language_list,
        "test_table": [r for r in test_rows if r is not None],
        "validation_table": [r for r in val_rows if r is not None],
        "plots": plots,
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Simple dark circle SVG favicon to prevent 404s."""
    svg_data = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="#1d1d1f"/></svg>'
    return Response(content=svg_data, media_type="image/svg+xml")


# Mount results directory to serve confusion matrix plots
if RESULTS_DIR.exists():
    app.mount("/results-static", StaticFiles(directory=str(RESULTS_DIR)), name="results-static")

# Mount static frontend files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)
