"""
Model training pipeline for Prompt Injection Detection.
Trains a high-performance TF-IDF + Calibrated Logistic Regression model on the
curated and expanded prompt injection benchmark dataset.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config import DATASET_JSON, ML_MODEL_PATH
from preprocessing.processor import PreprocessingPipeline


def train_ml_pipeline(max_train_samples: int = None, c_param: float = 3.0):
    """
    Trains and serializes the TF-IDF + Logistic Regression prompt injection model.
    
    :param max_train_samples: Optional maximum number of training prompts (e.g., 100, 1000).
    :param c_param: Regularization strength for Logistic Regression.
    """
    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_JSON}")

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    preprocessor = PreprocessingPipeline()

    train_records = [d for d in data if d["split"] == "train"]
    val_records = [d for d in data if d["split"] == "validation"]
    test_records = [d for d in data if d["split"] == "test"]

    # If max_train_samples is specified, sample stratifiably
    if max_train_samples and max_train_samples < len(train_records):
        import random
        random.seed(42)
        mal_train = [d for d in train_records if d["numeric_label"] == 1]
        safe_train = [d for d in train_records if d["numeric_label"] == 0]
        
        # Balance according to original ratio
        safe_ratio = len(safe_train) / len(train_records)
        n_safe = max(5, int(max_train_samples * safe_ratio))
        n_mal = max_train_samples - n_safe

        sampled_train = random.sample(safe_train, min(n_safe, len(safe_train))) + \
                        random.sample(mal_train, min(n_mal, len(mal_train)))
        random.shuffle(sampled_train)
        train_records = sampled_train

    print(f"[*] Dataset Split Breakdown:")
    print(f"    - Train Set:      {len(train_records)} records")
    print(f"    - Validation Set: {len(val_records)} records")
    print(f"    - Test Set:       {len(test_records)} records")

    # Preprocess text
    print("[*] Preprocessing dataset prompts...")
    X_train = [preprocessor.process(d["prompt"]).sanitized_for_detection for d in train_records]
    y_train = [d["numeric_label"] for d in train_records]

    X_val = [preprocessor.process(d["prompt"]).sanitized_for_detection for d in val_records]
    y_val = [d["numeric_label"] for d in val_records]

    X_test = [preprocessor.process(d["prompt"]).sanitized_for_detection for d in test_records]
    y_test = [d["numeric_label"] for d in test_records]

    # Define ML Pipeline with expanded vocabulary support
    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                max_features=10000,
                token_pattern=r"(?u)\b\w+\b",
                strip_accents="unicode",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=c_param,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="liblinear",
            ),
        ),
    ])

    print(f"[*] Training pipeline on {len(train_records)} prompt samples...")
    pipeline.fit(X_train, y_train)

    # Evaluate on Validation Set
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    print("\n=======================================================")
    print("             VALIDATION SET PERFORMANCE                ")
    print("=======================================================")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")

    # Evaluate on Held-Out Test Set
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n=======================================================")
    print("             FINAL HELD-OUT TEST SET EVALUATION        ")
    print("=======================================================")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")

    # Save model
    ML_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ML_MODEL_PATH)
    print(f"[+] Successfully saved trained ML model to {ML_MODEL_PATH}")

    return {
        "pipeline": pipeline,
        "train_samples": len(train_records),
        "val_metrics": {"accuracy": val_acc, "precision": val_prec, "recall": val_rec, "f1_score": val_f1},
        "test_metrics": {"accuracy": test_acc, "precision": test_prec, "recall": test_rec, "f1_score": test_f1},
    }


if __name__ == "__main__":
    train_ml_pipeline()

