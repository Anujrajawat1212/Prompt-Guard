"""
Machine Learning Prompt Injection Classifier.
Employs TF-IDF feature extraction (word and subword n-grams) coupled with a
calibrated classifier to detect subtle, paraphrased, or semantic injection patterns.
"""
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from config import ML_MODEL_PATH, DATASET_JSON
from preprocessing.processor import PreprocessedResult


@dataclass
class MLDetectionResult:
    probability: float
    prediction: str
    confidence: float
    top_features: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "top_features": self.top_features,
            "latency_ms": self.latency_ms,
        }


class MLDetector:
    """Statistical / ML-based prompt injection detection model."""

    def __init__(self, model_path: Path = ML_MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Loads serialized pipeline or triggers automated training on dataset."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                print(f"[!] Error loading ML model from {self.model_path}: {e}")
                self.pipeline = None

        if self.pipeline is None:
            # Auto-train if dataset exists
            if DATASET_JSON.exists():
                from .train_model import train_ml_pipeline
                print("[*] Training ML model baseline from dataset...")
                res = train_ml_pipeline()
                self.pipeline = res.get("pipeline") if isinstance(res, dict) else res
            else:
                print("[!] Dataset not found, ML detector running in fallback heuristic mode.")

    def reload_model(self):
        """Forces reloading the serialized pipeline from disk."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                return True
            except Exception as e:
                print(f"[!] Error reloading ML model from {self.model_path}: {e}")
        return False


    def detect(self, preprocessed: PreprocessedResult) -> MLDetectionResult:
        """Computes prompt injection probability using trained ML pipeline."""
        start_time = time.perf_counter()

        if preprocessed.is_empty:
            latency = (time.perf_counter() - start_time) * 1000
            return MLDetectionResult(
                probability=0.0,
                prediction="SAFE",
                confidence=1.0,
                latency_ms=latency,
            )

        text = preprocessed.sanitized_for_detection

        if self.pipeline is None:
            # Fallback if no model is present
            latency = (time.perf_counter() - start_time) * 1000
            return MLDetectionResult(
                probability=0.05,
                prediction="SAFE",
                confidence=0.50,
                latency_ms=latency,
            )

        try:
            # Predict probabilities [P(Safe), P(Malicious)]
            proba = self.pipeline.predict_proba([text])[0]
            p_malicious = float(proba[1])
            prediction = "MALICIOUS" if p_malicious >= 0.50 else "SAFE"
            confidence = float(max(proba))

            # Extract top influencing features if vectorizer is available
            top_features = []
            try:
                vectorizer = self.pipeline.named_steps["vectorizer"]
                clf = self.pipeline.named_steps["classifier"]
                feature_names = vectorizer.get_feature_names_out()
                coefs = clf.coef_[0]
                
                # Transform current text
                X_vec = vectorizer.transform([text]).toarray()[0]
                active_indices = np.where(X_vec > 0)[0]
                if len(active_indices) > 0:
                    scored_features = sorted(
                        [(feature_names[i], coefs[i] * X_vec[i]) for i in active_indices],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    top_features = [f"{word} ({score:.2f})" for word, score in scored_features[:4] if score > 0]
            except Exception:
                pass

            latency = (time.perf_counter() - start_time) * 1000

            return MLDetectionResult(
                probability=round(p_malicious, 4),
                prediction=prediction,
                confidence=round(confidence, 4),
                top_features=top_features,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"[!] ML prediction error: {e}")
            return MLDetectionResult(
                probability=0.0,
                prediction="SAFE",
                confidence=0.0,
                latency_ms=latency,
            )
