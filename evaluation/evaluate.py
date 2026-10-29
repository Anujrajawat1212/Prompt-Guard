"""
Scientific Evaluation & Benchmarking Suite.
Evaluates the multi-layered detection pipeline against the test split,
computing Precision, Recall, F1, FPR, FNR, latency breakdown, and category metrics.
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_JSON
from preprocessing.processor import PreprocessingPipeline
from detection.rule_detector import RuleDetector
from detection.ml_detector import MLDetector
from detection.llama_detector import LlamaDetector
from detection.scorer import SuspicionScoreAggregator
from prevention.blocker import SecurityBlocker


class BenchmarkEvaluator:
    """Runs automated security benchmarking across the evaluation dataset."""

    def __init__(self, dataset_path: Path = DATASET_JSON):
        self.dataset_path = Path(dataset_path)
        self.preprocessor = PreprocessingPipeline()
        self.rule_detector = RuleDetector()
        self.ml_detector = MLDetector()
        self.llama_detector = LlamaDetector()
        self.scorer = SuspicionScoreAggregator()
        self.blocker = SecurityBlocker()

    def run_evaluation(self, split: str = "test", use_llama: bool = True) -> Dict[str, Any]:
        """Runs evaluation over the specified partition and computes performance metrics."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        if split != "all":
            records = [r for r in all_records if r.get("split") == split]
        else:
            records = all_records

        y_true = []
        y_pred = []
        scores = []
        categories = []
        latencies = {
            "preprocessing": [],
            "rule": [],
            "ml": [],
            "total_detection": [],
        }

        category_stats = {}

        print(f"[*] Commencing evaluation on '{split}' set ({len(records)} samples)...")

        for sample in records:
            prompt = sample["prompt"]
            ground_truth_label = sample["label"]  # "SAFE" or "MALICIOUS"
            numeric_truth = 1 if ground_truth_label == "MALICIOUS" else 0
            category = sample.get("attack_category", "unknown")

            # 1. Preprocessing
            prep_res = self.preprocessor.process(prompt)
            latencies["preprocessing"].append(prep_res.latency_ms)

            # 2. Rule-Based Detection
            rule_res = self.rule_detector.detect(prep_res)
            latencies["rule"].append(rule_res.latency_ms)

            # 3. ML-Based Detection
            ml_res = self.ml_detector.detect(prep_res)
            latencies["ml"].append(ml_res.latency_ms)

            # 4. Llama Semantic Detection (or fallback)
            if use_llama:
                llama_res = self.llama_detector.detect(prep_res)
            else:
                llama_res = self.llama_detector._heuristic_fallback(prep_res, 0.1)

            # 5. Score Aggregation & Prevention Decision
            decision = self.scorer.aggregate(rule_res, ml_res, llama_res)
            prev_result = self.blocker.evaluate(decision)
            latencies["total_detection"].append(decision.total_detection_latency_ms)

            # Predicted classification (1 if blocked/malicious, 0 if safe/allowed)
            predicted_label = 1 if decision.decision == "BLOCKED" else 0

            y_true.append(numeric_truth)
            y_pred.append(predicted_label)
            scores.append(decision.final_risk_score)
            categories.append(category)

            # Track category-level performance
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0, "blocked": 0, "allowed": 0}
            category_stats[category]["total"] += 1
            if predicted_label == numeric_truth:
                category_stats[category]["correct"] += 1
            if predicted_label == 1:
                category_stats[category]["blocked"] += 1
            else:
                category_stats[category]["allowed"] += 1

        # Calculate Confusion Matrix
        # TN, FP, FN, TP
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results = {
            "split_evaluated": split,
            "total_samples": total,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            },
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            },
            "latencies_ms": {
                "avg_preprocessing": round(float(np.mean(latencies["preprocessing"])), 2),
                "avg_rule_detector": round(float(np.mean(latencies["rule"])), 2),
                "avg_ml_detector": round(float(np.mean(latencies["ml"])), 2),
                "avg_total_detection": round(float(np.mean(latencies["total_detection"])), 2),
            },
            "category_performance": category_stats,
        }

        # Save to results.json
        output_file = PROJECT_ROOT / "evaluation" / "results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Prints a clean, formatted report of evaluation metrics."""
    m = results["metrics"]
    cm = results["confusion_matrix"]
    lat = results["latencies_ms"]

    print("\n=======================================================")
    print(f"      BENCHMARK EVALUATION REPORT ({results['split_evaluated'].upper()} SET)")
    print("=======================================================")
    print(f"  Total Samples Evaluated : {results['total_samples']}")
    print(f"  Accuracy                : {m['accuracy'] * 100:.2f}%")
    print(f"  Precision (Security)    : {m['precision'] * 100:.2f}%")
    print(f"  Recall (Detection Rate) : {m['recall'] * 100:.2f}%")
    print(f"  F1-Score                : {m['f1_score'] * 100:.2f}%")
    print(f"  False Positive Rate     : {m['false_positive_rate'] * 100:.2f}%")
    print(f"  False Negative Rate     : {m['false_negative_rate'] * 100:.2f}%")
    print("-------------------------------------------------------")
    print(f"  Confusion Matrix        : TP={cm['true_positive']} | TN={cm['true_negative']} | FP={cm['false_positive']} | FN={cm['false_negative']}")
    print("-------------------------------------------------------")
    print("  Average Latency Breakdown:")
    print(f"    - Preprocessing       : {lat['avg_preprocessing']} ms")
    print(f"    - Rule Detector       : {lat['avg_rule_detector']} ms")
    print(f"    - ML Detector         : {lat['avg_ml_detector']} ms")
    print(f"    - Total Detection     : {lat['avg_total_detection']} ms")
    print("=======================================================\n")


if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    res = evaluator.run_evaluation(split="test")
    print_evaluation_summary(res)
