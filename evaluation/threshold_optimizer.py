"""
Dynamic Threshold Calibration & Dataset Analysis Engine.
Evaluates dataset partitions to determine empirical risk score distributions,
calculate average safe and malicious risk levels, and derive statistically optimal
Safe and Block thresholds for the security gateway.
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
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


class ThresholdOptimizer:
    """Computes empirical risk distributions and derives optimal gateway thresholds."""

    CALIBRATION_FILE = PROJECT_ROOT / "evaluation" / "threshold_calibration.json"

    def __init__(self, dataset_path: Path = DATASET_JSON):
        self.dataset_path = Path(dataset_path)
        self.preprocessor = PreprocessingPipeline()
        self.rule_detector = RuleDetector()
        self.ml_detector = MLDetector()
        self.llama_detector = LlamaDetector()
        self.scorer = SuspicionScoreAggregator()
        self.blocker = SecurityBlocker()

    def run_calibration(
        self,
        split: str = "all",
        max_samples: Optional[int] = None,
        use_llama: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes pipeline against the dataset and computes statistical threshold parameters.
        
        Args:
            split: 'all', 'test', 'validation', 'train'
            max_samples: Optional limit on total samples tested
            use_llama: Whether to query live Ollama or use the isolated heuristic detector
            progress_callback: Optional callback receiving (current_idx, total_count)
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        if split != "all":
            records = [r for r in all_records if r.get("split") == split]
        else:
            records = all_records

        if max_samples and max_samples < len(records):
            # Balance sample selection between safe and malicious
            safe_subset = [r for r in records if r.get("label") == "SAFE"]
            mal_subset = [r for r in records if r.get("label") == "MALICIOUS"]
            
            # Maintain proportional ratio
            safe_ratio = len(safe_subset) / len(records) if len(records) > 0 else 0.1
            n_safe = max(20, int(max_samples * safe_ratio))
            n_mal = max_samples - n_safe

            records = safe_subset[:n_safe] + mal_subset[:n_mal]

        total_samples = len(records)
        safe_scores: List[float] = []
        malicious_scores: List[float] = []
        
        rule_safe_scores: List[float] = []
        rule_mal_scores: List[float] = []
        ml_safe_scores: List[float] = []
        ml_mal_scores: List[float] = []
        
        category_records: Dict[str, List[float]] = {}
        detailed_samples: List[Dict[str, Any]] = []

        start_time = time.perf_counter()

        for idx, sample in enumerate(records):
            prompt = sample["prompt"]
            label = sample.get("label", "SAFE")
            category = sample.get("attack_category", "unknown")

            # 1. Preprocessing
            prep_res = self.preprocessor.process(prompt)

            # 2. Rule-Based Detection
            rule_res = self.rule_detector.detect(prep_res)

            # 3. ML-Based Detection
            ml_res = self.ml_detector.detect(prep_res)

            # 4. Semantic / Heuristic Llama Detection
            if use_llama:
                llama_res = self.llama_detector.detect(prep_res)
            else:
                llama_res = self.llama_detector._heuristic_fallback(prep_res, 0.1)

            # 5. Score Aggregation
            decision = self.scorer.aggregate(rule_res, ml_res, llama_res)
            risk = decision.final_risk_score

            if label == "SAFE":
                safe_scores.append(risk)
                rule_safe_scores.append(rule_res.score)
                ml_safe_scores.append(ml_res.probability)
            else:
                malicious_scores.append(risk)
                rule_mal_scores.append(rule_res.score)
                ml_mal_scores.append(ml_res.probability)

            if category not in category_records:
                category_records[category] = []
            category_records[category].append(risk)

            # Store a subset of sample details for UI inspector
            if len(detailed_samples) < 200:
                detailed_samples.append({
                    "prompt": prompt[:120] + ("..." if len(prompt) > 120 else ""),
                    "ground_truth": label,
                    "category": category,
                    "risk_score": risk,
                    "rule_score": rule_res.score,
                    "ml_score": ml_res.probability,
                    "decision": decision.decision,
                })

            if progress_callback and (idx % 25 == 0 or idx == total_samples - 1):
                progress_callback(idx + 1, total_samples)

        elapsed_sec = time.perf_counter() - start_time

        # Compute empirical distribution metrics
        safe_arr = np.array(safe_scores) if safe_scores else np.array([0.05])
        mal_arr = np.array(malicious_scores) if malicious_scores else np.array([0.95])

        avg_safe_score = float(np.mean(safe_arr))
        std_safe_score = float(np.std(safe_arr))
        median_safe_score = float(np.median(safe_arr))
        p95_safe_score = float(np.percentile(safe_arr, 95))
        p99_safe_score = float(np.percentile(safe_arr, 99))
        max_safe_score = float(np.max(safe_arr))

        avg_mal_score = float(np.mean(mal_arr))
        std_mal_score = float(np.std(mal_arr))
        median_mal_score = float(np.median(mal_arr))
        p5_mal_score = float(np.percentile(mal_arr, 5))
        p1_mal_score = float(np.percentile(mal_arr, 1))
        min_mal_score = float(np.min(mal_arr))

        # Statistically Calibrated Threshold Recommendations
        # Recommended Safe Threshold: Covers 95-98% of safe prompts without false alarms
        recommended_safe = round(float(np.clip(p95_safe_score + 0.05, 0.15, 0.45)), 2)
        
        # Recommended Block Threshold: Catches >= 95% of malicious prompts
        recommended_block = round(float(np.clip(p5_mal_score - 0.05, max(recommended_safe + 0.15, 0.55), 0.85)), 2)

        if recommended_block <= recommended_safe:
            recommended_block = min(0.85, recommended_safe + 0.20)

        separation_margin = round(avg_mal_score - avg_safe_score, 4)

        # Attack category averages
        category_summary = {}
        for cat, scores in category_records.items():
            arr = np.array(scores)
            category_summary[cat] = {
                "count": len(scores),
                "mean_risk": round(float(np.mean(arr)), 4),
                "std_risk": round(float(np.std(arr)), 4),
                "median_risk": round(float(np.median(arr)), 4),
                "min_risk": round(float(np.min(arr)), 4),
                "max_risk": round(float(np.max(arr)), 4),
            }

        # Threshold Sweep (Sensitivity / ROC Analysis)
        threshold_steps = np.arange(0.05, 1.00, 0.05)
        sweep_data = []

        for thresh in threshold_steps:
            t = round(float(thresh), 2)
            # Binary classification benchmark at single threshold t
            # Predicted Malicious if risk >= t
            tp = int(np.sum(mal_arr >= t))
            fn = int(np.sum(mal_arr < t))
            fp = int(np.sum(safe_arr >= t))
            tn = int(np.sum(safe_arr < t))

            acc = (tp + tn) / (len(safe_arr) + len(mal_arr)) if (len(safe_arr) + len(mal_arr)) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            sweep_data.append({
                "threshold": t,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            })

        results = {
            "evaluated_split": split,
            "total_evaluated": total_samples,
            "safe_count": len(safe_scores),
            "malicious_count": len(malicious_scores),
            "elapsed_seconds": round(elapsed_sec, 2),
            "safe_metrics": {
                "mean": round(avg_safe_score, 4),
                "std": round(std_safe_score, 4),
                "median": round(median_safe_score, 4),
                "p95": round(p95_safe_score, 4),
                "p99": round(p99_safe_score, 4),
                "max": round(max_safe_score, 4),
                "rule_mean": round(float(np.mean(rule_safe_scores)), 4) if rule_safe_scores else 0.0,
                "ml_mean": round(float(np.mean(ml_safe_scores)), 4) if ml_safe_scores else 0.0,
            },
            "malicious_metrics": {
                "mean": round(avg_mal_score, 4),
                "std": round(std_mal_score, 4),
                "median": round(median_mal_score, 4),
                "p5": round(p5_mal_score, 4),
                "p1": round(p1_mal_score, 4),
                "min": round(min_mal_score, 4),
                "rule_mean": round(float(np.mean(rule_mal_scores)), 4) if rule_mal_scores else 0.0,
                "ml_mean": round(float(np.mean(ml_mal_scores)), 4) if ml_mal_scores else 0.0,
            },
            "recommended_thresholds": {
                "safe_threshold": recommended_safe,
                "block_threshold": recommended_block,
                "uncertain_margin": round(recommended_block - recommended_safe, 2),
                "separation_margin": separation_margin,
            },
            "category_summary": category_summary,
            "threshold_sweep": sweep_data,
            "sample_scores": {
                "safe_sample": safe_scores[:500],
                "malicious_sample": malicious_scores[:500],
            },
            "inspected_records": detailed_samples,
        }

        # Cache to calibration JSON file
        try:
            with open(self.CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"[!] Warning: Could not save calibration to {self.CALIBRATION_FILE}: {e}")

        return results

    @classmethod
    def load_cached_calibration(cls) -> Optional[Dict[str, Any]]:
        """Loads previously saved calibration results if available."""
        if cls.CALIBRATION_FILE.exists():
            try:
                with open(cls.CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None


if __name__ == "__main__":
    optimizer = ThresholdOptimizer()
    print("[*] Running calibration on test partition...")
    res = optimizer.run_calibration(split="test", max_samples=1000)
    print("=== THRESHOLD CALIBRATION REPORT ===")
    print(f"Safe Prompts Average Risk Score     : {res['safe_metrics']['mean']:.4f} (+/- {res['safe_metrics']['std']:.4f})")
    print(f"Malicious Prompts Average Risk Score: {res['malicious_metrics']['mean']:.4f} (+/- {res['malicious_metrics']['std']:.4f})")
    print(f"Recommended Safe Threshold          : {res['recommended_thresholds']['safe_threshold']:.2f}")
    print(f"Recommended Block Threshold         : {res['recommended_thresholds']['block_threshold']:.2f}")
    print(f"Separation Margin                   : {res['recommended_thresholds']['separation_margin']:.4f}")
