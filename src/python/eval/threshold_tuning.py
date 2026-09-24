"""
SPRINT.md Sec 2.4 -- per-label threshold maximising F1, tuned on validation,
applied to test. AP is threshold-free; F1 is not.

The test split is structurally isolated from tuning: tune_threshold() only
ever receives validation arrays -- there is no code path by which test data
can influence the chosen threshold.

Input: a local directory of BinaryRelevanceRunner predictions, pulled from
HDFS via `hdfs dfs -get` (see src/main/scala/multilabel/BinaryRelevanceRunner.scala):
    <predictions_dir>/label_NN/validation/*.parquet
    <predictions_dir>/label_NN/test/*.parquet
each with columns patch_id, role, label_idx, label_name, true_label, probability.

Output: results/thresholds_<tag>.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

N_LABELS = 19


def _load(predictions_dir: str, label_idx: int, role: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(predictions_dir, f"label_{label_idx:02d}", role, "*.parquet"))
    if not files:
        return pd.DataFrame(columns=["patch_id", "true_label", "probability", "label_name"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def tune_threshold(val_true: np.ndarray, val_prob: np.ndarray) -> tuple:
    """F1-maximising threshold on VALIDATION ONLY, found in one vectorized
    O(n log n) pass via precision_recall_curve (not a per-candidate rescan --
    that was O(n * n_unique_probs) and unusable at >100k-row scale).
    Callers must never pass test-split arrays into this function."""
    if len(val_true) == 0 or val_true.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(val_true, val_prob)
    # precision/recall have one extra trailing point (for threshold=inf) that
    # `thresholds` doesn't; drop it so the arrays align.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def tune_and_evaluate(predictions_dir: str) -> dict:
    results = {}
    for i in range(N_LABELS):
        val_df = _load(predictions_dir, i, "validation")
        test_df = _load(predictions_dir, i, "test")
        if val_df.empty or test_df.empty:
            print(f"[skip] label {i}: missing validation or test predictions")
            continue

        label_name = val_df["label_name"].iloc[0]
        threshold, val_f1 = tune_threshold(val_df["true_label"].to_numpy(), val_df["probability"].to_numpy())

        # test data is only ever touched here, strictly after the threshold is fixed
        test_true = test_df["true_label"].to_numpy()
        test_preds = (test_df["probability"].to_numpy() >= threshold).astype(int)
        test_f1 = f1_score(test_true, test_preds, zero_division=0)

        results[label_name] = {
            "threshold": threshold, "val_f1": val_f1, "test_f1": float(test_f1),
            "n_val": len(val_df), "n_test": len(test_df),
        }
        print(f"{label_name:<70s} threshold={threshold:.3f}  val_F1={val_f1:.3f}  test_F1={test_f1:.3f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--tag", required=True, help="output filename tag, e.g. e1_lr_s2")
    args = p.parse_args()

    results = tune_and_evaluate(args.predictions_dir)
    macro_test_f1 = float(np.mean([r["test_f1"] for r in results.values()])) if results else float("nan")
    print(f"\nmacro test F1 across {len(results)} labels: {macro_test_f1:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/thresholds_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump({"per_label": results, "macro_test_f1": macro_test_f1}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
