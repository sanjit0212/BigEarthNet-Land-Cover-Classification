"""
Week-1 rewrite (see docs/PROJECT_STATUS.md and the strengthening plan) of the
S1->S2->S3->S4 optimism-gap computation. Fixes two flaws in the original version:

  (A) CLASS-SET MISMATCH: the old macro_ap() silently skipped classes with zero
      test positives, so S2's macro-AP averaged over 19 classes while S4 folds
      averaged over 13-18 (Luxembourg/Kosovo worst). The skipped classes are
      disproportionately the hard rare ones, so the old S4 numbers were biased
      UPWARD and the reported gap was understated. Fixed here via class-major
      aggregation: build a per-class-per-fold AP matrix, average WITHIN each
      class across folds first (skipping only that class's missing cells), then
      macro-average across classes -- so every regime's headline number is a
      mean over the same class set.

  (B) POOLED-AP CONFOUND: pooling raw probabilities from N different classifiers
      (one per S4 fold) into one ranking conflates discrimination loss with
      cross-model score incommensurability. Demoted from headline to a labeled,
      caveated secondary field.

Also adds macro-AUROC (prevalence-invariant) alongside macro-AP, the "S4 trains
on more data" fact, and explicit 18-class-common-set / 19-class-imputed-sensitivity
handling of the Agro-forestry/Portugal structural confound (see build_splits.py's
documented train-coverage violation).

Input: predictions in results/predictions/<name>/label_NN/test/*.parquet
Output: results/optimism_gap.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

N_LABELS = 19
CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]
AGRO_FORESTRY_IDX = CLASSES.index("Agro-forestry areas")
assert AGRO_FORESTRY_IDX == 0

COUNTRIES = ["Finland", "Portugal", "Serbia", "Lithuania", "Ireland",
             "Austria", "Belgium", "Switzerland", "Luxembourg", "Kosovo"]


def _load_run(predictions_root: str, run_dir_name: str) -> dict:
    """Returns {label_idx: (true, prob)} for the 'test' role of one run.
    A missing label directory means either (a) the model was never trained for
    that label (zero train positives -- only Portugal/Agro-forestry in this
    dataset) or (b) the label simply wasn't requested; a present-but-single-class
    label means the fold's test set has zero positives (common for small/
    landlocked countries lacking a class). Both are left for the caller to
    handle via class-major aggregation, not silently dropped here."""
    out = {}
    for i in range(N_LABELS):
        files = glob.glob(os.path.join(predictions_root, run_dir_name, f"label_{i:02d}", "test", "*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out[i] = (df["true_label"].to_numpy(), df["probability"].to_numpy())
    return out


def _safe_ap(t: np.ndarray, p: np.ndarray):
    return float(average_precision_score(t, p)) if t.sum() > 0 else None


def _safe_auroc(t: np.ndarray, p: np.ndarray):
    if t.sum() == 0 or t.sum() == len(t):
        return None  # roc_auc_score undefined with only one class present
    return float(roc_auc_score(t, p))


def class_major_aggregate(fold_datas: list, metric_fn, exclude_classes: set = frozenset()) -> dict:
    """fold_datas: list of per-fold {label_idx: (true, prob)} dicts (one fold for
    S1/S2, five for S3, ten for S4). Averages WITHIN each class across folds
    first (skipping that class's undefined cells), then macro-averages across
    classes -- this is the fix for flaw (A): every regime is compared over the
    same 19 (or 18, with exclude_classes) classes, never a smaller ad-hoc set."""
    per_class_vals = {i: [] for i in range(N_LABELS) if i not in exclude_classes}
    cell_counts = {i: 0 for i in per_class_vals}
    for fold_data in fold_datas:
        for i, (t, p) in fold_data.items():
            if i in exclude_classes:
                continue
            v = metric_fn(t, p)
            if v is not None:
                per_class_vals[i].append(v)
                cell_counts[i] += 1
    per_class_mean = {CLASSES[i]: (float(np.mean(v)) if v else None) for i, v in per_class_vals.items()}
    valid = [v for v in per_class_mean.values() if v is not None]
    macro = float(np.mean(valid)) if valid else float("nan")
    return {"macro": macro, "n_classes_used": len(valid), "per_class_mean": per_class_mean,
            "cell_counts": {CLASSES[i]: c for i, c in cell_counts.items()}}


def pooled_metric(run_datas: list, metric_fn, exclude_classes: set = frozenset()) -> float:
    """CAVEATED secondary statistic -- see module docstring flaw (B). Pooling
    raw probabilities from different classifiers conflates discrimination loss
    with cross-model score incommensurability. Do not headline this."""
    per_label_true, per_label_prob = {}, {}
    for run_data in run_datas:
        for i, (t, p) in run_data.items():
            if i in exclude_classes:
                continue
            per_label_true.setdefault(i, []).append(t)
            per_label_prob.setdefault(i, []).append(p)
    vals = []
    for i in per_label_true:
        t, p = np.concatenate(per_label_true[i]), np.concatenate(per_label_prob[i])
        v = metric_fn(t, p)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def agro_forestry_imputed_ap(portugal_prevalence: float) -> float:
    """Sensitivity analysis: impute the untrainable Portugal/Agro-forestry cell
    at its random-ranking baseline (AP == prevalence within the evaluation set),
    i.e. the honest worst case for a model with zero information about the class."""
    return portugal_prevalence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-root", default="results/predictions")
    ap.add_argument("--learner", default="lr")
    ap.add_argument("--metadata-path", default="metadata.parquet")
    ap.add_argument("--split-stats-path", default="results/split_stats.json")
    args = ap.parse_args()
    L = args.learner
    root = args.predictions_root

    s1 = _load_run(root, f"s1_{L}_full")
    s2 = _load_run(root, f"s2_{L}_full")
    s3_dirs = [f"s3_fold{f}_{L}_full" for f in range(5)]
    s3_folds = [_load_run(root, d) for d in s3_dirs]
    s3_available = [d for d, f in zip(s3_dirs, s3_folds) if f]
    s4_folds = {c: _load_run(root, f"s4_{c}_{L}_full") for c in COUNTRIES}

    if len(s3_available) < 5:
        print(f"WARNING: only {len(s3_available)}/5 S3 folds found under {root} "
              f"(looked for s3_fold{{0..4}}_{L}_full) -- S3 results will be partial or NaN")

    meta = pd.read_parquet(args.metadata_path)
    portugal_prevalence = float(
        meta.loc[meta.country == "Portugal", "labels"].apply(lambda l: "Agro-forestry areas" in l).mean()
    )

    with open(args.split_stats_path) as f:
        split_stats = json.load(f)
    s2_train_n = split_stats["s2_counts"]["train"]
    s4_train_ratio = {
        c: (split_stats["n_patches"] - split_stats["s4_fold_sizes"][c]) / s2_train_n
        for c in COUNTRIES
    }

    EXCLUDE = set()  # 19-class view (imputed sensitivity below); {AGRO_FORESTRY_IDX} for 18-class primary

    def regime_block(fold_datas, exclude):
        return {
            "AP": class_major_aggregate(fold_datas, _safe_ap, exclude),
            "AUROC": class_major_aggregate(fold_datas, _safe_auroc, exclude),
        }

    result = {
        "method_note": (
            "Class-major aggregation: for multi-fold regimes (S3, S4), each class's "
            "score is averaged only over folds where that class had test positives, "
            "then macro-averaged across classes -- so S1/S2/S3/S4 are always compared "
            "over an identical class set, unlike the original pooled/skip-empty version."
        ),
        "primary_18class_common_set": {
            "note": "Agro-forestry areas excluded from ALL regimes (incl. S1/S2) for a "
                    "fully comparable set, since it is untrainable in the Portugal-held-out "
                    "S4 fold by construction (100% of its positives are in Portugal).",
            "S1": regime_block([s1], {AGRO_FORESTRY_IDX}),
            "S2": regime_block([s2], {AGRO_FORESTRY_IDX}),
            "S3_fold_avg": regime_block(s3_folds, {AGRO_FORESTRY_IDX}),
            "S4_fold_avg": regime_block(list(s4_folds.values()), {AGRO_FORESTRY_IDX}),
        },
        "sensitivity_19class_portugal_imputed": {
            "note": "Agro-forestry INCLUDED; its untrainable Portugal/S4 cell is imputed "
                    "at the random-ranking baseline (AP = within-Portugal prevalence = "
                    f"{portugal_prevalence:.4f}) rather than silently dropped -- the honest "
                    "worst case for a model with zero information about the class.",
            "portugal_agro_forestry_prevalence_imputed_ap": portugal_prevalence,
            "S1": regime_block([s1], EXCLUDE),
            "S2": regime_block([s2], EXCLUDE),
            "S3_fold_avg": regime_block(s3_folds, EXCLUDE),
            "S4_fold_avg_with_imputation": None,  # filled below
        },
        "secondary_caveated": {
            "note": "Pooled-AP conflates discrimination loss with cross-model score "
                    "incommensurability (different classifier per fold). Do not headline.",
            "S3_pooled_AP": pooled_metric(s3_folds, _safe_ap, {AGRO_FORESTRY_IDX}),
            "S4_pooled_AP": pooled_metric(list(s4_folds.values()), _safe_ap, {AGRO_FORESTRY_IDX}),
        },
        "training_data_asymmetry": {
            "note": "S4 models train on MORE data than S2 despite scoring worse -- rules "
                    "out 'smaller training set' as an explanation for the gap.",
            "s2_train_n": s2_train_n,
            "s4_train_n_ratio_to_s2": s4_train_ratio,
        },
        "s3_folds_found": s3_available,
        "s4_per_country": {c: regime_block([f], {AGRO_FORESTRY_IDX}) for c, f in s4_folds.items()},
    }

    # 19-class S4 with Portugal's Agro-forestry cell imputed rather than dropped
    s4_imputed_per_class = {i: [] for i in range(N_LABELS)}
    for country, fold_data in s4_folds.items():
        for i in range(N_LABELS):
            if i == AGRO_FORESTRY_IDX and country == "Portugal":
                s4_imputed_per_class[i].append(agro_forestry_imputed_ap(portugal_prevalence))
                continue
            if i in fold_data:
                v = _safe_ap(*fold_data[i])
                if v is not None:
                    s4_imputed_per_class[i].append(v)
    s4_imputed_means = {CLASSES[i]: (float(np.mean(v)) if v else None) for i, v in s4_imputed_per_class.items()}
    valid = [v for v in s4_imputed_means.values() if v is not None]
    result["sensitivity_19class_portugal_imputed"]["S4_fold_avg_with_imputation"] = {
        "macro_AP": float(np.mean(valid)) if valid else float("nan"),
        "n_classes_used": len(valid),
        "per_class_mean_AP": s4_imputed_means,
    }

    # headline gaps, primary (18-class) view
    p = result["primary_18class_common_set"]
    result["headline_optimism_gap_points"] = {
        "S1_to_S2_AP": (p["S1"]["AP"]["macro"] - p["S2"]["AP"]["macro"]) * 100,
        "S2_to_S3_AP": (p["S2"]["AP"]["macro"] - p["S3_fold_avg"]["AP"]["macro"]) * 100,
        "S2_to_S4_AP": (p["S2"]["AP"]["macro"] - p["S4_fold_avg"]["AP"]["macro"]) * 100,
        "S1_to_S4_AP": (p["S1"]["AP"]["macro"] - p["S4_fold_avg"]["AP"]["macro"]) * 100,
        "S2_to_S4_AUROC": (p["S2"]["AUROC"]["macro"] - p["S4_fold_avg"]["AUROC"]["macro"]) * 100,
    }

    print(json.dumps(result["headline_optimism_gap_points"], indent=2))
    print(f"\nS1 macro-AP: {p['S1']['AP']['macro']*100:.2f}  (n_classes={p['S1']['AP']['n_classes_used']})")
    print(f"S2 macro-AP: {p['S2']['AP']['macro']*100:.2f}  (n_classes={p['S2']['AP']['n_classes_used']})")
    print(f"S3 macro-AP: {p['S3_fold_avg']['AP']['macro']*100:.2f}  (n_classes={p['S3_fold_avg']['AP']['n_classes_used']})")
    print(f"S4 macro-AP: {p['S4_fold_avg']['AP']['macro']*100:.2f}  (n_classes={p['S4_fold_avg']['AP']['n_classes_used']})")
    print(f"\nS2 macro-AUROC: {p['S2']['AUROC']['macro']*100:.2f}")
    print(f"S4 macro-AUROC: {p['S4_fold_avg']['AUROC']['macro']*100:.2f}  "
          f"(persisting AUROC gap would refute pure prior-shift)")

    os.makedirs("results", exist_ok=True)
    # Learner-specific filename: this used to be a fixed "results/optimism_gap.json"
    # regardless of --learner, so running it for RF silently clobbered the LR
    # results (discovered when Week 2's RF ladder overwrote Week 1's LR numbers).
    out_path = f"results/optimism_gap_{L}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
