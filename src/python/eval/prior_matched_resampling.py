"""
Week 2 item 11 of the strengthening plan: prior-matched resampling, the
confirmatory prior-shift control (secondary to the same-region paired
evaluation in paired_country_eval.py, which is sound by construction; this
one is weaker -- matches P(y) only, not P(x|y) -- but cheap and worth having
as a second line of defense).

For each class and each S4 country fold, subsample whichever class (positives
or negatives) is in excess so the fold's positive rate matches S2's test-set
prevalence for that class, then recompute AP. Average over N_DRAWS random
subsamples. If the S2->S4 gap survives prior-matching, prevalence shift alone
cannot explain it.

Input: results/predictions/{s2_lr_full, s4_<Country>_lr_full}
Output: results/prior_matched_resampling.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
COUNTRIES = ["Finland", "Portugal", "Serbia", "Lithuania", "Ireland",
             "Austria", "Belgium", "Switzerland", "Luxembourg", "Kosovo"]
N_DRAWS = 100
SEED = 42


def _load_label(predictions_root, run_dir, label_idx, role="test"):
    files = glob.glob(os.path.join(predictions_root, run_dir, f"label_{label_idx:02d}", role, "*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def prior_matched_ap(y_true: np.ndarray, y_prob: np.ndarray, target_prevalence: float,
                      n_draws: int, rng: np.random.Generator):
    """Subsample (never oversample) whichever class is in excess so the
    resampled set's positive rate equals target_prevalence; average AP over
    n_draws random subsamples. Returns (mean_ap, mode, n_kept) or None if the
    target prevalence can't be hit by subsampling alone (degenerate fold)."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0 or target_prevalence <= 0 or target_prevalence >= 1:
        return None

    n_neg_needed = int(round(n_pos * (1 - target_prevalence) / target_prevalence))
    if n_neg_needed <= n_neg:
        mode, n_pos_draw, n_neg_draw = "subsample_neg", n_pos, n_neg_needed
    else:
        n_pos_needed = int(round(n_neg * target_prevalence / (1 - target_prevalence)))
        if 0 < n_pos_needed <= n_pos:
            mode, n_pos_draw, n_neg_draw = "subsample_pos", n_pos_needed, n_neg
        else:
            return None

    aps = []
    for _ in range(n_draws):
        sampled_pos = pos_idx if n_pos_draw == n_pos else rng.choice(pos_idx, size=n_pos_draw, replace=False)
        sampled_neg = neg_idx if n_neg_draw == n_neg else rng.choice(neg_idx, size=n_neg_draw, replace=False)
        idx = np.concatenate([sampled_pos, sampled_neg])
        aps.append(average_precision_score(y_true[idx], y_prob[idx]))
    return {"mean_ap": float(np.mean(aps)), "mode": mode, "n_kept": int(n_pos_draw + n_neg_draw)}


def main():
    root = "results/predictions"
    rng = np.random.default_rng(SEED)

    # S2 reference prevalence per class (target for matching)
    s2_prevalence = {}
    s2_raw_ap = {}
    for i in range(N_LABELS):
        if i == AGRO_FORESTRY_IDX:
            continue
        df = _load_label(root, "s2_lr_full", i)
        if df is None:
            continue
        t = df["true_label"].to_numpy()
        s2_prevalence[i] = float(t.mean())
        s2_raw_ap[i] = float(average_precision_score(t, df["probability"].to_numpy()))

    per_class_matched = {i: [] for i in s2_prevalence}
    per_class_raw = {i: [] for i in s2_prevalence}
    per_class_cells = {i: 0 for i in s2_prevalence}
    skipped = []

    for country in COUNTRIES:
        for i in s2_prevalence:
            df = _load_label(root, f"s4_{country}_lr_full", i)
            if df is None:
                continue
            t = df["true_label"].to_numpy()
            p = df["probability"].to_numpy()
            if t.sum() == 0:
                continue
            raw_ap = float(average_precision_score(t, p))
            result = prior_matched_ap(t, p, s2_prevalence[i], N_DRAWS, rng)
            if result is None:
                skipped.append({"country": country, "label": CLASSES[i], "reason": "target prevalence unreachable by subsampling"})
                continue
            per_class_matched[i].append(result["mean_ap"])
            per_class_raw[i].append(raw_ap)
            per_class_cells[i] += 1

    per_class_matched_mean = {CLASSES[i]: (float(np.mean(v)) if v else None) for i, v in per_class_matched.items()}
    per_class_raw_mean = {CLASSES[i]: (float(np.mean(v)) if v else None) for i, v in per_class_raw.items()}

    valid_matched = [v for v in per_class_matched_mean.values() if v is not None]
    valid_raw = [v for v in per_class_raw_mean.values() if v is not None]
    s2_macro_ap = float(np.mean(list(s2_raw_ap.values())))
    s4_macro_ap_raw = float(np.mean(valid_raw))
    s4_macro_ap_matched = float(np.mean(valid_matched))

    result = {
        "method_note": (
            "For each (class, S4 country) cell, subsample whichever class (pos/neg) is "
            "in excess so the fold's positive rate matches S2's test-set prevalence for "
            "that class, then recompute AP (mean over 100 random subsamples). This "
            "matches P(y) only, not P(x|y) -- a weaker, confirmatory control alongside "
            "the same-region paired evaluation in paired_country_eval.py, which is sound "
            "by construction. If the gap survives prior-matching, prevalence shift alone "
            "cannot explain it."
        ),
        "n_draws": N_DRAWS,
        "s2_macro_ap": s2_macro_ap,
        "s4_macro_ap_raw": s4_macro_ap_raw,
        "s4_macro_ap_prior_matched": s4_macro_ap_matched,
        "gap_s2_to_s4_raw_points": (s2_macro_ap - s4_macro_ap_raw) * 100,
        "gap_s2_to_s4_prior_matched_points": (s2_macro_ap - s4_macro_ap_matched) * 100,
        "gap_explained_by_prior_shift_points": (s4_macro_ap_matched - s4_macro_ap_raw) * 100,
        "per_class_s2_prevalence": {CLASSES[i]: v for i, v in s2_prevalence.items()},
        "per_class_raw_ap": per_class_raw_mean,
        "per_class_prior_matched_ap": per_class_matched_mean,
        "per_class_cells_used": {CLASSES[i]: c for i, c in per_class_cells.items()},
        "skipped_cells": skipped,
    }

    print(f"S2 macro-AP (reference):              {s2_macro_ap*100:.2f}")
    print(f"S4 macro-AP (raw):                     {s4_macro_ap_raw*100:.2f}")
    print(f"S4 macro-AP (prior-matched to S2):     {s4_macro_ap_matched*100:.2f}")
    print(f"\nRaw S2->S4 gap:            {result['gap_s2_to_s4_raw_points']:.2f} points")
    print(f"Prior-matched S2->S4 gap:  {result['gap_s2_to_s4_prior_matched_points']:.2f} points")
    print(f"Gap 'explained' by prior shift: {result['gap_explained_by_prior_shift_points']:.2f} points "
          f"({result['gap_explained_by_prior_shift_points']/result['gap_s2_to_s4_raw_points']*100:.1f}% of the raw gap)")
    if skipped:
        print(f"\n{len(skipped)} (class, country) cells skipped (prevalence unreachable by subsampling)")

    os.makedirs("results", exist_ok=True)
    with open("results/prior_matched_resampling.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote results/prior_matched_resampling.json")


if __name__ == "__main__":
    main()
