"""
Week-1 item 2 of the strengthening plan: the decisive prior-shift control.

For each country c, evaluate two models on the IDENTICAL patch set
(S2's test patches that happen to lie in country c):
  1. the S2 model      -- country c WAS in its training data
  2. the S4-fold-c model -- country c was NOT in its training data

Because both models are scored on the same patches, P(y) is held fixed by
construction: any AP/AUROC gap cannot be explained by label-prior shift. This
is a stronger argument than a persisting-AUROC check alone, and it gives a
clean paired design (n=10 countries) for a Wilcoxon signed-rank test.

Verified feasible before writing this: all 40,802 S2-test Finland patches are
present in the S4 Finland fold's test set (S4's test set for a held-out
country is that country's ENTIRE patch set, a superset of S2's ~25% test
slice of it).

Input: results/predictions/{s2_lr_full, s4_<Country>_lr_full}/label_NN/test/*.parquet
Output: results/paired_country_eval.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
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
COUNTRIES = ["Finland", "Portugal", "Serbia", "Lithuania", "Ireland",
             "Austria", "Belgium", "Switzerland", "Luxembourg", "Kosovo"]


def _load_label(predictions_root, run_dir, label_idx):
    files = glob.glob(os.path.join(predictions_root, run_dir, f"label_{label_idx:02d}", "test", "*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _safe_ap(t, p):
    return float(average_precision_score(t, p)) if t.sum() > 0 else None


def _safe_auroc(t, p):
    if t.sum() == 0 or t.sum() == len(t):
        return None
    return float(roc_auc_score(t, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", default="results/predictions")
    parser.add_argument("--learner", default="lr")
    parser.add_argument("--metadata-path", default="metadata.parquet")
    parser.add_argument("--split-stats-path", default="results/split_stats.json")
    args = parser.parse_args()
    L = args.learner
    root = args.predictions_root

    meta = pd.read_parquet(args.metadata_path)[["patch_id", "country"]]
    with open(args.split_stats_path) as f:
        split_stats = json.load(f)
    country_frac_of_data = {c: split_stats["s4_fold_sizes"][c] / split_stats["n_patches"] for c in COUNTRIES}

    per_country_results = {}
    for country in COUNTRIES:
        country_patch_ids = set(meta.loc[meta.country == country, "patch_id"])

        per_label_ap_in, per_label_ap_out = {}, {}
        per_label_auroc_in, per_label_auroc_out = {}, {}
        n_shared_patches = None

        for i in range(N_LABELS):
            if i == AGRO_FORESTRY_IDX and country == "Portugal":
                continue  # untrainable by construction -- see compute_optimism_gap.py

            s2_df = _load_label(root, f"s2_{L}_full", i)
            s4_df = _load_label(root, f"s4_{country}_{L}_full", i)
            if s2_df is None or s4_df is None:
                continue

            s2_in_country = s2_df[s2_df["patch_id"].isin(country_patch_ids)]
            if s2_in_country.empty:
                continue
            shared_ids = set(s2_in_country["patch_id"])
            s4_matched = s4_df[s4_df["patch_id"].isin(shared_ids)]

            if len(s4_matched) != len(s2_in_country):
                raise AssertionError(
                    f"{country} label {i}: S2-in-country has {len(s2_in_country)} patches but "
                    f"only {len(s4_matched)} found in the S4 fold's test set -- the paired "
                    f"evaluation set is not identical, this control is invalid until fixed."
                )
            if n_shared_patches is not None and n_shared_patches != len(shared_ids):
                raise AssertionError(
                    f"{country} label {i}: shared-patch count {len(shared_ids)} differs from "
                    f"{n_shared_patches} seen for an earlier label in this country -- the test "
                    f"set is supposed to be identical across all 19 labels within one regime; "
                    f"something upstream (role assignment) is inconsistent."
                )
            n_shared_patches = len(shared_ids)

            # align on patch_id so "in" and "out" predictions refer to the same rows
            s2_sorted = s2_in_country.sort_values("patch_id").reset_index(drop=True)
            s4_sorted = s4_matched.sort_values("patch_id").reset_index(drop=True)
            assert (s2_sorted["patch_id"].values == s4_sorted["patch_id"].values).all()
            assert (s2_sorted["true_label"].values == s4_sorted["true_label"].values).all(), \
                f"{country} label {i}: true_label mismatch between S2 and S4 predictions for the same patches"

            t = s2_sorted["true_label"].to_numpy()
            ap_in = _safe_ap(t, s2_sorted["probability"].to_numpy())
            ap_out = _safe_ap(t, s4_sorted["probability"].to_numpy())
            auroc_in = _safe_auroc(t, s2_sorted["probability"].to_numpy())
            auroc_out = _safe_auroc(t, s4_sorted["probability"].to_numpy())

            if ap_in is not None and ap_out is not None:
                per_label_ap_in[CLASSES[i]] = ap_in
                per_label_ap_out[CLASSES[i]] = ap_out
            if auroc_in is not None and auroc_out is not None:
                per_label_auroc_in[CLASSES[i]] = auroc_in
                per_label_auroc_out[CLASSES[i]] = auroc_out

        macro_ap_in = float(np.mean(list(per_label_ap_in.values()))) if per_label_ap_in else None
        macro_ap_out = float(np.mean(list(per_label_ap_out.values()))) if per_label_ap_out else None
        macro_auroc_in = float(np.mean(list(per_label_auroc_in.values()))) if per_label_auroc_in else None
        macro_auroc_out = float(np.mean(list(per_label_auroc_out.values()))) if per_label_auroc_out else None

        per_country_results[country] = {
            "n_shared_patches": n_shared_patches,
            "n_classes_used_ap": len(per_label_ap_in),
            "macro_AP_in_training": macro_ap_in,
            "macro_AP_held_out": macro_ap_out,
            "delta_AP_points": (macro_ap_in - macro_ap_out) * 100 if macro_ap_in is not None else None,
            "macro_AUROC_in_training": macro_auroc_in,
            "macro_AUROC_held_out": macro_auroc_out,
            "delta_AUROC_points": (macro_auroc_in - macro_auroc_out) * 100 if macro_auroc_in is not None else None,
        }

    ap_deltas = [v["delta_AP_points"] for v in per_country_results.values() if v["delta_AP_points"] is not None]
    auroc_deltas = [v["delta_AUROC_points"] for v in per_country_results.values() if v["delta_AUROC_points"] is not None]
    ap_countries = [c for c, v in per_country_results.items() if v["delta_AP_points"] is not None]
    ap_weights = np.array([per_country_results[c]["n_shared_patches"] for c in ap_countries], dtype=float)
    ap_fracs = np.array([country_frac_of_data[c] for c in ap_countries])

    def summarize(deltas, name):
        arr = np.array(deltas)
        if len(arr) == 0:
            print(f"{name}: n=0 -- no country produced a valid delta, cannot summarize")
            return {"n": 0, "median": None, "iqr_25": None, "iqr_75": None,
                    "wilcoxon_statistic": None, "wilcoxon_pvalue": None, "all_deltas": []}
        # one-sample Wilcoxon signed-rank: H0 is that the median delta is zero
        stat, pval = wilcoxon(arr) if np.any(arr != 0) else (float("nan"), float("nan"))
        print(f"{name}: n={len(arr)}  median={np.median(arr):.2f}  "
              f"IQR=[{np.percentile(arr,25):.2f}, {np.percentile(arr,75):.2f}]  "
              f"Wilcoxon p={pval:.4g}")
        return {"n": len(arr), "median": float(np.median(arr)),
                "iqr_25": float(np.percentile(arr, 25)), "iqr_75": float(np.percentile(arr, 75)),
                "wilcoxon_statistic": float(stat), "wilcoxon_pvalue": float(pval),
                "all_deltas": deltas}

    result = {
        "method_note": (
            "Same-region paired evaluation: for each country, the S2 model (trained WITH "
            "that country) and the S4-fold model (trained WITHOUT it) are both scored on "
            "the identical S2-test patch subset lying in that country. P(y) is held fixed "
            "by construction, so any gap cannot be a label-prior-shift artifact."
        ),
        "per_country": per_country_results,
        "summary_delta_AP": summarize(ap_deltas, "Delta-AP (in-training minus held-out)"),
        "summary_delta_AUROC": summarize(auroc_deltas, "Delta-AUROC (in-training minus held-out)"),
    }

    # Secondary finding: the paired penalty scales with how much of the total
    # training distribution the held-out country represents. Large countries
    # (Finland, 32% of data) lose a lot when excluded; tiny ones (Kosovo, 0.3%)
    # lose almost nothing, because excluding them barely changes the model in
    # the first place. This explains why the unweighted median delta (dominated
    # by four near-zero small countries) undersells the effect for the patches
    # that actually make up most of the dataset.
    if len(ap_deltas) < 2:
        print(f"\n[skip] delta_scales_with_country_data_share: only {len(ap_deltas)} "
              f"countries produced a valid delta, need >=2 for a correlation")
        result["delta_scales_with_country_data_share"] = None
    else:
        rho, pval = spearmanr(np.array(ap_deltas), ap_fracs)
        weighted_mean_delta = float(np.average(ap_deltas, weights=ap_weights))
        result["delta_scales_with_country_data_share"] = {
            "note": (
                "Spearman correlation between each country's paired Delta-AP and its share "
                "of total training data. A strong positive correlation shows the country-"
                "absence effect is real and mechanistic (proportional to how much of the "
                "training distribution is removed), not noise concentrated in one country."
            ),
            "spearman_rho": float(rho),
            "spearman_pvalue": float(pval),
            "unweighted_median_delta_AP": float(np.median(ap_deltas)),
            "unweighted_mean_delta_AP": float(np.mean(ap_deltas)),
            "patch_count_weighted_mean_delta_AP": weighted_mean_delta,
        }
        print(f"\nSpearman(delta_AP, country_data_share) = {rho:.3f}, p = {pval:.4g}")
        print(f"Patch-count-weighted mean delta-AP: {weighted_mean_delta:.2f} "
              f"(vs unweighted median {np.median(ap_deltas):.2f})")

    os.makedirs("results", exist_ok=True)
    with open("results/paired_country_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote results/paired_country_eval.json")


if __name__ == "__main__":
    main()
