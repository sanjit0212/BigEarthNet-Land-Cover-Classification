"""
Week-1 item 4: tile-clustered bootstrap 95% CIs for the S1/S2/S3 macro-AP and
macro-AUROC point estimates (metrics.py, validated against sklearn).

S4 is deliberately excluded here -- per the strengthening plan, individual S4
folds don't support tile clustering (Kosovo/Luxembourg are ~1-2 MGRS tiles, so
the resample collapses to almost no variation); S4's correct uncertainty is
the between-fold spread across the 10 countries, already reported in
compute_optimism_gap.py and paired_country_eval.py.

Output: results/bootstrap_cis.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import load_regime_matrix, multilabel_metrics, tile_clustered_bootstrap  # noqa: E402


def main():
    root = "results/predictions"
    result = {}

    regime_runs = {
        "S1": "s1_lr_full",
        "S2": "s2_lr_full",
        "S3_fold0": "s3_fold0_lr_full",
        "S3_fold1": "s3_fold1_lr_full",
        "S3_fold2": "s3_fold2_lr_full",
        "S3_fold3": "s3_fold3_lr_full",
        "S3_fold4": "s3_fold4_lr_full",
    }

    out_path = "results/bootstrap_cis.json"
    os.makedirs("results", exist_ok=True)
    # Resume support: a regime already present in a prior partial run is not
    # recomputed. This matters because one crashed regime used to discard
    # every regime computed before it (nothing was written until the full loop
    # finished) -- each regime is now saved to disk as soon as it completes.
    if os.path.exists(out_path):
        with open(out_path) as f:
            result = json.load(f)
        if result:
            print(f"Resuming: {list(result.keys())} already computed in {out_path}")

    for name, run_dir in regime_runs.items():
        if name in result:
            print(f"[skip] {name}: already in {out_path}")
            continue
        if not os.path.isdir(os.path.join(root, run_dir)):
            print(f"[skip] {name}: {run_dir} not found under {root} yet")
            continue

        patch_ids, Y_true, Y_prob, label_names = load_regime_matrix(root, run_dir, role="test")
        point = multilabel_metrics(Y_true, Y_prob)
        ci = tile_clustered_bootstrap(patch_ids, Y_true, Y_prob, n_boot=1000)
        result[name] = {
            "n_samples": point["n_samples"], "n_classes_used_ap": point["n_classes_used_ap"],
            "n_tiles": ci["n_tiles"], "n_workers": ci["n_workers"],
            "AP_macro": ci["AP_macro"], "AUROC_macro": ci["AUROC_macro"],
        }
        ap_ci = ci["AP_macro"]
        print(f"{name:<10s} AP_macro={ap_ci['point_estimate']*100:5.2f}  "
              f"95% CI=[{ap_ci['ci_lower_2.5']*100:5.2f}, {ap_ci['ci_upper_97.5']*100:5.2f}]  "
              f"(n_tiles={ci['n_tiles']}, n_workers={ci['n_workers']})", flush=True)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
