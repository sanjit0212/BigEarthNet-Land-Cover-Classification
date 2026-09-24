"""
PLAN.md-specified evaluation metrics library (never written until now).
Validated against sklearn on synthetic data -- run this file directly
(`python metrics.py`) to self-test before any reported number depends on it.

Provides:
  - load_regime_matrix(...): build an aligned (patch_id x label) matrix from
    BinaryRelevanceRunner's per-label parquet output. Handles regimes where
    not all 19 labels were trained (e.g. Portugal/Agro-forestry) by returning
    only the labels actually present.
  - multilabel_metrics(Y_true, Y_prob): AP^M/AP^mu, AUROC^M/AUROC^mu,
    F1^M/F1^mu, Hamming loss, subset accuracy, per-class breakdowns.
  - tile_clustered_bootstrap(...): bootstrap CI resampling whole MGRS tiles,
    not patches -- patches within a tile are spatially autocorrelated and
    reBEN's official split puts train/val/test inside the SAME tile, so a
    patch-level bootstrap gives falsely narrow CIs (see docstring below).
"""
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, hamming_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingest"))
from patch_meta import parse_patch_id  # noqa: E402

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


def load_regime_matrix(predictions_root: str, run_dir_name: str, role: str = "test"):
    """Returns (patch_ids, Y_true [n,k], Y_prob [n,k], label_names) where k<=19
    is however many labels have prediction files for this regime (a missing
    label means it was never trained -- zero train positives, e.g. Portugal/
    Agro-forestry). Rows are the intersection of patch_ids present in every
    available label's predictions (labels are trained on the same role split,
    so in practice this is all of them, but the intersection is taken
    defensively rather than assumed)."""
    label_dfs = {}
    for i in range(N_LABELS):
        files = glob.glob(os.path.join(predictions_root, run_dir_name, f"label_{i:02d}", role, "*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        label_dfs[i] = df.set_index("patch_id")[["true_label", "probability"]]

    if not label_dfs:
        raise ValueError(f"no label predictions found under {predictions_root}/{run_dir_name}/*/{role}")

    common_ids = None
    for df in label_dfs.values():
        ids = set(df.index)
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids)

    label_idxs = sorted(label_dfs.keys())
    Y_true = np.zeros((len(common_ids), len(label_idxs)), dtype=int)
    Y_prob = np.zeros((len(common_ids), len(label_idxs)), dtype=float)
    for col, i in enumerate(label_idxs):
        sub = label_dfs[i].loc[common_ids]
        Y_true[:, col] = sub["true_label"].to_numpy()
        Y_prob[:, col] = sub["probability"].to_numpy()

    return np.array(common_ids), Y_true, Y_prob, [CLASSES[i] for i in label_idxs]


def multilabel_metrics(Y_true: np.ndarray, Y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Y_true, Y_prob: [n_samples, n_classes]. Classes with zero positives (AP)
    or only one class present (AUROC) are skipped for that metric and reported
    in n_classes_used_{ap,auroc} -- never silently averaged as zero."""
    n, k = Y_true.shape
    Y_pred = (Y_prob >= threshold).astype(int)

    per_class_ap, per_class_auroc, per_class_f1 = {}, {}, {}
    for j in range(k):
        t, p = Y_true[:, j], Y_prob[:, j]
        if t.sum() > 0:
            per_class_ap[j] = float(average_precision_score(t, p))
        if 0 < t.sum() < n:
            per_class_auroc[j] = float(roc_auc_score(t, p))
        per_class_f1[j] = float(f1_score(t, Y_pred[:, j], zero_division=0))

    ap_macro = float(np.mean(list(per_class_ap.values()))) if per_class_ap else float("nan")
    auroc_macro = float(np.mean(list(per_class_auroc.values()))) if per_class_auroc else float("nan")
    f1_macro = float(np.mean(list(per_class_f1.values())))

    ap_cols = sorted(per_class_ap.keys())
    ap_micro = (float(average_precision_score(Y_true[:, ap_cols].ravel(), Y_prob[:, ap_cols].ravel()))
                if ap_cols else float("nan"))
    auroc_cols = sorted(per_class_auroc.keys())
    auroc_micro = (float(roc_auc_score(Y_true[:, auroc_cols].ravel(), Y_prob[:, auroc_cols].ravel()))
                   if auroc_cols else float("nan"))
    f1_micro = float(f1_score(Y_true.ravel(), Y_pred.ravel(), zero_division=0))

    return {
        "AP_macro": ap_macro, "AP_micro": ap_micro,
        "AUROC_macro": auroc_macro, "AUROC_micro": auroc_micro,
        "F1_macro": f1_macro, "F1_micro": f1_micro,
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "subset_accuracy": float(accuracy_score(Y_true, Y_pred)),
        "n_classes_used_ap": len(ap_cols), "n_classes_used_auroc": len(auroc_cols),
        "n_samples": n, "n_classes": k,
        "per_class_AP": per_class_ap, "per_class_AUROC": per_class_auroc, "per_class_F1": per_class_f1,
    }


def _macro_ap_auroc_only(Y_true: np.ndarray, Y_prob: np.ndarray) -> dict:
    """Lean inner loop for bootstrap replicates: only AP_macro/AUROC_macro,
    skipping F1/Hamming/subset-accuracy (which need a threshold and aren't
    used for these CIs) and skipping the micro variants -- full
    multilabel_metrics() was the bottleneck at 1000+ replicates."""
    n, k = Y_true.shape
    ap_vals, auroc_vals = [], []
    for j in range(k):
        t, p = Y_true[:, j], Y_prob[:, j]
        if t.sum() > 0:
            ap_vals.append(average_precision_score(t, p))
        if 0 < t.sum() < n:
            auroc_vals.append(roc_auc_score(t, p))
    return {
        "AP_macro": float(np.mean(ap_vals)) if ap_vals else float("nan"),
        "AUROC_macro": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
    }


_worker_state = {}


def _init_bootstrap_worker(Y_true, Y_prob, tile_index_arrays, metric_keys):
    """Pool initializer -- ships the (read-only) data ONCE per worker process
    instead of pickling it into every task, which is what made a naive
    multiprocessing.Pool.map over individual replicates not worth it here."""
    _worker_state["Y_true"] = Y_true
    _worker_state["Y_prob"] = Y_prob
    _worker_state["tile_index_arrays"] = tile_index_arrays
    _worker_state["metric_keys"] = metric_keys


def _run_replicate_batch(Y_true, Y_prob, tile_index_arrays, metric_keys, seed, n_reps):
    n_tiles = len(tile_index_arrays)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in metric_keys}
    for _ in range(n_reps):
        sampled = rng.integers(0, n_tiles, size=n_tiles)
        idx = np.concatenate([tile_index_arrays[i] for i in sampled])
        rep = _macro_ap_auroc_only(Y_true[idx], Y_prob[idx])
        for k in metric_keys:
            if not np.isnan(rep[k]):
                out[k].append(rep[k])
    return out


def _bootstrap_replicate_batch(seed_and_count):
    """Runs in a worker process; reads data shipped once via the Pool initializer."""
    seed, n_reps = seed_and_count
    return _run_replicate_batch(_worker_state["Y_true"], _worker_state["Y_prob"],
                                 _worker_state["tile_index_arrays"], _worker_state["metric_keys"],
                                 seed, n_reps)


def tile_clustered_bootstrap(patch_ids: np.ndarray, Y_true: np.ndarray, Y_prob: np.ndarray,
                              metric_keys=("AP_macro", "AUROC_macro"), n_boot: int = 1000, seed: int = 42,
                              tile_ids: np.ndarray = None, n_workers: int = None) -> dict:
    """Resamples whole MGRS TILES with replacement (not patches), computing all
    requested metric_keys per replicate in one pass. Patches within a tile are
    spatially autocorrelated, and reBEN's concentric split puts train/val/test
    inside the SAME tile, so a naive patch-level bootstrap understates
    uncertainty. tile_ids is derived from patch_ids via patch_meta.parse_patch_id
    if not supplied directly (real data path); passing tile_ids explicitly is
    how the self-test exercises this with synthetic labels that aren't valid
    BigEarthNet patch_ids. Only AP_macro and AUROC_macro are supported (see
    _macro_ap_auroc_only).

    Parallelized across n_workers processes (default 4, see the comment below
    for why not higher): each sklearn AP/AUROC call is ~15-20ms and a real run
    needs 19 classes x 2 metrics x n_boot replicates, which is minutes of
    single-threaded work per regime -- multiprocessing gives near-linear
    speedup since replicates are embarrassingly parallel. Requires being
    called from an `if __name__ == "__main__":` guard (Windows uses spawn).
    Falls back to sequential (single-process) computation if the pool itself
    fails to start or crashes (observed in practice: spawning many workers
    that each re-import numpy/scipy/sklearn/pandas from scratch can exhaust
    the Windows paging file when Docker/Spark containers are also running)."""
    if tile_ids is None:
        tile_ids = np.array([parse_patch_id(pid)["tile"] for pid in patch_ids])
    unique_tiles = np.unique(tile_ids)
    tile_index_arrays = [np.where(tile_ids == t)[0] for t in unique_tiles]

    point = _macro_ap_auroc_only(Y_true, Y_prob)

    # Conservative default: each spawned worker re-imports the full numpy/scipy/
    # sklearn/pandas stack from scratch (Windows has no fork/copy-on-write), which
    # is memory-heavy enough to hit "paging file too small" DLL-load errors at
    # 14 workers when the Spark/HDFS Docker containers are also running. 4 is a
    # safe default; raise it only if you know the machine has headroom.
    n_workers = n_workers or 4
    n_workers = min(n_workers, n_boot)
    rng = np.random.default_rng(seed)
    worker_seeds = rng.integers(0, 2**31 - 1, size=n_workers)
    base, extra = divmod(n_boot, n_workers)
    batch_sizes = [base + (1 if i < extra else 0) for i in range(n_workers)]
    tasks = [(int(s), n) for s, n in zip(worker_seeds, batch_sizes) if n > 0]

    # A worker that dies during its own startup (e.g. the Windows "paging file
    # too small" DLL-load error observed in practice, spawning many workers
    # that each re-import numpy/scipy/sklearn/pandas while Docker/Spark
    # containers are also using memory) does NOT reliably surface as an
    # exception from plain Pool.map() -- it can just hang forever instead.
    # map_async().get(timeout=...) converts that hang into a catchable
    # TimeoutError; Pool.__exit__ terminates any stuck worker processes.
    used_workers = len(tasks)
    timeout_seconds = max(600, n_boot * 0.5)  # generous: normal runs finish in well under this
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks), initializer=_init_bootstrap_worker,
                      initargs=(Y_true, Y_prob, tile_index_arrays, metric_keys)) as pool:
            batch_results = pool.map_async(_bootstrap_replicate_batch, tasks).get(timeout=timeout_seconds)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any multiprocessing
        # failure or timeout should degrade to a working, if slower,
        # single-process run rather than lose all progress or hang forever.
        print(f"[tile_clustered_bootstrap] multiprocessing failed ({e!r}), "
              f"falling back to sequential bootstrap -- this will be slower", flush=True)
        used_workers = 1
        batch_results = [_run_replicate_batch(Y_true, Y_prob, tile_index_arrays, metric_keys, seed, n_boot)]

    boot_vals = {k: [] for k in metric_keys}
    for batch in batch_results:
        for k in metric_keys:
            boot_vals[k].extend(batch[k])

    out = {"n_boot_requested": n_boot, "n_tiles": len(unique_tiles), "n_workers": used_workers}
    for k in metric_keys:
        vals = np.array(boot_vals[k])
        out[k] = {
            "point_estimate": point[k],
            "boot_mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "ci_lower_2.5": float(np.percentile(vals, 2.5)) if len(vals) else float("nan"),
            "ci_upper_97.5": float(np.percentile(vals, 97.5)) if len(vals) else float("nan"),
            "n_boot_valid": len(vals),
        }
    return out


def patch_level_bootstrap(Y_true: np.ndarray, Y_prob: np.ndarray, metric_key: str = "AP_macro",
                           n_boot: int = 1000, seed: int = 42, threshold: float = 0.5) -> dict:
    """Naive iid bootstrap over individual patches -- kept only as the
    comparison point proving the tile-clustered CI is wider (see self-test)."""
    n = Y_true.shape[0]
    point = multilabel_metrics(Y_true, Y_prob, threshold=threshold)[metric_key]
    rng = np.random.default_rng(seed)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = multilabel_metrics(Y_true[idx], Y_prob[idx], threshold=threshold)[metric_key]
        if not np.isnan(val):
            boot_vals.append(val)
    boot_vals = np.array(boot_vals)
    return {
        "metric": metric_key, "point_estimate": point,
        "boot_mean": float(np.mean(boot_vals)) if len(boot_vals) else float("nan"),
        "ci_lower_2.5": float(np.percentile(boot_vals, 2.5)) if len(boot_vals) else float("nan"),
        "ci_upper_97.5": float(np.percentile(boot_vals, 97.5)) if len(boot_vals) else float("nan"),
        "n_boot_valid": len(boot_vals),
    }


# ---------------------------------------------------------------------------
# Self-test: validate multilabel_metrics against sklearn's own averaging, and
# confirm the tile-clustered bootstrap gives a wider CI than the patch-level one.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.metrics import average_precision_score as sk_ap, roc_auc_score as sk_auroc, f1_score as sk_f1

    rng = np.random.default_rng(0)
    n, k = 5000, 6
    Y_true = (rng.random((n, k)) < 0.15).astype(int)
    Y_true[:, 0] = 1  # force one always-positive column so sklearn's macro doesn't special-case it away
    Y_prob = np.clip(Y_true * 0.5 + rng.normal(0.25, 0.2, (n, k)), 0, 1)

    ours = multilabel_metrics(Y_true, Y_prob)

    # sklearn's own multilabel-aware averaging as ground truth
    sk_ap_macro = sk_ap(Y_true, Y_prob, average="macro")
    sk_ap_micro = sk_ap(Y_true, Y_prob, average="micro")
    sk_f1_macro = sk_f1(Y_true, (Y_prob >= 0.5).astype(int), average="macro", zero_division=0)
    sk_f1_micro = sk_f1(Y_true, (Y_prob >= 0.5).astype(int), average="micro", zero_division=0)

    print(f"AP_macro:  ours={ours['AP_macro']:.6f}  sklearn={sk_ap_macro:.6f}")
    print(f"AP_micro:  ours={ours['AP_micro']:.6f}  sklearn={sk_ap_micro:.6f}")
    print(f"F1_macro:  ours={ours['F1_macro']:.6f}  sklearn={sk_f1_macro:.6f}")
    print(f"F1_micro:  ours={ours['F1_micro']:.6f}  sklearn={sk_f1_micro:.6f}")

    assert abs(ours["AP_macro"] - sk_ap_macro) < 1e-9, "AP_macro mismatch vs sklearn"
    assert abs(ours["AP_micro"] - sk_ap_micro) < 1e-9, "AP_micro mismatch vs sklearn"
    assert abs(ours["F1_macro"] - sk_f1_macro) < 1e-9, "F1_macro mismatch vs sklearn"
    assert abs(ours["F1_micro"] - sk_f1_micro) < 1e-9, "F1_micro mismatch vs sklearn"
    print("PASS: multilabel_metrics matches sklearn to 1e-9\n")

    # tile-clustered vs patch-level bootstrap width, on synthetic tiles with
    # deliberately strong within-tile correlation (so clustering should matter a lot)
    n_tiles = 20
    tile_ids = np.repeat(np.arange(n_tiles), n // n_tiles)
    tile_effect = rng.normal(0, 0.3, n_tiles)[tile_ids]  # shared per-tile noise -> within-tile correlation
    Y_prob_corr = np.clip(Y_prob + tile_effect[:, None], 0, 1)

    tile_ci = tile_clustered_bootstrap(None, Y_true, Y_prob_corr, tile_ids=tile_ids, n_boot=500)
    patch_ci = patch_level_bootstrap(Y_true, Y_prob_corr, metric_key="AP_macro", n_boot=500)
    tile_width = tile_ci["AP_macro"]["ci_upper_97.5"] - tile_ci["AP_macro"]["ci_lower_2.5"]
    patch_width = patch_ci["ci_upper_97.5"] - patch_ci["ci_lower_2.5"]
    print(f"Tile-clustered CI width:  {tile_width:.4f}  {tile_ci['AP_macro']}")
    print(f"Patch-level CI width:     {patch_width:.4f}  {patch_ci}")
    assert tile_width > patch_width, "tile-clustered CI should be wider than patch-level under within-tile correlation"
    print("PASS: tile-clustered bootstrap is wider than patch-level under spatial correlation")
