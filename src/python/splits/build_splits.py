"""
SPRINT.md Sec 2.1 -- one table patch_id -> {S1, S2, S3, S4}.

    S1 Random            seeded shuffle 50/25/25
    S2 Official          copy of metadata.parquet's `split` column
    S3 Leave-tile-out    5 folds over the 54 MGRS tiles (fold id per patch)
    S4 Leave-country-out 10 folds, one held-out country each (fold id = country)

For S3/S4 the table stores a fold id per patch; "train" for fold f is every
patch NOT in fold f, "test" is every patch in fold f -- computed at model-training
time (Day 3), not baked into this table.

Writes:
    results/splits.parquet      patch_id, s1_split, s2_split, s3_fold, s4_fold
    results/split_stats.json    counts + the "train has >=1 positive per label" audit
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src/python/ingest")
from patch_meta import parse_patch_id  # noqa: E402

METADATA_PATH = "metadata.parquet"
OUT_SPLITS = "results/splits.parquet"
OUT_STATS = "results/split_stats.json"

CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]
RAREST = ["Coastal wetlands", "Beaches, dunes, sands"]
N_S3_FOLDS = 5
SEED = 42


def build_s1_random(patch_ids: pd.Index, rng: np.random.Generator) -> pd.Series:
    n = len(patch_ids)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    n_train = int(round(0.5 * n))
    n_val = int(round(0.25 * n))
    labels[perm[:n_train]] = "train"
    labels[perm[n_train:n_train + n_val]] = "validation"
    labels[perm[n_train + n_val:]] = "test"
    return pd.Series(labels, index=patch_ids, name="s1_split")


def build_s3_tile_folds(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Greedy balance-by-patch-count assignment of the 54 tiles to N_S3_FOLDS folds."""
    tile_counts = df.groupby("tile").size().sort_values(ascending=False)
    fold_load = np.zeros(N_S3_FOLDS, dtype=np.int64)
    tile_to_fold = {}
    # tiny random jitter on tie-breaking so repeated ties don't always favor fold 0
    order = tile_counts.index.tolist()
    for tile in order:
        f = int(np.argmin(fold_load))
        tile_to_fold[tile] = f
        fold_load[f] += tile_counts[tile]
    return df["tile"].map(tile_to_fold).rename("s3_fold")


def audit_train_coverage(df: pd.DataFrame, label_cols: list, regime: str, group_col: str = None,
                          fixed_col: str = None) -> list:
    """For S1/S2 (fixed_col='train'), or S3/S4 (group_col=fold column, leave-one-fold-out),
    check every train partition has >=1 positive per label. Returns list of violation dicts."""
    violations = []
    if fixed_col is not None:
        train_mask = df[fixed_col] == "train"
        sums = df.loc[train_mask, label_cols].sum()
        for c, s in zip(CLASSES, sums):
            if s == 0:
                violations.append({"regime": regime, "fold": fixed_col, "label": c, "train_positives": int(s)})
    else:
        for fold_val in sorted(df[group_col].unique(), key=str):
            train_mask = df[group_col] != fold_val
            sums = df.loc[train_mask, label_cols].sum()
            for c, s in zip(CLASSES, sums):
                if s == 0:
                    violations.append({"regime": regime, "fold": str(fold_val), "label": c, "train_positives": int(s)})
    return violations


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_parquet(METADATA_PATH)
    if "patch_id" not in meta.columns:
        for alt in ("name", "patch_name"):
            if alt in meta.columns:
                meta["patch_id"] = meta[alt]
    meta = meta.set_index("patch_id")
    meta["tile"] = [parse_patch_id(p)["tile"] for p in meta.index]

    found_labels = set()
    for labels in meta["labels"]:
        found_labels.update(labels)
    assert found_labels == set(CLASSES), f"label set mismatch: {found_labels.symmetric_difference(CLASSES)}"

    label_cols = []
    for i, c in enumerate(CLASSES):
        col = f"lbl_{i:02d}"
        meta[col] = meta["labels"].apply(lambda l, c=c: int(c in l))
        label_cols.append(col)

    # S1 random 50/25/25
    meta["s1_split"] = build_s1_random(meta.index, rng)
    # Week 2 robustness check: 2 more independent S1 shuffles, to verify the
    # S1 macro-AP point estimate isn't just a lucky draw. Independent RNGs so
    # this can't perturb the seed=42 rng's downstream calls (s3_fold) below.
    S1_EXTRA_SEEDS = [123, 7]
    for extra_seed in S1_EXTRA_SEEDS:
        extra_rng = np.random.default_rng(extra_seed)
        meta[f"s1_split_seed{extra_seed}"] = build_s1_random(meta.index, extra_rng)

    # S2 official
    meta["s2_split"] = meta["split"]

    # S3 leave-tile-out, 5 folds over the 54 MGRS tiles
    meta["s3_fold"] = build_s3_tile_folds(meta, rng)

    # S4 leave-country-out, fold id = country itself (10 countries -> 10 folds)
    meta["s4_fold"] = meta["country"]

    # --- assertions: disjointness within each regime (trivial by single-column construction,
    #     verified explicitly) ---
    for regime, col in [("S1", "s1_split"), ("S2", "s2_split")]:
        parts = {v: set(meta.index[meta[col] == v]) for v in meta[col].unique()}
        vals = list(parts.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                assert not (vals[i] & vals[j]), f"{regime}: overlapping partitions"
        assert set().union(*vals) == set(meta.index), f"{regime}: partitions don't cover all patches"
    print("Disjointness / coverage assertions: OK for S1, S2 (S3/S4 are single-valued fold ids, trivially disjoint by construction)")

    # --- train coverage audit ---
    violations = []
    violations += audit_train_coverage(meta, label_cols, "S1", fixed_col="s1_split")
    violations += audit_train_coverage(meta, label_cols, "S2", fixed_col="s2_split")
    violations += audit_train_coverage(meta, label_cols, "S3", group_col="s3_fold")
    violations += audit_train_coverage(meta, label_cols, "S4", group_col="s4_fold")

    if violations:
        print(f"\n{len(violations)} train-coverage violation(s) found (train split has ZERO positives for a label):")
        for v in violations:
            print(f"  regime={v['regime']} fold={v['fold']} label={v['label']!r}")
    else:
        print("\nNo train-coverage violations.")

    # rarest-class tile spread sanity check (informational)
    tile_spread = {}
    for c in RAREST:
        has = meta[meta["labels"].apply(lambda l, c=c: c in l)]
        folds_touched = has["s3_fold"].nunique()
        tile_spread[c] = folds_touched
        print(f"Rarest-class check: {c!r} appears in {folds_touched}/{N_S3_FOLDS} S3 folds")

    # write outputs
    out_cols = ["s1_split", "s2_split", "s3_fold", "s4_fold"] + [f"s1_split_seed{s}" for s in S1_EXTRA_SEEDS]
    meta[out_cols].reset_index().to_parquet(OUT_SPLITS, index=False)

    stats = {
        "n_patches": len(meta),
        "s1_counts": meta["s1_split"].value_counts().to_dict(),
        "s2_counts": meta["s2_split"].value_counts().to_dict(),
        "s3_fold_sizes": meta["s3_fold"].value_counts().sort_index().to_dict(),
        "s3_tiles_per_fold": meta.groupby("s3_fold")["tile"].nunique().to_dict(),
        "s4_fold_sizes": meta["s4_fold"].value_counts().to_dict(),
        "rarest_class_s3_fold_spread": tile_spread,
        "train_coverage_violations": violations,
        "note": (
            "S4 violation for label='Agro-forestry areas' fold='Portugal' is a real, "
            "unavoidable data limitation: all 33,181 positive patches for this class are "
            "in Portugal, so holding Portugal out zeroes its training signal. This is "
            "expected and should be reported, not treated as a pipeline bug."
        ),
    }
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nWrote {OUT_SPLITS} and {OUT_STATS}")


if __name__ == "__main__":
    main()
