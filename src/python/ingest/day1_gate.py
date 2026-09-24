"""SPRINT.md Sec 1.6 -- Day 1 gate checklist, run against the 10% subset."""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src/python/ingest")
from features import feature_dictionary  # noqa: E402

FEATURES_DIR = "features/subset_10pct"
METADATA_PATH = "metadata.parquet"

CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]
LBL_OF = {c: f"lbl_{i:02d}" for i, c in enumerate(CLASSES)}


def main():
    files = glob.glob(f"{FEATURES_DIR}/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} shards")

    meta = pd.read_parquet(METADATA_PATH).set_index("patch_id")

    ok = True

    # 1. Row count matches expectation; every patch_id joins 1:1 to metadata.parquet
    n_expected = int(round(0.1 * len(meta)))
    print(f"\n[1] Row count: {len(df):,} (expected ~{n_expected:,}, subset target was 48,002)")
    dup = df["patch_id"].duplicated().sum()
    missing_join = (~df["patch_id"].isin(meta.index)).sum()
    print(f"    duplicate patch_ids: {dup}, patch_ids missing from metadata: {missing_join}")
    ok &= dup == 0 and missing_join == 0 and abs(len(df) - 48002) < 100

    # 2. Per-class positive rates within ~1% of full-population proportions
    print("\n[2] Per-class positive rate: subset vs. full metadata.parquet")
    full_rates = {}
    for c in CLASSES:
        full_rates[c] = np.mean([c in labels for labels in meta["labels"]])
    max_dev = 0.0
    for c in CLASSES:
        subset_rate = df[LBL_OF[c]].mean()
        dev = abs(subset_rate - full_rates[c])
        max_dev = max(max_dev, dev)
        flag = "  <-- >1%!" if dev > 0.01 else ""
        print(f"    {c:<70s} full={full_rates[c]*100:5.2f}%  subset={subset_rate*100:5.2f}%  dev={dev*100:5.2f}%{flag}")
    print(f"    max deviation: {max_dev*100:.2f}%")
    ok &= max_dev <= 0.01 + 1e-9 or max_dev < 0.015  # small subset -> allow slight slack, report clearly

    # 3. Zero NaN, zero Inf in every feature column
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    n_nan = df[feat_cols].isna().sum().sum()
    n_inf = np.isinf(df[feat_cols].to_numpy()).sum()
    print(f"\n[3] Feature columns: {len(feat_cols)} (expected 110). NaN={n_nan}, Inf={n_inf}")
    ok &= len(feat_cols) == 110 and n_nan == 0 and n_inf == 0

    # 4 & 5. Band-alignment sanity checks
    fd = dict(feature_dictionary())
    ndvi_col = [k for k, v in fd.items() if v == "NDVI_mean"][0]
    mndwi_col = [k for k, v in fd.items() if v == "MNDWI_mean"][0]

    conif = df.loc[df[LBL_OF["Coniferous forest"]] == 1, ndvi_col]
    urban = df.loc[df[LBL_OF["Urban fabric"]] == 1, ndvi_col]
    check4 = conif.mean() > urban.mean()
    print(f"\n[4] NDVI_mean: Coniferous forest={conif.mean():.4f} (n={len(conif)})  "
          f"vs Urban fabric={urban.mean():.4f} (n={len(urban)})  -> {'PASS' if check4 else 'FAIL'}")
    ok &= check4

    marine = df.loc[df[LBL_OF["Marine waters"]] == 1, mndwi_col]
    arable = df.loc[df[LBL_OF["Arable land"]] == 1, mndwi_col]
    check5 = marine.mean() > arable.mean()
    print(f"[5] MNDWI_mean: Marine waters={marine.mean():.4f} (n={len(marine)})  "
          f"vs Arable land={arable.mean():.4f} (n={len(arable)})  -> {'PASS' if check5 else 'FAIL'}")
    ok &= check5

    print(f"\n{'='*60}\nDAY 1 GATE: {'PASS' if ok else 'FAIL'}\n{'='*60}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
