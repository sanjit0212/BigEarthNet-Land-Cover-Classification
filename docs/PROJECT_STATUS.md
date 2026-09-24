# BigEarthNet-S2 Land Cover — Project Status

**Last updated:** 2026-09-20 (Week 1 of the 3-week strengthening plan)
**Actual timeline (supersedes the dates below):** viva/submission is second
week of October 2026, with weekly progress updates due. The original 5-day
`SPRINT.md` calendar (deadline 2026-09-20) is kept below as historical record
of Days 1-3, but the project is no longer running on that calendar — see
`C:\Users\sanji\.claude\plans\compare-my-work-with-concurrent-thompson.md`
for the current 3-week plan (Week 1: fix analysis + fill the S1-S4 ladder;
Week 2: model-capacity robustness; Week 3: figures/write-up/viva prep),
targeting an IGARSS short paper post-viva.
**Governing docs:** `SPRINT.md` (the original 5-day plan), `PLAN.md` (full
8-week reference), the strengthening plan above (current source of truth).

This file is a from-scratch account of everything done in this session, why it
was done that way, what is verified vs. assumed, and exactly what is left. It
exists so the project can be picked up cold — by you, or a fresh session —
without re-deriving any of this.

---

## 0. Week 1 of the strengthening plan (2026-09-20) — COMPLETE

Triggered by asking "is my accuracy bad?" and then "compare my work to recent
IEEE papers on this dataset" — research confirmed the split-optimism audit is
genuinely unpublished novel work (see the plan doc's literature section), but
found two real flaws in how the S1-S4 gap was being computed, plus identified
the decisive control needed to rule out the obvious reviewer objection
("your gap is just label-prior shift"). All of that is now fixed and re-run.

### 0.1 Two bugs fixed in `compute_optimism_gap.py`

- **Class-set mismatch (real, verified):** the old code silently skipped
  classes with zero test positives per fold, so S2's macro-AP averaged over
  19 classes while individual S4 folds averaged over 13-18 (worst:
  Luxembourg/Kosovo). The skipped classes were disproportionately the hard
  rare ones, so old S4 numbers were biased **upward** and the reported gap
  was **understated**. Fixed via class-major aggregation (average within each
  class across folds first, skipping only that class's missing cells, then
  macro-average across classes) — confirmed the fix moved the gap in the
  predicted direction (8.15 -> 8.33 points, `S2_to_S4_AP`).
- **Pooled-AP confound:** pooling raw probabilities from 10 different
  per-fold classifiers into one ranking conflated discrimination loss with
  cross-model score incommensurability. Demoted from headline to a labeled,
  caveated secondary field (`secondary_caveated` in `results/optimism_gap.json`).
- Also added: macro-AUROC (prevalence-invariant, independent evidence against
  pure prior-shift), the "S4 trains on 37-101% more data than S2 and still
  loses" fact, and explicit 18-class-common-set (primary) / 19-class
  Portugal-imputed (sensitivity) handling of the Agro-forestry structural
  confound instead of silently dropping it.

### 0.2 S3 (leave-tile-out) finally run — completes the ladder

Built in Day 2 (`build_splits.py`) but never executed until now: 5 Spark
jobs, LR, full 480,038 patches, ~4.3-5 min/fold (`scripts/run_s3_audit.sh`).
**The corrected, complete S1->S2->S3->S4 ladder (18-class common set, macro-AP):**

| Regime | Macro-AP | 95% CI (tile-clustered bootstrap) | n_tiles feeding CI |
|---|---|---|---|
| S1 (random) | **50.75** | [48.04, 52.76] | 54 |
| S2 (official) | **46.05** | [43.81, 48.35] | 46 |
| S3 (leave-tile-out, fold-avg) | **45.16** | wide per-fold (see below) | 10-11/fold |
| S4 (leave-country-out, fold-avg) | **37.72** | between-fold spread only (see 0.1) | n/a |

S3 sits just below S2 (only a **0.89-point** drop) while S4 drops a further
**7.44 points** below S3 — this is a clean, quotable finding: leave-tile-out
(removing only *adjacency* leakage) barely moves the number, while
leave-country-out (removing *regional* leakage) is where almost all the real
optimism lives. `AUROC` gap S2->S4 is **11.04 points**, persisting evidence
against pure prior-shift.

**Individual S3 fold CIs are much wider than S1/S2's** (e.g. fold0: 95% CI
[35.55, 51.91], a 16-point span, vs S2's [43.81, 48.35], 4.5 points) because
each fold only has 10-11 MGRS tiles feeding the tile-clustered resample vs
S1's 54 / S2's 46 — so the S3 **fold-average** (45.16) is far more trustworthy
than any single fold's point estimate. Both S1's and S2's CIs clearly exclude
S4's point estimate, so the S2->S4 gap is statistically well-supported even
accounting for tile-clustering uncertainty.

### 0.3 New: same-region paired evaluation (`paired_country_eval.py`) — the decisive prior-shift control

For each of the 10 countries, the S2 model (trained *with* that country) and
the matching S4-fold model (trained *without* it) are scored on the
**identical** patch set (verified: all of S2's test patches in a country are
a subset of that country's S4-fold test set). Because both models see the
same patches, label-prior shift is eliminated **by construction**, not
corrected post hoc — this directly answers "isn't your gap just because the
class distribution changed?"

- Median delta-AP (in-training minus held-out) = **1.11 points**, Wilcoxon
  signed-rank p = **0.002** (n=10, the minimum attainable two-sided p at this
  sample size) — real and significant, but small in the median because 4 of
  10 countries are tiny (Belgium/Switzerland/Luxembourg/Kosovo, each <1% of
  total data) and excluding them barely changes the model at all.
- **Unexpected, genuinely interesting secondary finding:** the paired penalty
  scales almost perfectly with how much of the total training distribution
  the held-out country represents — Spearman rho = **0.939**, p = 5.5e-05.
  Finland (32% of all data) loses 12.6 AP points when excluded; Kosovo (0.3%)
  loses 0.07. The **patch-count-weighted mean delta (6.16)** is much closer
  to the naive S2->S4 gap (8.33) than the unweighted median (1.11) is — this
  explains the discrepancy mechanistically rather than leaving it as an
  unexplained anomaly. This is genuinely new analysis, not in the original
  plan when it was drafted, and reads as a strength (a mechanistic
  explanation, not just a bigger number) rather than a hedge.

### 0.4 New: `metrics.py` — PLAN.md-specified library, sklearn-validated

Never written until now despite being specified since the 8-week plan.
Provides `multilabel_metrics()` (AP/AUROC/F1 macro+micro, Hamming loss,
subset accuracy, per-class breakdowns) and `tile_clustered_bootstrap()`.
Self-test (`python metrics.py`) validates against sklearn's own
`average='macro'/'micro'` to 1e-9, and demonstrates the tile-clustered CI is
wider than a naive patch-level bootstrap under synthetic within-tile
correlation (8.6x wider in the demo) — the concrete justification for why
patch-level bootstrapping would have been wrong for this spatially
autocorrelated data.

### 0.5 Two real bugs hit and fixed *during* this work (own up to these in the viva if asked "did everything just work?")

- **Windows multiprocessing memory exhaustion:** `tile_clustered_bootstrap()`
  originally defaulted to `cpu_count()-2` (14) worker processes for
  parallelizing the 1000-replicate bootstrap. Each spawned worker re-imports
  the full numpy/scipy/sklearn/pandas stack from scratch (Windows has no
  fork/copy-on-write), and with the Spark/HDFS Docker containers also
  running, one worker crashed with `ImportError: DLL load failed... paging
  file too small`. Worse, that failure mode doesn't reliably raise a
  catchable exception from `Pool.map()` — it can just hang (this run needed
  a manual kill after several minutes of silence). **Fixed:** default
  workers reduced to 4; switched to `pool.map_async().get(timeout=...)` so a
  hang becomes a catchable `TimeoutError`; added a sequential-computation
  fallback so a multiprocessing failure degrades gracefully instead of
  losing the run.
- **Silent data loss on partial failure:** `run_bootstrap_cis.py` only wrote
  `results/bootstrap_cis.json` after *all* regimes finished, so the crash
  above would have discarded every regime already computed. Fixed to save
  after each regime and resume from a partial file on rerun.

### 0.6 Threading-tuning perf bug (also fixed, smaller): O(n²) threshold search

Not part of the strengthening plan but hit along the way in Day 3's
`threshold_tuning.py`: `tune_threshold()` originally rescanned every unique
probability value and recomputed `f1_score` from scratch at each candidate —
O(n x n_unique) per label, unusable above ~100k rows. Rewritten to a single
`precision_recall_curve` pass (O(n log n)), verified correct (reproduces the
same optimal threshold/F1) and ~100x+ faster in practice.

### 0.7 Files touched this week

```
src/python/eval/compute_optimism_gap.py    REWRITTEN (class-major aggregation, AUROC, sensitivity analysis)
src/python/eval/paired_country_eval.py     NEW (same-region paired control + data-share correlation)
src/python/eval/metrics.py                 NEW (PLAN.md library, sklearn-validated, tile-clustered bootstrap)
src/python/eval/run_bootstrap_cis.py       NEW (incremental-save driver for the CIs above)
scripts/run_s3_audit.sh                    NEW (the 5 S3 Spark jobs, finally run)
results/predictions/s3_fold{0-4}_lr_full/  NEW (pulled from HDFS)
results/optimism_gap.json                  REGENERATED (corrected class-major numbers)
results/paired_country_eval.json           NEW
results/bootstrap_cis.json                 NEW
```

### 0.8 What's left after Week 1 (see §1 below, now superseded — Week 2 progress)

---

## 0a. Week 2 of the strengthening plan (2026-09-22) — RF ladder + seed robustness DONE

### 0a.1 `--seed` flag added to `BinaryRelevanceRunner.scala`

RF's `.setSeed(42)` was hard-coded; now a `--seed` CLI arg (default 42, so
every existing seed=42 run's HDFS output path is unchanged). Non-default
seeds get a `_seed{N}` suffix on the output path so seed sweeps don't
overwrite each other. Also threaded into the compute-stats JSON for
traceability.

### 0a.2 RF ladder — S1, S3 (5 folds), S4 (10 countries) — 16 jobs, all complete

RF previously only existed for S2 (Table 1). Ran the full ladder
(`scripts/run_rf_ladder.sh`, resumed via `run_rf_ladder_resume.sh` after two
mid-run Docker Desktop crashes — see §0a.4). **Regime x learner grid
(macro-AP, 18-class common set):**

| Regime | LR | RF | RF−LR |
|---|---|---|---|
| S1 (random) | 50.75 | 61.49 | +10.74 |
| S2 (official) | 46.05 | 56.48 | +10.43 |
| S3 (leave-tile-out) | 45.16 | 50.75 | +5.59 |
| S4 (leave-country-out) | 37.72 | 39.86 | +2.14 |
| **S2->S4 gap** | **8.33 pts** | **16.62 pts** | |

**This is a genuinely strong result, not just a robustness check that passed:**
RF's optimism gap is roughly **double** LR's (16.62 vs 8.33 points). The more
expressive model fits regional/spatial signal harder when the region is known
(RF beats LR by ~10 points on S1/S2) but that advantage nearly vanishes on a
truly unseen country (RF beats LR by only ~2 points on S4). **Increased model
capacity increases the measured optimism, not decreases it** -- the opposite
of what "your gap is a weak-linear-model artifact" would predict, and a
sharper warning for anyone applying CNNs/transformers to this benchmark.
AUROC confirms it: S2->S4 AUROC gap is 13.83 points (RF) vs 11.04 (LR).

### 0a.3 Seed robustness -- both checks pass decisively

Per the plan's minimal scope ("report SD once", not a full ladder x 3 seeds):

- **S1 split seed** (does the "optimistic" 50.75 baseline depend on which
  random shuffle you got?): 2 more independent 50/25/25 shuffles added to
  `build_splits.py` (`s1_split_seed123`, `s1_split_seed7`, generated from
  fresh independent RNGs so the seed=42 `s1_split`/`s3_fold` columns are
  byte-identical to before). `BinaryRelevanceRunner.scala`'s `roleColumn`
  extended to select `s1_split_<fold>` when `--fold` is non-empty for
  regime=s1, reusing existing plumbing instead of a new flag.
  **Result: macro-AP = 51.22, 51.01, 51.06 -> SD = 0.11 points**, against an
  8.33-point gap (SD is ~1.3% of the effect).
- **RF training seed** (is the doubled RF gap just training noise?): S2 rerun
  with `--seed 123` and `--seed 7`. **Result: macro-AP = 56.48, 57.20, 56.81
  -> SD = 0.36 points**, against a 16.62-point gap (SD is ~2.2% of the
  effect, i.e. the real effect is ~46x the noise floor).

Both anticipated reviewer objections ("lucky shuffle", "training noise") are
now closed with data. Full numbers in `results/seed_robustness.json`.

### 0a.4 Real problems hit and fixed this week

- **Two more Docker Desktop crashes during the 16-job RF ladder**, on top of
  the Week 1 multiprocessing crash. First crash: killed mid-Ireland after the
  session was paused for hours (machine likely slept); second: died again
  seconds into the Austria resume attempt, before training even started.
  Docker Desktop on this machine is not reliable across long unattended runs
  or system sleep -- **budget for manual restarts when running multi-hour
  Spark batches**, and always verify partial output (`hdfs dfs -ls` label
  count, expect 19) before trusting a "completed" job after any interruption.
  Austria's first attempt left only 6/19 labels written; `.mode("overwrite")`
  in the Scala job made the clean rerun trivial, but it had to be checked for
  explicitly, not assumed.
- **`compute_optimism_gap.py` wrote to a fixed `results/optimism_gap.json`
  regardless of `--learner`** -- running it for RF silently overwrote Week
  1's LR results (no data was actually lost; predictions were still on disk
  and the numbers were already documented above, but the JSON artifact was
  gone). Fixed: output is now `results/optimism_gap_{learner}.json`; both
  regenerated and confirmed to match their previously-reported values exactly.

### 0a.5 Files touched this week

```
src/main/scala/multilabel/BinaryRelevanceRunner.scala   MODIFIED (--seed flag, seed-suffixed output paths, s1 fold->column selector)
src/python/splits/build_splits.py                       MODIFIED (2 extra independent S1 shuffles)
src/python/eval/compute_optimism_gap.py                 MODIFIED (learner-specific output filename)
scripts/run_rf_ladder.sh                                NEW
scripts/run_rf_ladder_resume.sh                          NEW
scripts/run_seed_robustness.sh                           NEW
results/predictions/{s1,s3_fold0-4,s4_<country>}_rf_full/   NEW (16 dirs, pulled from HDFS)
results/predictions/s1_seed{123,7}_lr_full/              NEW
results/predictions/s2_rf_seed{123,7}_full/              NEW
results/optimism_gap_lr.json, results/optimism_gap_rf.json   REGENERATED (learner-specific)
results/seed_robustness.json                             NEW
```

### 0a.6 Week 2 remainder — DONE: prior-matched resampling, hyperparameter tuning, retuned LR ladder

**Prior-matched resampling** (`src/python/eval/prior_matched_resampling.py`):
per (class, S4 country) cell, subsample whichever of positives/negatives is
in excess so the fold's positive rate matches S2's, recompute AP, average
over 100 draws. Result: S4 macro-AP raw 37.72 -> prior-matched 37.05 -- the
gap **grows slightly** (8.33 -> 8.99 points) under prior-matching. Prior
shift explains **none** of the gap (a negative "-8.0% explained"), confirming
the paired same-region control's conclusion via a completely independent
method.

**Hyperparameter tuning** (`--reg-param`/`--num-trees`/`--max-depth` flags
added to `BinaryRelevanceRunner.scala`; grid run on S2, scored on
**validation only**, never test): RF's defaults (numTrees=50, maxDepth=10)
were already near-optimal (60.89% vs 60.93% for numTrees=100, confirmed;
maxDepth=15 clearly overfits at 56.49%) -- kept as-is. **LR's default
regParam=0.01 was genuinely suboptimal** -- regParam=0.001 beat it by 2.93
points on validation (53.26% vs 50.33%).

**Retuned LR ladder** -- given the finding was real, not noise, reran the
full LR ladder (16 jobs: S1, S3x5, S4x10) at regParam=0.001 rather than
document-and-ignore it:

| Regime | LR untuned (reg=0.01) | LR tuned (reg=0.001) |
|---|---|---|
| S1 | 50.75 | 53.42 |
| S2 | 46.05 | 48.71 |
| S3 | 45.16 | 46.51 |
| S4 | 37.72 | 38.02 |
| **S2->S4 gap** | **8.33 pts** | **10.70 pts** |

**This is the same pattern as the RF finding, now confirmed twice over:** the
tuned model is better everywhere (+2.7-3 points on S1/S2/S3) but S4 barely
moves (+0.30) -- so the gap **widens**, not narrows, once the model is
properly tuned. Combined with the RF-vs-LR result (§0a.2), both "your
baseline was undertuned" and "your gap is a weak-model artifact" objections
are now closed with data. AUROC S2->S4 gap under tuning: 11.63 points (up
from 11.04). Full numbers: `results/optimism_gap_lr_tuned.json`,
`results/prior_matched_resampling.json`, `results/hp_tuning.json`.

### 0a.7 Infrastructure note: two more incidents this week, both resolved

- A Spark job hung for ~2h unable to get worker resources accepted while
  Docker Desktop's API was silently returning 500 errors -- required a full
  `docker desktop restart` and verifying "Alive Workers: 1" on the Spark UI
  before relaunching.
- Docker's WSL2 data disk (`docker_data.vhdx`) grew to **104GB** on the
  Windows host (down to ~6GB actually used per `docker system df` --
  deleted-but-not-trimmed blocks in the sparse virtual disk), dropping free
  space on C: to 8.7GB. Fixed via Docker Desktop's "Reset to factory
  defaults" (a manual `compact vdisk` via diskpart hit an unrelated stale
  Virtual Disk Service lock requiring a reboot, then still didn't reclaim
  space without an `fstrim` inside WSL2 first -- the factory reset was
  faster and more reliable than either). **All code, docs, and every
  aggregated JSON/figure result are now pushed to GitHub**
  (`github.com/sanjit0212/BigEarthNet-Land-Cover-Classification`) as of this
  update, specifically so a future Docker/WSL crash or disk issue cannot
  cost more than the currently in-flight job -- raw per-run prediction
  parquets stay local/HDFS-only (regenerable by rerunning the Scala job
  against the features+splits, which are themselves in the repo or
  regenerable from the archive) and are excluded via `.gitignore`.

### 0a.8 What's left (Week 3)

The paired slope/dumbbell figure (paper's headline image), F3 + dashboard
for the college submission, the write-up, and viva prep.

---

## 1. What this project is

Multi-label land cover classification on BigEarthNet-S2 (reBEN): 480,038
Sentinel-2 patches, 19 possible land-cover classes per patch (avg 2.95 labels/
patch, max 11), 10 European countries, 54 distinct Sentinel-2 MGRS tiles.

The **headline contribution** (per `SPRINT.md` §5.1/§5.2) is not raw accuracy
(a CPU-trained binary-relevance model with hand-crafted features will not beat
a published ResNet-50 CNN). It is an **audit**: the same cheap, reproducible
model trained and evaluated under four split geometries of increasing
strictness (S1 random → S2 official → S3 leave-tile-out → S4
leave-country-out), to measure how much of reBEN's published accuracy is
optimism from spatial autocorrelation rather than genuine generalisation. The
distributed-systems stack (streaming ingestion, HDFS, Spark) is the enabling
infrastructure that makes running the audit's ~17 training runs affordable on
8 cores / 16GB RAM, not the contribution itself.

A legacy single-label baseline (`src/python/0_create_subset_from_archive.py`,
`src/main/scala/LandCoverClassifier.scala`) already existed in the repo before
this session — a heavy RandomForest (numTrees=100, maxDepth=20) on a 40k
dominant-label subset using only 4 of 12 spectral bands. It is kept as-is for
comparison/history; all new work below is a separate, parallel pipeline.

---

## 2. Day 1 — data pipeline (COMPLETE, gate PASSED)

### 2.1 Archive verification — `src/python/ingest/verify_archive.py`

Streams `BigEarthNet-S2.tar.zst` (63GB compressed, never extracted to disk:
`zstandard.stream_reader` → `tarfile.open(mode='r|')`) and confirms the real
layout rather than assuming it. **Full-archive scan result (ground truth):**

- Path layout: `BigEarthNet-S2/<acquisition_folder>/<patch_id>/<patch_id>_B{01,02,03,04,05,06,07,08,8A,09,11,12}.tif`, depth 4, confirmed.
- Band resolution/dtype, all confirmed against real bytes:
  10m (B02,B03,B04,B08)=120×120, 20m (B05,B06,B07,B8A,B11,B12)=60×60, 60m (B01,B09)=20×20, all `uint16`.
- **549,488** total patches in the archive.
- **480,038** overlap exactly with `metadata.parquet` (perfect 1:1 join, 0 missing).
- **69,450** archive-only patches (snow/cloud patches excluded from metadata) — correctly never extracted.
- **115** unique acquisition folders (this is the "tile" checkpoint granularity used by the extractor — see §2.3; distinct from the 54 MGRS grid tiles used for the S3 split).

Output written: `docs/archive_layout.md`.

### 2.2 Patch ID parsing — `src/python/ingest/patch_meta.py`

`patch_id` format `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57` decodes
to `{tile: T33UUP, gx: 26, gy: 57, year: 2017, month: 6}`. Regex-validated
against **all 480,038** metadata patch_ids — 0 failures. Confirms exactly
**54 unique MGRS tiles**, matching `SPRINT.md`'s split-audit assumption.

### 2.3 Streaming extractor — `src/python/ingest/stream_extract.py`

Architecture (matches `SPRINT.md` §1.3 exactly):
`reader (main process, owns the tar/zstd stream)` → bounded `multiprocessing.Queue(maxsize=512)` → `N-1 worker processes` (feature computation) → `result queue` → `writer thread` (groups rows by acquisition folder, writes one Parquet shard per folder, checkpoints to `state/completed_tiles_{full,frac10}.json`).

- CLI: `--subset-fraction`, `--seed` (stratified by MGRS tile via `patch_meta`), `--workers`, `--queue-maxsize`, `--batch-size`.
- Resume-safe: a completed acquisition folder is skipped on rerun (checkpoint keyed by folder name; separate checkpoint files for subset vs. full runs so they can't cross-contaminate).
- **Bug found and fixed during this session:** the writer only re-checked "is this folder complete?" when a new prediction row arrived. If a folder's expected-patch-count message arrived *after* its last row had already landed in the buffer, the flush was silently deferred to a shutdown-time fallback (which did still write correct data, but printed alarming `WARNING: never reached its expected count` lines even though the counts matched, e.g. `619/619`). Root cause: the flush check lived only in the "row received" branch. **Fix:** also flush-check immediately when the expected-count message itself is drained (`stream_extract.py` lines ~96–109). Verified with a fresh multi-folder run afterward — zero warnings.

### 2.4 Feature extraction — `src/python/ingest/features.py`

Pure numpy, no I/O. **110 features per patch**, self-tested (`python features.py` asserts exactly 110):

- **(a) 72** — 12 bands × {mean, std, p10, p50, p90, iqr}
- **(b) 26** — 13 spectral indices × {mean, std}: NDVI, EVI, SAVI, NDWI, MNDWI, NDMI, NDBI, BSI, NBR, and the four Sentinel-2-specific red-edge indices (NDRE1/2/3, CIre) that justify calling the feature set "physics-informed" (the legacy 10-feature pipeline didn't have these).
- **(c) 12** — cheap texture: std/p90 of 3×3 local variance, 7×7 local variance, and Sobel gradient magnitude, for B08 and B04 (`scipy.ndimage.uniform_filter`/`sobel`, **not** `skimage.graycomatrix` — GLCM was cut for cost per `SPRINT.md` §0.2).
- All divisions guarded (`denom = np.where(|d|<1e-6, 1e-6, d)`); `extract_features()` raises on any non-finite value rather than propagating it silently.
- `docs/feature_dictionary.md` — the frozen `f_000`..`f_109` → human-name mapping, machine-generated so it can never drift from the actual code.

20m/60m bands are upsampled to 120×120 via nearest-neighbour (`scipy.ndimage.zoom`, order=0) before feature computation.

### 2.5 Day-1 gate — `src/python/ingest/day1_gate.py` — **ALL 5 CHECKS PASS** (run against the real 48,002-patch 10% stratified subset, not synthetic data)

1. Row count 48,002, 0 duplicate patch_ids, 0 patch_ids unmatched to metadata.
2. Per-class positive rate: max deviation from full-population proportion = **0.21%** (well under the 1% bar).
3. 110 feature columns present, **0 NaN, 0 Inf**.
4. `NDVI_mean`: Coniferous forest **0.619** > Urban fabric **0.560** — PASS.
5. `MNDWI_mean`: Marine waters **0.544** > Arable land **−0.493** — PASS (this is the decisive check: it rules out band misalignment/swapping, which would otherwise produce a silently-broken-but-runs-fine pipeline).

### 2.6 Full extraction — **COMPLETE**

`state/completed_tiles_full.json` covers all 115 acquisition folders.
`features/full/*.parquet` — **115 shards, 229MB total, 480,038 rows** (verified
by the writer's own completion log: `total patches written this run: 480,038`,
matching the archive's exact metadata-matched patch count). Ran with Docker
**stopped** (see §3 Rule 1 below): 2924s (~49 min) at ~164 patches/sec — about
2.3× faster than the same code run with the Spark/HDFS cluster also up
(~70 patches/sec observed during an earlier, since-corrected run), confirming
`SPRINT.md`'s explicit warning not to run both simultaneously.

**This data has NOT yet been pushed to HDFS.** Only the 10% subset is
currently in HDFS (see §4). Pushing `features/full/*.parquet` to
`/bigearthnet/features/` in HDFS is the first Day-3 action.

---

## 3. Hardware/cluster configuration (fixed during this session)

The repo's original `docker-compose.yml` and `hadoop.env` did **not** match
`SPRINT.md` §0.0's 8-core/16GB budget — they provisioned 3 datanodes (RF=3)
and a 2-core/4GB Spark worker. Both were corrected:

- `docker-compose.yml`: dropped the 3rd datanode (2 datanodes is enough at RF=2 for 500MB-scale feature data); `spark-worker` now `SPARK_WORKER_CORES=6`, `SPARK_WORKER_MEMORY=6G` (leaves 2 cores for the OS).
- `hadoop.env`: `HDFS_CONF_dfs_replication` changed from 3 → **2**, matching the 2 provisioned datanodes.
- Both containers (`namenode`, `spark-master`) also got a new bind mount `<repo>:/workspace` so `hdfs dfs -put`/`-get` and `spark-submit` can reach local build artifacts and feature files directly (the pre-existing `bigearthnet_subset` mount only covered the legacy 4-band pipeline's output).
- **Rule 1 is being actively followed**: the cluster is brought down (`docker compose down`, not just stopped) before any extraction run, and back up only for HDFS/Spark work. Named volumes (`bda_project_hadoop_namenode/datanode/datanode2`) persist across `down`/`up`, so HDFS state survives.
- **Windows/Git-Bash gotcha discovered and worked around:** any `docker exec ... hdfs dfs -...` command with a Unix-style absolute path (e.g. `/bigearthnet/...`) gets silently mangled by MSYS path conversion into a bogus Windows path (`C:/Program Files/Git/bigearthnet/...`), producing a cryptic `No FileSystem for scheme "C"` error. Fix: prefix every such command with `MSYS_NO_PATHCONV=1`. All commands in this project that touch HDFS/Docker paths need this prefix on this machine.

**Current state (as of this file being written): Docker containers are DOWN**
(`docker compose down` was run before the full extraction, per Rule 1, and
have not been brought back up since — **per your explicit instruction, they
are being left down and will not be started without being asked**). The named
volumes still hold whatever was in HDFS as of the last `up`: the 10% subset
features (115 shards) and `results/splits.parquet`, plus one validation run's
worth of S2/LR predictions and compute stats (see §4.3).

---

## 4. Day 2 — splits, cluster, and the Spark training job (COMPLETE)

### 4.1 Split tables — `src/python/splits/build_splits.py`

Single table (`results/splits.parquet`, all 480,038 patches — this already
covers the **full** population, it does not need to be rerun for Day 3) with
columns `s1_split`, `s2_split`, `s3_fold`, `s4_fold`:

- **S1 random**: seeded (42) shuffle, 50/25/25 → 240,019 / 120,010 / 120,009.
- **S2 official**: copy of `metadata.parquet`'s `split` column → 237,871 / 122,342 / 119,825 (train/val/test).
- **S3 leave-tile-out**: 5 folds over the 54 MGRS tiles, greedy balance-by-patch-count assignment → fold sizes 95,896–96,121 (very even), 10–11 tiles/fold. **Both rarest classes (Coastal wetlands, Beaches/dunes/sands) confirmed present in all 5 folds** — no fold's train set is starved of them.
- **S4 leave-country-out**: fold id = country name directly (10 folds, sizes = the natural country sizes, Finland 155,227 down to Kosovo 1,616).

`s3_fold`/`s4_fold` store a fold id per patch; "train" = everything **not**
in the held-out fold, computed at training time (the Scala job's
`roleColumn()`), not baked into the table.

**Audit finding (real, not a bug) — `results/split_stats.json`:** exactly one
train-coverage violation across all 17 possible train partitions (S1, S2, 5×S3
folds, 10×S4 folds): **"Agro-forestry areas" has zero training positives when
Portugal is the held-out S4 country**, because all 33,181 positive patches for
that class are in Portugal — it appears in no other country at all. This is a
genuine, structural limitation of leave-one-country-out evaluation with only
10 countries, documented in `split_stats.json`'s `note` field, and is good
viva material for Q3/Q8 ("which classes fail, and why").

### 4.2 HDFS setup

`/bigearthnet/{features,splits,results}` created, replication set to 2. The
10% subset (115 shards) and `results/splits.parquet` were `put` into HDFS and
their replication corrected to 2 (they were uploaded before the `hadoop.env`
fix was applied and had inherited the old default of 3).

### 4.3 Scala Binary Relevance job — `src/main/scala/multilabel/BinaryRelevanceRunner.scala`

Plain `object`, no `Estimator`/`Params`/`MLWritable` (per `SPRINT.md` §2.3
scope cut). Note the file lives at `src/main/scala/multilabel/...`, **not**
the literal `src/scala/main/scala/multilabel/...` path written in
`SPRINT.md` §2.3 — that path has a typo (doubled `scala/`) and doesn't match
standard sbt layout; `src/main/scala/` is what `build.sbt` actually compiles.

Pipeline: read features + splits from HDFS → join → `VectorAssembler(f_000..f_109)` → `StandardScaler` (fit on **train only**, to avoid leaking val/test statistics — a small deliberate improvement over the sprint's literal wording) → `persist(MEMORY_AND_DISK)` once → for each of 19 labels, **in parallel** over a fixed thread pool (`Executors.newFixedThreadPool`, default size 6), train a weighted (inverse-class-frequency / "balanced": `n / (2·n_pos)` and `n / (2·n_neg)`) LogisticRegression (`maxIter=50, regParam=0.01, standardization=true`) or RandomForest (`numTrees=50, maxDepth=10`) and write per-label probability Parquet. A `SparkListener` accumulates `executorCpuTime` across all tasks and writes `results/compute/<run_id>.json` (wall-clock vs. total executor CPU time — the evidence for the parallel-training speedup claim). Runtime `require()` asserts no metadata column (`country` etc.) can appear in the feature list — the literal defence against label leakage `SPRINT.md` §1.4(d) asks for.

CLI: `--features-path --splits-path --regime {s1,s2,s3,s4} --fold --learner {lr,rf} --output-path --run-id --thread-pool-size`.

**Bug found and fixed during this session:** the per-label output path used
Scala's `s"..."` string interpolator with a `%02d` format specifier — which
does nothing in `s"..."` (that's only meaningful in the `f"..."` interpolator)
— producing literal directory names like `label_0%02d` instead of `label_00`.
Caught immediately by inspecting the first real HDFS output rather than
trusting a clean exit code. **Fixed** (line ~184, now `f"..."`), rebuilt, and
the corrected run verified to produce correct paths (`label_00`..`label_18`).

**End-to-end validation run (Day 2 gate, §2.5) — S2/official split, LR, 10%
subset:** all 19 labels trained successfully in **73–80s wall-clock** with
**~54s total executor CPU time** across 2 executors × 3 cores — real
parallelism confirmed (CPU time < wall time would be impossible without
concurrent execution). Predictions pulled locally and scored:

- **AP^M (macro) = 46.8%, AP^μ (micro) = 58.5%** on the subset's official
  test split. This is *below* `SPRINT.md`'s nominal 0.60–0.85 plausible band,
  but that band was written for the **full 480k** run — this was only 24,050
  training rows (10% subset). The per-class ranking is physically sane
  (Marine waters 96%, Arable land 85%, Coniferous forest 81% at the top; the
  rarest classes with only ~55–73 train positives at the bottom) — a
  misaligned/randomized pipeline could not produce that ordering, so this is
  read as "expected subset-scale underperformance," not a red flag. **The
  real gate check (0.60–0.85) has not yet been run on the full dataset** —
  that's a Day 3 item.

### 4.4 Threshold tuning — `src/python/eval/threshold_tuning.py`

Per-label F1-maximising threshold, tuned on validation only, applied to test.
`tune_threshold()` takes only validation arrays as arguments — there is no
code path by which test data can influence the chosen threshold (the
structural version of `SPRINT.md`'s "assert the test split is never touched
during tuning"). Self-tested against synthetic data: tuned threshold's F1
(0.930) beats the naive 0.5 cutoff (0.891), as expected. **Not yet run against
real predictions** — needs a completed training run's val+test predictions.

---

## 5. Files created/modified this session

```
docker-compose.yml                              MODIFIED (2 datanodes, 6-core worker, /workspace mount)
hadoop.env                                       MODIFIED (replication 3 -> 2)

docs/archive_layout.md                           NEW (verify_archive.py output)
docs/feature_dictionary.md                       NEW (features.py output)
docs/PROJECT_STATUS.md                           NEW (this file)

src/python/ingest/verify_archive.py              NEW
src/python/ingest/patch_meta.py                  NEW
src/python/ingest/features.py                    NEW
src/python/ingest/stream_extract.py              NEW
src/python/ingest/day1_gate.py                   NEW

src/python/splits/build_splits.py                NEW

src/python/eval/threshold_tuning.py              NEW
src/python/eval/compute_optimism_gap.py          NEW (written, NOT yet run -- needs Day 3 E3 runs first)

src/main/scala/multilabel/BinaryRelevanceRunner.scala   NEW

scripts/run_e3_audit.sh                          NEW (written, NOT yet run -- Day 3)

features/subset_10pct/*.parquet                  NEW (115 shards, 32MB, 48,002 rows, in HDFS)
features/full/*.parquet                          NEW (115 shards, 229MB, 480,038 rows, NOT yet in HDFS)
state/completed_tiles_frac10.json                NEW
state/completed_tiles_full.json                  NEW
results/splits.parquet                           NEW (full 480,038-row split table, in HDFS)
results/split_stats.json                         NEW
```

Legacy files untouched: `src/python/0_create_subset_from_archive.py` through
`4_generate_dashboard.py`, `src/main/scala/LandCoverClassifier.scala`,
`bigearthnet_subset/`.

---

## 6. What remains

### Day 3 (`SPRINT.md` §3) — NOT STARTED, waiting for the go-ahead

1. `docker compose up -d` (cluster back up).
2. Push `features/full/*.parquet` to `/bigearthnet/features/` in HDFS
   (replace/augment the subset shards currently there — decide whether to
   clear the subset shards first or use a separate HDFS path so subset and
   full data don't get mixed in the same join).
3. **E1 — Table 1 of the report:** rerun `BinaryRelevanceRunner` (S2/official,
   both LR and RF) on the **full** 480,038 patches, fill in AP^M/AP^μ/F1^M/F1^μ
   next to the published ResNet-50 numbers (70.72/85.86/64.74/76.34). Expect
   to land meaningfully below ResNet-50 — that's the expected, correct result
   feeding the efficiency argument, not something to tune away.
4. **E3 — the split audit, the paper's real contribution:** run
   `scripts/run_e3_audit.sh` (already written, untested against real full
   data) — S1, S2, S3×5 folds, S4×10 folds = 17 Spark jobs, LR only (the
   "workhorse" per §2.3 — consistency across regimes matters more than raw
   strength for this specific measurement). Then run
   `src/python/eval/compute_optimism_gap.py` against the pulled-down
   predictions to get the headline number: macro-AP decay from S2 to S4,
   reported both as fold-macro-average and pooled (since Finland at 32% of
   the data and Kosovo at 0.3% are not comparable folds). Expect a 10–25
   point drop per the sprint's own estimate.
5. Launch BR-RandomForest on S2 (full data) overnight, `numTrees=50,
   maxDepth=10`, ~1.5h estimated.
6. Day 3 gate: E1 table filled for full 480k; E3 complete across all 4
   regimes; `results/optimism_gap.json` written.

### Day 4 (`SPRINT.md` §4) — NOT STARTED

- `src/python/eval/metrics.py`: AP^M/AP^μ/F1^M/F1^μ, per-class AP/F1, Hamming
  loss, subset accuracy, bootstrap 95% CIs (1000 resamples). Validate against
  `sklearn.metrics.average_precision_score` on a synthetic case first.
- **F1 (split-strictness decay)**, **F2 (Europe choropleth by leave-country-out
  macro-AP)**, **F3 (per-class AP, ours vs. ResNet-50)** — the three figures
  that matter per the sprint; F1 is described as "the paper in one image."
  Vector PDF + PNG, viridis/cividis, axis units labelled, *n* in every
  caption.
- `dashboard.html`: single self-contained file, Plotly inlined, no server,
  tabs for Overview / Split audit / Per-class explorer. (A `generate_dashboard.py`
  already exists from the legacy pipeline — decide whether to extend it or
  write fresh; it currently reads legacy single-label output, not the new
  binary-relevance predictions.)
- Day 4 gate: F1/F2/F3 rendered, dashboard opens by double-click.

### Day 5 (`SPRINT.md` §5) — NOT STARTED

- Report/deck write-up (7 sections per §5.1), written honestly about
  limitations (S2-only, hand-crafted features, single seed, LR workhorse, 10
  countries, the Portugal/Agro-forestry gap found in §4.1 above).
- Viva prep: 10 anticipated questions with prepared answers already drafted
  in `SPRINT.md` §5.2 — worth rehearsing against what was *actually* built
  this session, since some answers reference specific numbers (e.g. Q2's
  "63GB compressed / 88GB raw" claim, Q6's "5.8M small files" claim) that
  should be double-checked against this project's real figures before the
  viva (this session's real "in archive but not metadata" count is 69,450,
  not directly stated in §5.2 but consistent with it).
- Live-demo checklist (`docker compose up -d` cold-start timing, HDFS/Spark
  UIs, dashboard, one-page cheat sheet) — rehearse once.

---

## 7. Known issues / things to double check before relying on them further

- **AP^μ on the subset (58.5%) is below the nominal full-data band (60–85%).**
  Read as expected subset-scale behavior (see §4.3), but the *first* thing to
  check once the full-data E1 run completes is whether AP^μ lands in-band —
  if it's still near 50%, per `SPRINT.md`'s own diagnostic, stop and recheck
  band alignment before trusting anything downstream.
- **`compute_optimism_gap.py` and `run_e3_audit.sh` are untested against real
  multi-fold data** — they were written and reasoned through carefully but
  the only real integration test so far is the single S2/LR subset run.
  Treat the first real E3 run as a test of the scripts too, not just the
  model.
- **HDFS currently holds only the subset features**, not the full 480k
  extraction — don't assume `/bigearthnet/features/` in HDFS matches
  `features/full/` on disk until the Day-3 `hdfs dfs -put` step actually runs.
- Legacy `predictions.csv` and `resuults.png` (note the typo in the original
  filename) were deleted from the working tree before this session started
  (visible in `git status` as `D`) — not touched or restored during this
  session; they're presumably superseded by the new pipeline's outputs.
