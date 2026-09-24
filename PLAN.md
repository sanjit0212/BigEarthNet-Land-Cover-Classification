# BigEarthNet-S2 v2.0 — Distributed Multi-Label Land-Cover Project

## Implementation plan for Claude Code

**Stack:** Docker + HDFS + Spark + Python (ingestion/eval/viz) + Scala (distributed ML)
**Dataset on disk:** `BigEarthNet-S2.tar.zst` (63.2 GB), `metadata.parquet` (3.6 MB)
**Goal:** a publishable contribution, not a course demo.

---

# PART 0 — Read this first: honest positioning

## 0.1 What will NOT be publishable

The current code (`LandCoverClassifier.scala`, `2_extract_features.py`) does three things that
make publication impossible, and **all three must be reversed**:

| Current | Problem | Fix |
|---|---|---|
| Predicts `dominant_label` (single-label) | BigEarthNet is **multi-label**; mean 2.95 labels/patch. Collapsing to one label is not the benchmark task and cannot be compared to any published number. | Predict all 19 labels as binary targets. |
| 10 features (NDVI, NDWI, 4 band mean/std) | Discards 8 of 12 Sentinel-2 bands, including all three red-edge bands that are the main advantage of S2 over Landsat. | ~150-feature physics-informed descriptor (Phase 2). |
| 40k random subset | Throws away 92% of the data; a "big data" claim on 40k patches is not credible. | Use all 480,038 patches. |

**Also do not attempt to beat the deep-learning state of the art on accuracy.** You will not.
A Random Forest on hand-crafted features will land roughly 10–20 AP points below ResNet-50.
Writing the paper as "our RF beats CNNs" guarantees rejection. The contribution has to come
from somewhere the Spark/HDFS stack is an *advantage*, not a handicap.

## 0.2 Where the stack IS an advantage

A CPU cluster that trains a model in minutes instead of GPU-hours lets you run **dozens of
training configurations**. Deep-learning papers report one split because one split costs them
a week of A100 time. You can report four splits × five learners × three feature sets. That
asymmetry is the whole strategy: **the contribution is a controlled study that is too expensive
to run with deep models, not a better model.**

## 0.3 Realistic target venues

IGARSS, IEEE JSTARS, *Remote Sensing* (MDPI), ISPRS Archives/Annals, EARTHVISION or the
BiDS (Big Data from Space) workshop. Not CVPR/NeurIPS. An honest, rigorous benchmark audit
is a very good fit for the first four.

---

# PART 1 — Verified facts about the data

These were confirmed by reading `metadata.parquet` directly. **Do not re-derive; trust these.**

## 1.1 metadata.parquet

```
shape: (480038, 8)

patch_id                  str    e.g. S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
labels                    list[str]   ← MULTI-LABEL, 1..11 labels per patch
split                     str    train | validation | test
country                   str    10 European countries
s1_name                   str    matching Sentinel-1 patch (not downloaded)
s2v1_name                 str    v1.0 patch name (for v1↔v2 comparison)
contains_seasonal_snow    bool   all False in this file
contains_cloud_or_shadow  bool   all False in this file
```

> **Note:** this is the *clean* metadata file. Snow/cloud patches live in a separate
> `metadata_snow_cloud.parquet` on Zenodo. The full reBEN archive is 549,488 patches;
> this clean subset is 480,038. Optional download — needed only for Experiment E5.

## 1.2 Splits (official reBEN geographic split)

```
train        237,871   (49.6%)
validation   122,342   (25.5%)
test         119,825   (25.0%)
```

## 1.3 The 19 classes and their frequency — severe long tail

```
 188,025  39.17%  Arable land
 165,780  34.53%  Mixed forest
 154,941  32.28%  Coniferous forest
 141,150  29.40%  Transitional woodland, shrub
 135,928  28.32%  Broad-leaved forest
 122,709  25.56%  Land principally occupied by agriculture, with significant
                  areas of natural vegetation
  99,598  20.75%  Complex cultivation patterns
  95,605  19.92%  Pastures
  63,758  13.28%  Urban fabric
  63,212  13.17%  Inland waters
  61,832  12.88%  Marine waters
  33,181   6.91%  Agro-forestry areas
  29,588   6.16%  Permanent crops
  20,919   4.36%  Inland wetlands
  13,894   2.89%  Moors, heathland and sclerophyllous vegetation
  11,882   2.48%  Natural grassland and sparsely vegetated areas
  11,142   2.32%  Industrial or commercial units
   1,397   0.29%  Coastal wetlands
   1,316   0.27%  Beaches, dunes, sands
```

145× imbalance between the most and least common class. **Macro metrics are dominated by
the bottom five classes** — this is where the interesting results will be.

## 1.4 Label cardinality

```
mean 2.95, median 3, min 1, max 11
1 label: 86,375   2: 94,649   3: 135,885   4: 102,036   5: 45,329
6: 12,963   7: 2,478   8: 302   9: 15   10: 1   11: 5
```

## 1.5 Countries

```
Finland      155,227      Austria       43,797      Luxembourg    3,460
Portugal      89,792      Belgium       11,196      Kosovo        1,616
Serbia        73,385      Switzerland    4,874
Lithuania     48,365      Ireland       48,326
```

## 1.6 ⚠️ THE KEY FINDING — the "geographic split" is only *locally* geographic

`patch_id` encodes tile and within-tile grid position:
`..._T33UUP_26_57` → tile `T33UUP`, grid cell `(26, 57)`.

Measured structure of the official split, per tile:

```
split         gx range    gy range    mean
test            23–68       23–68      45.5   ← innermost square
validation      13–77       13–77      44.7   ← middle frame
train            0–90        0–90      45.3   ← outer frame
```

**46 of 54 tiles contain all three splits.** Train, validation and test therefore share:

- the same 54 Sentinel-2 tiles
- the same 10 countries
- the same acquisition dates and atmospheric conditions
- the same regional land-cover idiom (Finnish boreal forest is in train *and* test)

The reBEN split removes **adjacency** leakage. It does **not** remove **tile-level** or
**country-level** leakage. Published reBEN numbers therefore measure *interpolation within
known regions*, not *generalization to a new region* — which is what an operational land-cover
product actually needs.

**This gap is the paper.**

## 1.7 Temporal coverage (also derivable from patch_id)

```
2017: 276,163      2018: 203,875
month:  1:685   2:38,092   3:10,467   4:48,414   5:106,217  6:17,797
        7:33,557  8:51,043  9:77,889  10:36,171  11:26,673  12:33,033
```

Every month is represented → season is a usable experimental factor and a confounder
nobody has reported on for v2.0.

## 1.8 Archive layout — VERIFY BEFORE CODING

Expected (confirm with a listing first — Phase 1, step 1):

```
BigEarthNet-S2/
└── <tile_folder>/
    └── <patch_id>/
        ├── <patch_id>_B01.tif   60 m   20×20 px
        ├── <patch_id>_B02.tif   10 m  120×120 px   (blue)
        ├── <patch_id>_B03.tif   10 m  120×120 px   (green)
        ├── <patch_id>_B04.tif   10 m  120×120 px   (red)
        ├── <patch_id>_B05.tif   20 m   60×60 px    (red-edge 1)
        ├── <patch_id>_B06.tif   20 m   60×60 px    (red-edge 2)
        ├── <patch_id>_B07.tif   20 m   60×60 px    (red-edge 3)
        ├── <patch_id>_B08.tif   10 m  120×120 px   (NIR)
        ├── <patch_id>_B8A.tif   20 m   60×60 px    (narrow NIR)
        ├── <patch_id>_B09.tif   60 m   20×20 px
        ├── <patch_id>_B11.tif   20 m   60×60 px    (SWIR 1)
        └── <patch_id>_B12.tif   20 m   60×60 px    (SWIR 2)
```

12 bands (no B10 — removed in L2A processing). Values are L2A surface reflectance,
uint16, scale factor 10000.

---

# PART 2 — Related work and the gap

## 2.1 Published baselines on reBEN (Clasen et al. 2024/2025, Table I)

These are the numbers to compare against. **S2-only column is your comparison target** —
you have no Sentinel-1 data.

| Model | AP^M | AP^μ | F1^M | F1^μ |
|---|---|---|---|---|
| ResNet-50 | **70.72** | **85.86** | **64.74** | **76.34** |
| ResNet-101 | 70.63 | 85.92 | 64.19 | 76.13 |
| MobileViT-S | 69.84 | 86.20 | 62.10 | 75.99 |
| InceptionNeXt Base | 69.32 | 85.48 | 62.93 | 75.63 |
| MobileNet V4 Hybrid M | 68.69 | 85.45 | 62.47 | 75.55 |
| ConvNeXt V2 Base | 68.61 | 85.13 | 62.64 | 75.43 |
| RDNet Base | 68.53 | 85.42 | 62.35 | 75.62 |
| MLP-Mixer Base | 67.77 | 84.32 | 62.49 | 74.59 |

Notation: `M` = macro, `μ` = micro, AP = average precision, F1 at tuned threshold.

**Use exactly these four metrics** so your table is directly comparable. Do not invent
metrics or report accuracy.

## 2.2 Gap table — what has and has not been done

| Question | Addressed in the literature? | Your opportunity |
|---|---|---|
| Deep architectures on reBEN | Yes — 8 models, Table I above | None. Don't compete. |
| Multi-modal S1+S2 fusion | Yes (reBEN paper) | None — you lack S1. |
| Spatial autocorrelation inflates reported accuracy | Yes **in general** (Karasiak et al. 2022, *Machine Learning*; spatial CV literature) | **Never applied to reBEN v2.0.** The dataset is new. |
| Does reBEN's own split fully remove spatial leakage? | **No — nobody has audited it** | **C2. Primary contribution.** |
| Distributed multi-label classification in Spark MLlib | **No native support exists.** MLlib ships `OneVsRest` (multi-class only) and `MultilabelMetrics` (evaluation only). No multi-label *estimator*. | **C1. Systems contribution.** |
| Hand-crafted spectral features vs deep features, cost-normalised | Partially, on older/smaller datasets | **C3.** Accuracy per CPU-hour on reBEN is unreported. |
| Seasonal robustness on reBEN | No | **C4.** |
| Spark for RS at scale | Yes (GeoPySpark, Spark+DL inference papers) | Cite as related; not a novelty claim on its own. |

## 2.3 Papers to read and cite (Claude Code: fetch and put in `docs/related_work.md`)

1. **Clasen et al., "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"**, arXiv:2407.03653 — the dataset paper. Primary comparison target.
2. **Sumbul et al., "BigEarthNet-MM"**, IEEE GRSM 2021 — v1 multi-modal benchmark.
3. **Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive"**, IGARSS 2019 — original.
4. **Karasiak et al., "Spatial dependence between training and test sets: another pitfall of classification accuracy assessment in remote sensing"**, *Machine Learning* 111, 2022 — **methodological backbone of C2.**
5. **Ploton et al., "Spatial validation reveals poor predictive performance of large-scale ecological mapping models"**, Nat. Commun. 2020 — supporting evidence for C2.
6. **Read et al., "Classifier chains for multi-label classification"**, Machine Learning 2011 — basis for C1.
7. **Papadopoulos et al. / Schwartz et al., "Green AI"**, CACM 2020 — framing for C3.
8. Any Spark-for-remote-sensing paper (e.g. *Big Data Cogn. Comput.* 5(2):21, 2021) — related work for the systems section.

---

# PART 3 — The four contributions

## C1 — `spark-multilabel`: a Spark-native distributed multi-label framework (Scala)

**Gap:** Spark MLlib has no multi-label estimator. Practitioners hand-roll 19 separate
training jobs, which re-reads and re-caches the dataset 19 times.

**What to build** — proper Spark ML `Estimator`/`Model` pairs, `Params`-compliant,
persistable via `MLWritable`:

1. **`BinaryRelevance`** — trains L independent binary classifiers, but with a **single
   cached feature DataFrame and one pass of the featurisation pipeline**, not L passes.
   Optionally train labels in parallel using a fixed-size thread pool over the shared
   `SparkContext` (this is the actual engineering win — measure it).
2. **`ClassifierChains`** — label *i*'s prediction is appended to the feature vector for
   label *i+1*.
3. **`CorrelationOrderedChains`** — **the novel part.** Chain order is derived from a
   label co-occurrence graph computed in **one distributed pass** (a 19×19 co-occurrence
   matrix via `reduce`), then ordered by descending normalised pointwise mutual
   information. Compare against random chain order and frequency order.
4. **`EnsembleOfClassifierChains`** — *m* chains with different orders, averaged.
5. **`StackedBinaryRelevance` (2BR)** — stage-2 features = original features + all 19
   stage-1 probabilities. Cheap, usually a solid gain on correlated labels.
6. **`PerLabelThresholdOptimizer`** — tunes a decision threshold per label on the
   *validation* split to maximise F1. **Most papers skip this and it is worth several
   macro-F1 points on rare classes.** Must never touch test.

**Deliverable:** a reusable Scala library + an evaluation of whether label-correlation
modelling helps on reBEN at all. (Plausible honest finding: it helps rare classes, not
common ones — that is a fine result.)

## C2 — Split-geometry generalization audit ★ PRIMARY CONTRIBUTION

**Claim to test:** *reBEN's concentric split reduces but does not eliminate spatial
optimism; published numbers overstate performance in unseen regions by a measurable margin.*

Train the **identical model and features** under four split regimes:

| Regime | Construction | What it measures |
|---|---|---|
| **S1 Random** | i.i.d. shuffle, 50/25/25 | v1.0-style — upper bound, maximum leakage |
| **S2 Official** | reBEN concentric within-tile | the published protocol |
| **S3 Leave-tile-out** | group by `tile`, disjoint tile sets | generalization to an unseen acquisition |
| **S4 Leave-country-out** | group by `country`, 10-fold | generalization to an unseen region ★ |

**The money figure:** macro-AP as a function of split strictness, S1 → S4. If the decay from
S2 to S4 is large (expect 10–25 AP points), you have a clean, quantitative, citable result
about a brand-new benchmark.

**Strengthen it:** compute a spatial-autocorrelation statistic (Moran's I on feature vectors
using within-tile grid coordinates `gx`, `gy`) and correlate autocorrelation strength with
the size of the optimism gap, per tile. That turns an observation into an explanation.

**Why it is safe:** the result is interesting whichever way it comes out. A large gap is a
warning to the community. A small gap is a *validation* of reBEN's split design — also
publishable, and generous to the dataset authors.

## C3 — Accuracy–compute Pareto (Green AI)

Same official split as the reBEN paper, so the comparison is apples-to-apples. Report for
each of your models **and** for the published deep baselines:

- AP^M, AP^μ, F1^M, F1^μ
- wall-clock training time, **total CPU-core-seconds**, peak cluster memory
- estimated energy (kWh) and gCO₂e, via a fixed TDP assumption — state the assumption
- **AP per CPU-hour**

**Question:** what fraction of ResNet-50's 70.72 macro-AP is recoverable at what fraction of
the compute, with no GPU? If a Spark GBT reaches ~55–62 macro-AP at 1–2% of the energy, that
is a genuinely useful number for anyone doing continental-scale mapping on a budget.

**Instrument honestly.** Use Spark's `SparkListener` / event-log JSON for real executor CPU
time — not wall clock. Do not claim energy numbers you cannot defend; give the formula.

## C4 — Seasonal and atmospheric robustness

Derive `month` and `year` from `patch_id`. Train on one season, test on another
(summer→winter is the hard case). Report per-season macro-AP. If you also download
`metadata_snow_cloud.parquet`, add a snow/cloud robustness slice. Section-level
contribution, not a whole paper.

## Proposed title

> **"How geographic is a geographic split? A distributed multi-label audit of spatial
> generalization in BigEarthNet v2.0"**

---

# PART 4 — System architecture

```
┌─ Phase 1 ─────────────────────────────────────────────────────────┐
│ STREAMING INGESTION (Python, host machine)                        │
│  BigEarthNet-S2.tar.zst ──stream──► per-patch feature vectors     │
│  ⚠ NEVER extract to disk: 480k patches × 12 = 5.76M small files.  │
│    On Windows/NTFS that is hours of I/O and ~110 GB.              │
│  Output: 54 Parquet shards (one per tile), ~1–3 GB total          │
└────────────────────────────┬──────────────────────────────────────┘
                             ▼
┌─ Phase 3 ─────────────────────────────────────────────────────────┐
│ HDFS (Docker: 1 NameNode + 3 DataNodes)                           │
│  /bigearthnet/features/  tile-sharded Parquet  ← few large files  │
│  /bigearthnet/splits/    split assignment tables                  │
│  /bigearthnet/results/   predictions + metrics                    │
└────────────────────────────┬──────────────────────────────────────┘
                             ▼
┌─ Phase 4–5 ───────────────────────────────────────────────────────┐
│ SPARK CLUSTER (Docker: 1 master + N workers) — Scala              │
│  spark-multilabel library (C1)                                    │
│  × 4 split regimes (C2) × learners × feature sets                 │
│  Output: per-label probabilities → HDFS                           │
└────────────────────────────┬──────────────────────────────────────┘
                             ▼
┌─ Phase 6–7 ───────────────────────────────────────────────────────┐
│ EVALUATION + VISUALIZATION (Python)                               │
│  metrics, bootstrap CIs, significance tests, figures, dashboard   │
└───────────────────────────────────────────────────────────────────┘
```

## Target repository layout

```
bda_project/
├─ docker/
│  ├─ docker-compose.yml          # rewritten (see Phase 0)
│  ├─ hadoop.env
│  └─ spark.Dockerfile
├─ src/
│  ├─ python/
│  │  ├─ ingest/
│  │  │  ├─ stream_extract.py     # Phase 1 — THE critical script
│  │  │  ├─ features.py           # feature definitions (pure numpy)
│  │  │  └─ verify_archive.py
│  │  ├─ splits/
│  │  │  └─ build_splits.py       # Phase 2 — S1–S4 split tables
│  │  ├─ eval/
│  │  │  ├─ metrics.py            # AP/F1 macro+micro, Hamming, CIs
│  │  │  └─ compare_baselines.py
│  │  └─ viz/                     # Phase 7
│  └─ scala/
│     └─ main/scala/multilabel/
│        ├─ BinaryRelevance.scala
│        ├─ ClassifierChains.scala
│        ├─ CorrelationOrderedChains.scala
│        ├─ EnsembleOfClassifierChains.scala
│        ├─ StackedBinaryRelevance.scala
│        ├─ LabelCorrelationGraph.scala
│        ├─ ThresholdOptimizer.scala
│        ├─ MultilabelEvaluator.scala
│        └─ ExperimentRunner.scala   # CLI driver
├─ conf/experiments/*.yaml
├─ results/
├─ figures/
├─ docs/
└─ paper/
```

---

# PART 5 — Phased implementation

Each phase has a **gate**. Do not start the next phase until the gate passes.

## Phase 0 — Infrastructure rebuild

**Problems with the current `docker-compose.yml`:**
- Absolute Windows host paths (`C:\Users\sanji\...`) — not portable, breaks reproducibility.
- Spark worker capped at 2 cores / 4 GB — far too small for 480k rows × ~150 features × 4 regimes.
- No Spark history server → **C3 cannot be measured without it.**
- Spark 3.3.0 images vs `build.sbt` Spark 3.3.0 / Scala 2.12.18 — keep aligned, verify.

**Tasks**
1. Rewrite `docker-compose.yml`: relative bind mounts, `.env` for paths.
2. Scale the Spark worker to the host's real capacity (leave 2 cores and 4 GB for the OS).
   Prefer 2–3 workers over one fat worker so shuffle behaviour is realistic.
3. Add `spark-history-server` with event logging to HDFS — **required for C3**.
4. Add an `sbt` build container so Scala never needs a host toolchain.
5. Pin every image digest. Record versions in `docs/environment.md`.

**Gate:** `hdfs dfsadmin -report` shows 3 live DataNodes; a `SparkPi` job runs on the
cluster and appears in the history server UI with non-zero executor CPU time.

---

## Phase 1 — Streaming ingestion and feature extraction ★ hardest phase

### Step 1: verify the archive (do this first, alone)

Stream the first ~200 members and print paths, sizes, and one patch's full band list.
Confirm folder nesting, band filenames, dtype, and pixel dimensions against §1.8.
**Write the confirmed layout into `docs/archive_layout.md` before writing the extractor.**

### Step 2: the streaming extractor

**Non-negotiable design:** never extract the tar to disk.

```
zstandard.ZstdDecompressor().stream_reader(f)
  └─ tarfile.open(fileobj=..., mode='r|')     # STREAMING mode, '|' not ':'
       └─ accumulate members until patch folder changes
            └─ 12 band buffers → rasterio.io.MemoryFile → numpy
                 └─ compute features → row
                      └─ flush per tile → Parquet shard
```

Key points:
- `mode='r|'` is sequential-only; you cannot seek. The tar is ordered by patch folder, so
  the 12 bands of a patch arrive contiguously — accumulate until `patch_id` changes, emit,
  reset. **Do not assume band order; key the buffer dict by band name.**
- Use `rasterio.io.MemoryFile(bytes)` to read a GeoTIFF from memory.
- **Parallelise correctly:** one reader process owns the decompression stream and pushes
  `(patch_id, {band: bytes})` onto a bounded `multiprocessing.Queue`; a pool of N−1 workers
  computes features. Decompression (~63 GB at several hundred MB/s) is not the bottleneck —
  feature computation is. A bounded queue prevents unbounded memory growth.
- **Checkpoint per tile.** If it crashes at tile 40, resume at tile 40. Write
  `state/completed_tiles.json`.
- Log throughput (patches/s) every 5,000 patches.

**Expected runtime:** order of 30–90 min on 8 cores. Measure and record — it goes in the paper.

### Step 3: the feature set (`features.py`)

Target ~150 features. Upsample 20 m and 60 m bands to 120×120 with nearest-neighbour
(or compute per-resolution and skip cross-resolution indices — state which you chose).

**(a) Per-band statistics — 12 bands × 8 = 96**
mean, std, min, max, p10, p25, p50, p75, p90, skew, kurtosis
(pick 8; record which).

**(b) Spectral indices — compute the index map, then take mean/std/p10/p90 of each map**

| Index | Formula | Targets |
|---|---|---|
| NDVI | (B08−B04)/(B08+B04) | vegetation |
| EVI | 2.5(B08−B04)/(B08+6·B04−7.5·B02+1) | dense canopy, less saturating |
| SAVI | 1.5(B08−B04)/(B08+B04+0.5) | sparse vegetation |
| NDWI | (B03−B08)/(B03+B08) | open water |
| MNDWI | (B03−B11)/(B03+B11) | water vs built-up |
| NDMI | (B08−B11)/(B08+B11) | vegetation moisture |
| NDBI | (B11−B08)/(B11+B08) | built-up |
| BSI | ((B11+B04)−(B08+B02))/((B11+B04)+(B08+B02)) | bare soil |
| NBR | (B08−B12)/(B08+B12) | burn/disturbance |
| **NDRE1** | (B08−B05)/(B08+B05) | **red-edge — S2-specific** |
| **NDRE2** | (B08−B06)/(B08+B06) | **red-edge** |
| **IRECI** | (B07−B04)/(B05/B06) | **red-edge chlorophyll** |
| **S2REP** | red-edge position | **red-edge** |

The four red-edge indices are the ones the current code throws away and are the strongest
argument that the feature set is *physics-informed* rather than arbitrary.

**(c) Texture — the second thing the current code is missing**
Vegetation and urban fabric differ in *spatial structure*, not just mean reflectance.
Cheap options (choose one, justify it):
- local variance / entropy of B08 and B04 at 3×3 and 7×7
- gradient magnitude statistics (Sobel)
- a reduced GLCM (contrast, homogeneity, energy, correlation) on a 32-level quantised B08

**(d) Geometry/context (metadata joins, not from pixels)**
`gx`, `gy`, tile id, month, year, country — **used only for split construction and
stratified analysis. NEVER as model features** (country would leak the answer). Store them
in the Parquet but keep them out of the `VectorAssembler` input list. Write an assertion
that enforces this.

### Output schema (per tile shard)

```
patch_id: string            tile: string        gx: int    gy: int
month: int                  year: int           country: string
split_official: string
lbl_00 … lbl_18: int        ← 19 binary columns, alphabetical class order
f_000 … f_NNN: float        ← features
```

Write a `docs/feature_dictionary.md` mapping every `f_i` to a human name. You will need it
for the paper and for feature-importance plots.

**Gate:**
- Row count is exactly **480,038**.
- Every `patch_id` joins 1:1 against `metadata.parquet`.
- Per-class positive counts match §1.3 **exactly**.
- Zero NaN/Inf (guard every division with a small epsilon).
- Feature distributions sanity-check: NDVI mean is higher for forest classes than for
  `Urban fabric`. **If this fails, the band mapping is wrong — stop and fix it.**

---

## Phase 2 — Split construction (C2)

`build_splits.py` emits one table: `patch_id → {S1,S2,S3,S4} → {train,val,test}`.

- **S1 Random:** seeded shuffle, 50/25/25.
- **S2 Official:** copy `split` from `metadata.parquet`.
- **S3 Leave-tile-out:** partition the 54 tiles into train/val/test tile sets. Stratify so
  every split covers all 19 classes (the two rarest classes are concentrated in coastal
  tiles — check this explicitly). Use 5 folds.
- **S4 Leave-country-out:** 10 folds, hold out one country per fold. Report per-country and
  the mean. Note honestly that Finland (32% of data) and Kosovo (0.3%) are not comparable
  folds; report both macro-averaged-over-folds and pooled.

**Also compute:** Moran's I per tile on the principal components of the feature vector using
`(gx, gy)` — this quantifies spatial autocorrelation and feeds the explanatory half of C2.

**Gate:** no `patch_id` appears in two splits of the same regime; every regime's train split
contains ≥1 positive for all 19 labels; split sizes logged to `results/split_stats.json`.

---

## Phase 3 — HDFS loading

- `hdfs dfs -put` the tile-sharded Parquet to `/bigearthnet/features/`.
- Target ~128–256 MB per file; **coalesce the 54 shards if any are tiny.** The small-files
  problem your existing study guide identifies is real — apply it here.
- Set replication to 2 (you have 3 DataNodes; 3 wastes space, 1 loses the point).
- Put split tables in `/bigearthnet/splits/`.

**Gate:** `spark.read.parquet(...).count() == 480038` from inside the cluster; `hdfs fsck`
reports no under-replicated blocks.

---

## Phase 4 — The Scala multi-label library (C1)

Build the classes in §C1. Design rules:

- Extend `Estimator[M]` / `Model[M]`, implement `DefaultParamsWritable`.
- The featurisation pipeline (`VectorAssembler` + `StandardScaler`) runs **once**; the
  resulting DataFrame is `persist(MEMORY_AND_DISK)` and reused across all 19 labels.
- Base learner is pluggable (`RandomForestClassifier`, `GBTClassifier`,
  `LogisticRegression`, `LinearSVC`).
- Emit **probabilities**, not hard labels, to HDFS — thresholds are tuned later and AP needs
  scores.
- Handle class imbalance: per-label class weights (`weightCol`) ∝ inverse frequency. Try
  both weighted and unweighted; report both.
- `ExperimentRunner` takes a YAML config so a full sweep is one command per config.
- Register a `SparkListener` that writes per-stage executor CPU time to
  `results/compute/<run_id>.json`. **This is the C3 measurement — build it in from the start,
  not retrofitted.**

**Unit tests** (ScalaTest, on a tiny synthetic DataFrame): chain ordering is deterministic
given a seed; BR with L=1 equals a plain binary classifier; threshold optimiser never reads
the test split; model save/load round-trips.

**Gate:** `BinaryRelevance` + RandomForest on the official split produces AP^μ in a
plausible band (roughly 0.60–0.80). If it is near 0.5, labels or features are misaligned.

---

## Phase 5 — Experiment matrix

| ID | Experiment | Splits | Models | Output |
|---|---|---|---|---|
| **E1** | Baseline multi-label | S2 | BR-RF, BR-GBT, BR-LR | Table 1 vs reBEN Table I |
| **E2** | Label-correlation methods | S2 | BR, CC-random, CC-correlation-ordered, ECC, 2BR | Does correlation help? |
| **E3** ★ | **Split-geometry audit** | **S1,S2,S3,S4** | best from E1 | **Money figure** |
| **E4** | Feature ablation | S2, S4 | best | bands-only / +indices / +red-edge / +texture |
| **E5** | Seasonal robustness | S2 by month | best | train-summer→test-winter |
| **E6** | Compute/energy Pareto | S2 | all | AP per CPU-hour vs deep baselines |
| **E7** | Scalability | S2 | best | 1/2/4/8 executors → speedup & efficiency curve |

**Rigour requirements:**
- 3 seeds minimum for anything stochastic; report mean ± std.
- Bootstrap 95% CIs on test metrics (1000 resamples).
- Paired significance test when claiming one method beats another.
- **Thresholds tuned on validation only.** Write an assertion.
- Log every run's config hash, git SHA, and duration to `results/runs.jsonl`.

---

## Phase 6 — Evaluation

`metrics.py` must implement, matching the reBEN paper:

- **AP^M / AP^μ** (macro / micro average precision) — primary
- **F1^M / F1^μ** at tuned thresholds
- Per-class AP and F1 (19 rows — this is where the rare-class story lives)
- Hamming loss, subset accuracy, one-error, ranking loss, coverage
- Per-country, per-tile, per-season, per-cardinality breakdowns

Validate `metrics.py` against `sklearn.metrics.average_precision_score` on a synthetic
case before trusting it.

---

## Phase 7 — Visualization

Two outputs: **publication figures** (matplotlib, vector PDF, 300 dpi, colour-blind-safe,
no chartjunk) and **one interactive HTML dashboard** (self-contained, Plotly inlined).

### Publication figures

| # | Figure | Type | Why it matters |
|---|---|---|---|
| **F1** ★ | **Split-strictness decay:** macro-AP for S1→S2→S3→S4, with CI bands, one line per model | line + error bands | **The paper's central claim in one image** |
| **F2** ★ | **Choropleth of Europe**, 10 countries shaded by leave-country-out macro-AP | map | Instantly shows *where* the model fails |
| F3 | Per-class AP: your best model vs ResNet-50, 19 classes sorted by frequency | horizontal diverging bar | Shows the deep-vs-shallow gap concentrates in rare classes |
| F4 | Label co-occurrence matrix, 19×19, ordered by the learned chain order | heatmap + dendrogram | Motivates C1 |
| F5 | Accuracy–compute Pareto: macro-AP vs CPU-core-hours, log-x; your models + published deep models as reference points | scatter with frontier | The Green-AI argument |
| F6 | Tile-level performance map: 54 tiles on a Europe basemap, coloured by AP, sized by patch count | geo scatter | Reveals tile-level heterogeneity |
| F7 | Moran's I vs optimism gap, one point per tile, with fitted line | scatter + regression | Turns C2 from observation into explanation |
| F8 | Season × class macro-AP | heatmap | C4 |
| F9 | Feature-importance by group (bands / indices / red-edge / texture) | stacked bar | Justifies the feature design |
| F10 | Strong-scaling curve: speedup vs executors, with the ideal line | line | E7 |

**Design rules:** one message per figure; no 3-D; no rainbow colormaps (use viridis /
cividis); label axes with units; state *n* in every caption; identical colour per model
across all figures.

### Interactive dashboard (`dashboard.html`)

Self-contained single file. Tabs: Overview (headline metrics vs baselines) · Split audit
(F1 + F2, interactive) · Per-class explorer (click a class → its per-country, per-season
performance) · Compute · Run log. Inline all JS/CSS so it opens with no server.

---

# PART 6 — Paper skeleton

```
1 Introduction — continental land-cover mapping; benchmarks drive the field;
                 benchmark splits decide what "generalization" means
2 Related work — BigEarthNet v1/v2; spatial CV literature; distributed RS; multi-label
3 Data       — reBEN; §1 facts; the split geometry finding (§1.6) with a diagram
4 Method     — 4.1 streaming ingestion  4.2 physics-informed features
               4.3 spark-multilabel (C1)  4.4 the four split regimes (C2)
5 Experiments— E1–E7
6 Results    — 6.1 baselines vs published   6.2 ★ the optimism gap
               6.3 why (Moran's I)  6.4 accuracy–compute  6.5 seasonal
7 Discussion — what reBEN's split does and does not guarantee;
               recommendation: report leave-country-out alongside the official split
8 Limitations— S2-only (no S1); hand-crafted features; 10 countries ≠ the world;
               energy figures are estimates
9 Conclusion
```

**Write §8 honestly and at length.** Reviewers punish overclaiming far harder than they
punish a modest, well-bounded result.

---

# PART 7 — Risks

| Risk | Mitigation |
|---|---|
| **Extracting the tar fills the disk** | Streaming design (Phase 1). Never extract. This is the single most important instruction in this document. |
| Archive layout differs from §1.8 | Phase 1 Step 1 verifies before any extraction code is written. |
| Feature extraction too slow | Bounded-queue worker pool; per-tile checkpointing; profile on 2 tiles first and extrapolate before committing to a full run. |
| Band/label misalignment silently ruins everything | Phase 1 gate: NDVI must be higher for forest than urban. Non-negotiable. |
| The optimism gap turns out small | Still publishable — it validates reBEN's design. Frame the paper as an *audit*, not an *attack*, from the start. |
| Spark worker OOM | 480k × 150 float64 ≈ 576 MB — comfortable. If GBT with 19 labels strains memory, train labels sequentially and persist to disk. |
| Threshold tuning leaks test data | Assertion in code + a unit test. |
| Scala/Spark version drift | Pin image digests; `build.sbt` Spark 3.3.0 / Scala 2.12.18 must match the container. |

---

# PART 8 — Suggested order of work

```
Week 1   Phase 0 infra + Phase 1 Step 1 (verify archive)
Week 2   Phase 1 full extraction  ← the hard part; do not rush the gate
Week 3   Phase 2 splits + Phase 3 HDFS + Phase 4 library skeleton
Week 4   Phase 4 complete + E1 baseline (first comparable number)
Week 5   E2, E3 ★ (the main result)
Week 6   E4–E7
Week 7   Phase 6 evaluation + Phase 7 figures
Week 8   Paper draft
```

**Earliest meaningful checkpoint:** end of Week 4, when E1 produces an AP^μ you can put next
to ResNet-50's 85.86. If that number is wildly off, something upstream is broken and
everything after it is wasted.

---

# APPENDIX A — Immediate first actions for Claude Code

1. `docs/archive_layout.md` — stream the first 200 tar members, record the true layout.
2. `src/python/ingest/verify_archive.py` — the script that does (1), reusable.
3. Extract **2 tiles only**, compute features, run the Phase 1 gate checks on that subset,
   and report measured throughput before attempting all 54.
4. Report back: confirmed layout, measured patches/sec, projected full-run time, and whether
   the NDVI forest-vs-urban sanity check passes.

**Do not write the Scala library until the Phase 1 gate passes.** The features are the
foundation; everything downstream is worthless if they are wrong.

---

# APPENDIX B — What to delete or archive

- `src/python/1_build_subset.py`, `0_create_subset_from_archive.py` — subsetting is
  abandoned; you now use all 480k.
- `LandCoverClassifier.scala` — single-label; superseded by the multi-label library.
  Keep it as `legacy/` for the "previous approach" paragraph.
- `generate_dashboard.py` duplicated at repo root and in `src/python/` — keep one.
- `Complete_Project_Documentation.docx`, `Code_Documentation.docx` — regenerate at the end.
