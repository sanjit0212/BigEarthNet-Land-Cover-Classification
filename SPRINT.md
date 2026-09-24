# 5-DAY SPRINT — BigEarthNet-S2 v2.0

**Deadline: 20 September. Today: 15 September.**
**Purpose: college submission + viva. Hardware: 8 cores / 16 GB.**

This document **overrides** `PLAN.md` wherever they conflict. `PLAN.md` remains the
reference for the full 8-week version and for the post-deadline paper.

Because this is a **viva**, you must be able to *defend* every choice out loud. Working
code you cannot explain scores worse than simpler code you understand completely.
§5.2 is the examiner-question prep and is as important as the code.

---

## 0.0 Hardware configuration — 8 cores / 16 GB

16 GB is tight for HDFS + Spark + Windows simultaneously. Two rules:

**Rule 1: Do not run Docker during Phase 1 extraction.** Extraction is pure Python
multiprocessing and wants all 8 cores and as much RAM as it can get. Shut the cluster
down, extract, then bring it back up. Running both halves your extraction speed.

**Rule 2: Budget the containers explicitly.**

| Component | Memory | Note |
|---|---|---|
| Windows + Docker Desktop | ~4 GB | unavoidable |
| NameNode | 1 GB | |
| DataNodes × 2 | 0.75 GB each | **use 2, not 3** — 3 replicas of 500 MB is pointless at RF=2 |
| Spark master | 512 MB | |
| **Spark worker** | **6 GB** | `SPARK_WORKER_CORES=6`, `SPARK_WORKER_MEMORY=6G` |
| Headroom | ~2 GB | |

Leave 2 cores for the OS — `SPARK_WORKER_CORES=6`, not 8. An oversubscribed worker on
Windows/WSL2 causes heartbeat timeouts that look like crashes.

Set the executor explicitly at submit time:
```
--executor-memory 5g --executor-cores 3 --num-executors 2 --driver-memory 2g
```
Two executors of 3 cores beats one of 6: you get real shuffle behaviour to talk about,
and a failed executor doesn't kill the run.

If Spark OOMs during RandomForest, drop `maxDepth` to 8 before touching memory settings.

---

## 0. Read this before writing any code

### 0.1 The one rule

**Build the whole pipeline on a 10% subset first, then scale to full data.**

Extraction on ~48k patches takes ~6 minutes; on 549k it takes ~1 hour. Get every stage
working end-to-end on the subset by end of Day 2. From that moment you always have a
submittable result. Without this, any Day-4 failure leaves nothing to hand in.

### 0.2 Scope cuts vs PLAN.md

| Cut | Reason | Replacement |
|---|---|---|
| Spark `Estimator`/`Model`/`Params`/`MLWritable` conformance | 1–2 weeks | Plain Scala object with a BR loop, ~200 lines |
| ClassifierChains, ECC, 2BR, CorrelationOrderedChains | Not reachable | Binary Relevance only |
| **E2** label-correlation methods | Depends on the above | — |
| **E4** feature ablation | Time | Mention as future work |
| **E5** seasonal robustness | Time | Optional: one heatmap if Day 5 is calm |
| **E7** scalability study | Time | Optional: one 2-vs-8-executor timing |
| 3 seeds | Time | 1 seed; 2 for the headline row |
| GLCM texture | ~15 ms/patch, triples extraction | Local variance + Sobel (~1 ms) |
| Moran's I explanatory analysis | Time | Optional Day 5 |

### 0.3 What must survive — the definition of done

1. All **480,038** patches, true **multi-label**, 19 classes
2. One **AP^μ** directly comparable to ResNet-50's published **85.86**
3. **The split audit (E3):** S1 random → S2 official → S3 leave-tile-out → S4 leave-country-out
4. Figures **F1, F2, F3** + dashboard

If you have these four, the project is complete. Everything else is decoration.

---

## 1. DAY 1 — Tuesday 15 Sept

### 1.1 First 30 minutes (start the slow things immediately)

```bash
docker compose pull          # ~10 GB, runs in background
```
While that downloads, do §1.2. Do **not** sit and watch it.

### 1.2 Verify the archive — before writing the extractor

`src/python/ingest/verify_archive.py`

Stream the first ~200 tar members and print: full paths, nesting depth, band filenames,
dtype, and pixel dimensions for one complete patch. Write the confirmed layout to
`docs/archive_layout.md`.

Expected (confirm, do not assume):
```
BigEarthNet-S2/<tile_folder>/<patch_id>/<patch_id>_B{01,02,03,04,05,06,07,08,8A,09,11,12}.tif
10 m: B02,B03,B04,B08 = 120x120     20 m: B05,B06,B07,B8A,B11,B12 = 60x60
60 m: B01,B09 = 20x20               uint16 L2A reflectance, scale 10000
```

**Also record the total patch count in the archive** (549,488 expected — includes
snow/cloud patches absent from `metadata.parquet`'s 480,038). Filter against
`metadata.parquet` during extraction; do not extract what you will discard.

### 1.3 The streaming extractor

`src/python/ingest/stream_extract.py`

**Never extract the tar to disk.** 6.6 M small files, ~115 GB, and hours of NTFS I/O.

```
zstandard.ZstdDecompressor().stream_reader(f)
  └─ tarfile.open(fileobj=..., mode='r|')      # '|' = streaming, NOT 'r:'
       └─ accumulate members until patch_id changes
            └─ 12 band buffers → rasterio.io.MemoryFile → numpy
                 └─ features → row → per-tile Parquet shard
```

- `mode='r|'` is sequential-only, no seeking. Bands of a patch arrive contiguously;
  buffer in a **dict keyed by band name** — never assume ordering.
- **Parallelism:** one reader process owns the stream and pushes
  `(patch_id, {band: bytes})` onto a **bounded** `multiprocessing.Queue(maxsize=512)`;
  N−1 workers compute features. The bound prevents unbounded memory growth.
- **Windows note:** `multiprocessing` uses spawn; put worker functions at module level
  and guard with `if __name__ == "__main__":`. Large-bytes IPC is slower on Windows —
  if the queue is the bottleneck, batch 16 patches per queue item.
- **Checkpoint per tile** to `state/completed_tiles.json`. Resume support is mandatory —
  you do not have time to restart from zero.
- Log patches/sec every 5,000 patches.
- CLI flag `--subset-fraction 0.1 --seed 42` for the Day-1 subset run
  (stratify by tile so all 54 tiles appear).

### 1.4 Feature set — ~110 features, cheap only

`src/python/ingest/features.py`, pure numpy, no I/O.

Upsample 20 m / 60 m bands to 120×120 nearest-neighbour.

**(a) Per-band stats — 12 bands × 6 = 72**
`mean, std, p10, p50, p90, iqr`

**(b) Spectral indices — 13 indices × mean/std = 26**

| Index | Formula |
|---|---|
| NDVI | (B08−B04)/(B08+B04) |
| EVI | 2.5(B08−B04)/(B08+6·B04−7.5·B02+1) |
| SAVI | 1.5(B08−B04)/(B08+B04+0.5) |
| NDWI | (B03−B08)/(B03+B08) |
| MNDWI | (B03−B11)/(B03+B11) |
| NDMI | (B08−B11)/(B08+B11) |
| NDBI | (B11−B08)/(B11+B08) |
| BSI | ((B11+B04)−(B08+B02))/((B11+B04)+(B08+B02)) |
| NBR | (B08−B12)/(B08+B12) |
| **NDRE1** | (B08−B05)/(B08+B05) |
| **NDRE2** | (B08−B06)/(B08+B06) |
| **NDRE3** | (B08−B07)/(B08+B07) |
| **CIre** | (B07/B05)−1 |

The four red-edge indices are Sentinel-2-specific and are what the legacy 10-feature
version threw away. They are the justification for calling the feature set
*physics-informed*.

**(c) Texture — 12** (cheap only)
`std` and `p90` of: 3×3 local variance of B08, 7×7 local variance of B08,
Sobel gradient magnitude of B08, and the same three for B04.
Use `scipy.ndimage.uniform_filter` for local variance — **do not use `skimage.graycomatrix`.**

**Guard every division:** `denom = np.where(np.abs(d) < 1e-6, 1e-6, d)`. Any NaN/Inf
propagating into Spark will produce silent garbage.

**(d) Metadata columns — stored, NEVER used as features**
`tile, gx, gy, month, year, country` — for splits and stratified analysis only.
Write an assertion in the Scala job that none of these appear in the
`VectorAssembler` input list. `country` would leak the answer outright.

### 1.5 Output schema (per-tile Parquet shard)

```
patch_id: string      tile: string     gx: int    gy: int
month: int            year: int        country: string    split_official: string
lbl_00 … lbl_18: int     ← 19 binary, alphabetical class order, order frozen in docs/
f_000 … f_109: float32   ← float32, not float64
```

Emit `docs/feature_dictionary.md` mapping every `f_i` to a human-readable name.

### 1.6 ▶ DAY 1 GATE — do not proceed until all pass

Run on the 10% subset:

- [ ] Row count matches expectation; every `patch_id` joins 1:1 to `metadata.parquet`
- [ ] Per-class positive rates within ~1% of PLAN.md §1.3 proportions
- [ ] Zero NaN, zero Inf, in every feature column
- [ ] **`NDVI_mean` for patches labelled `Coniferous forest` > `NDVI_mean` for
      patches labelled `Urban fabric`**
- [ ] **`MNDWI_mean` for `Marine waters` > for `Arable land`**

> The last two are the only defence against silent band misalignment. Misaligned bands
> produce a pipeline that runs perfectly and yields a completely invalid paper. **If
> either fails, stop and fix the band mapping. Do not continue.**

---

## 2. DAY 2 — Wednesday 16 Sept

### 2.1 Split tables

`src/python/splits/build_splits.py` → one table `patch_id → {S1,S2,S3,S4}`.

- **S1 Random** — seeded shuffle 50/25/25
- **S2 Official** — copy `split` from `metadata.parquet`
- **S3 Leave-tile-out** — 5 folds over the 54 tiles; stratify so every fold's train set
  contains ≥1 positive for all 19 labels (check the two rarest, `Coastal wetlands` and
  `Beaches, dunes, sands` — they concentrate in coastal tiles)
- **S4 Leave-country-out** — 10 folds, one held-out country each

**Assert:** no `patch_id` in two splits of the same regime; every train split has ≥1
positive per label. Write `results/split_stats.json`.

### 2.2 HDFS

```bash
hdfs dfs -mkdir -p /bigearthnet/{features,splits,results}
hdfs dfs -setrep -w 2 /bigearthnet
hdfs dfs -put features/*.parquet /bigearthnet/features/
```
Coalesce shards to ~128–256 MB. Replication **2**, not 1 — it costs ~0.5 GB and lets you
demo fault tolerance. Do **not** put the tar.zst in HDFS.

### 2.3 The Scala job — keep it simple

`src/scala/main/scala/multilabel/BinaryRelevanceRunner.scala`

Plain `object` with a `main`. No `Estimator`, no `Params`, no `MLWritable`.

```
1. read features Parquet from HDFS, join split table
2. VectorAssembler(f_000..f_109) → StandardScaler → persist(MEMORY_AND_DISK)   ← ONCE
3. for each of 19 labels, IN PARALLEL over a fixed thread pool (size 4–8):
      train base learner on train split with weightCol = inverse class frequency
      predict probabilities on validation + test
4. write per-label probabilities to HDFS as Parquet
5. SparkListener → results/compute/<run_id>.json  (executor CPU time)
```

**Parallel label training is the one piece of engineering worth keeping from C1.**
`scala.collection.parallel` or a `Future` pool over the shared `SparkContext`. Expect
3–4× speedup. Measure it — it is a legitimate result.

**Hyperparameters — these differ from the legacy code, deliberately:**

| Learner | Settings | Use |
|---|---|---|
| **LogisticRegression** | `maxIter=50, regParam=0.01, standardization=true` | **Workhorse. All of E3.** |
| RandomForest | `numTrees=50, maxDepth=10` | Headline table only |

> Legacy `LandCoverClassifier.scala` uses `numTrees=100, maxDepth=20`. On 237,871 rows ×
> 110 features that is ~8–25 min **per label** → ~4 h per configuration → E3 alone would
> take 30+ hours. Depth 20 also overfits at this feature count. **Do not reuse those values.**

Why LR for E3 is methodologically correct, not just fast: the split audit measures
*relative decay* across regimes. That requires a **consistent** model, not the strongest
one. Say this explicitly in the write-up — it is a defensible choice, not a shortcut.

### 2.4 Threshold tuning

Per-label threshold maximising F1, **tuned on validation, applied to test.** Assert in
code that the test split is never touched during tuning. AP is threshold-free; F1 is not.

### 2.5 ▶ DAY 2 GATE

- [ ] End-to-end run completes on the subset
- [ ] `AP^μ` is in a plausible band (0.60–0.85). Near 0.5 ⇒ labels/features misaligned
- [ ] **Launch full extraction before you sleep.** `--subset-fraction 1.0`, ~1 hour

---

## 3. DAY 3 — Thursday 17 Sept ★ the result

### 3.1 Morning

Load full features to HDFS. Rerun E1 (BR-LR, official split) on all 480,038.

**E1 output — Table 1 of the report:**

| Model | AP^M | AP^μ | F1^M | F1^μ |
|---|---|---|---|---|
| ResNet-50 (Clasen et al., S2-only) | 70.72 | 85.86 | 64.74 | 76.34 |
| **BR-LogisticRegression (ours)** | ? | ? | ? | ? |
| **BR-RandomForest (ours)** | ? | ? | ? | ? |

Expect to land meaningfully below ResNet-50. **That is the expected and correct result** —
do not tune to close the gap. The gap is the input to the efficiency argument.

### 3.2 All day — E3, the split audit

Same model (BR-LR), same features, four regimes:

| Regime | Folds | Est. time |
|---|---|---|
| S1 Random | 1 | ~45 min |
| S2 Official | 1 | ~45 min |
| S3 Leave-tile-out | 5 | ~2 h |
| S4 Leave-country-out | 10 | ~3 h |

With parallel label training, S3+S4 run overnight comfortably.

**The headline number:** macro-AP decay from S2 (official) to S4 (leave-country-out).
Expect 10–25 points. Report per-country for S4 — Finland (32% of data) and Kosovo (0.3%)
are not comparable folds, so give both the fold-macro-average and the pooled figure.

### 3.3 Evening

Launch BR-RandomForest on S2 overnight (`numTrees=50, maxDepth=10`, ~1.5 h).

### 3.4 ▶ DAY 3 GATE

- [ ] E1 table filled for the full 480k
- [ ] E3 complete across all four regimes
- [ ] The S2→S4 gap is computed and written to `results/optimism_gap.json`

---

## 4. DAY 4 — Friday 18 Sept — evaluation and figures

### 4.1 Metrics

`src/python/eval/metrics.py`: `AP^M, AP^μ, F1^M, F1^μ`, per-class AP and F1, Hamming loss,
subset accuracy. Bootstrap 95% CIs, 1000 resamples.

**Validate against `sklearn.metrics.average_precision_score` on a synthetic case before
trusting a single number.**

### 4.2 The three figures that matter

| # | Figure | Content |
|---|---|---|
| **F1** ★ | **Split-strictness decay** | macro-AP on y, regimes S1→S2→S3→S4 on x, CI bands. **This is the paper in one image.** |
| **F2** ★ | **Europe choropleth** | 10 countries shaded by leave-country-out macro-AP. Shows *where* it fails. |
| **F3** | **Per-class AP** | Your best vs ResNet-50, 19 classes sorted by frequency. Shows the gap concentrates in rare classes. |

Optional if time allows: F4 label co-occurrence heatmap, F5 accuracy–compute Pareto.

**Rules:** vector PDF, 300 dpi, viridis/cividis (never rainbow), axis units labelled,
*n* stated in every caption, identical colour per model across figures.

### 4.3 Dashboard

Single self-contained `dashboard.html`, Plotly inlined, no server needed.
Tabs: Overview · Split audit · Per-class explorer.

### 4.4 ▶ DAY 4 GATE

- [ ] F1, F2, F3 rendered as PDF and PNG
- [ ] `dashboard.html` opens by double-click with no server

---

## 5. DAY 5 — Saturday 19 Sept — write-up, viva prep, buffer

### 5.1 Report / deck structure

```
1 Problem      multi-label land cover on reBEN; 480,038 patches, 19 classes
2 System       Docker + HDFS(RF=2) + Spark + Scala; streaming ingestion,
               63 GB → <1 GB features, no tar extraction
3 Method       110 physics-informed features incl. 4 red-edge indices;
               distributed binary relevance (MLlib has no multi-label estimator);
               parallel label training, measured Nx speedup
4 Finding 1    comparison vs published ResNet-50 (Table 1)
5 Finding 2 ★  reBEN's "geographic split" is concentric WITHIN tiles — train/val/test
               share all 54 tiles and all 10 countries. Measured optimism gap: X points.
6 Limitations  S2 only; hand-crafted features; single seed; LR workhorse; 10 countries
7 Future work  classifier chains, feature ablation, seasonal analysis, S1 fusion
```

**Write §6 honestly and at length.** Overclaiming is punished far harder than a modest,
well-bounded result.

### 5.2 Viva preparation — the questions you WILL be asked

Write your own answers out. These are the ones that decide the grade.

**Q1. "Your accuracy is below the published CNNs. Isn't your method just worse?"**
Yes, on accuracy — by design. A CNN learns features from raw pixels on a GPU; we use
110 hand-crafted features on CPUs. The contribution is not a better classifier, it is
the **audit**: we ran the same model under four split geometries, which the CNN papers
cannot afford to do because each of their runs costs GPU-days. Cheap training is what
buys the experiment. *Have the AP-per-CPU-hour ratio memorised.*

**Q2. ⚠ "Your features are only 500 MB. Why do you need HDFS and Spark at all?"**
The sharpest question you will get. **Do not bluff.** Honest answer:
The *input* is 63 GB compressed / 88 GB of raw pixels — that genuinely needs a streaming
distributed-storage design. The *derived features* are 500 MB, and yes, that table alone
would fit on one machine. What HDFS and Spark buy at this stage is (a) parallel training
across 19 labels × 4 split regimes × 16 folds, which is where the real compute sits, and
(b) a pipeline that scales unchanged to the full 549k-patch archive or to continental
Sentinel-2 coverage. Then add: *"the honest framing is that the feature extraction is the
big-data problem and the ML is the parallel-compute problem."* Admitting the limit
scores far better than defending an inflated claim.

**Q3. "What is actually novel here?"**
reBEN advertises a geographic split that "significantly reduces spatial correlation." We
measured it: it is **concentric within each tile** — test is the innermost square, train
the outer frame, and 46 of 54 tiles contain all three splits. So train and test share
every tile, every country, and every acquisition date. It removes adjacency leakage, not
regional leakage. We quantify how much optimism remains. *Show F1.*

**Q4. "Why binary relevance? Why not multi-class?"**
Because each patch has 2.95 labels on average, up to 11. It is inherently multi-label.
Spark MLlib has no multi-label estimator — only `OneVsRest`, which is multi-class, and
`MultilabelMetrics`, which only evaluates. So we implemented binary relevance: 19 binary
classifiers over a single shared, cached feature DataFrame, trained concurrently.

**Q5. "Why not just put the 63 GB archive into HDFS?"**
Three reasons. `.tar.zst` is **not splittable** — Hadoop's `ZStandardCodec` doesn't
implement `SplittableCompressionCodec`, so Spark creates one input split and one task
processes all 63 GB single-threaded. TAR has **no index**, so it is sequential-only.
And HDFS here runs in Docker on the same physical disk, so it would be a second copy,
not extra storage. We stream it once instead and store the derived features.

**Q6. "Why Parquet and not the original GeoTIFFs?"**
480k patches × 12 bands = 5.8 M small files. HDFS blocks are 128 MB; millions of
kilobyte-sized files exhaust NameNode memory and destroy throughput. Parquet is
columnar, compressed and splittable — 54 shards instead of 5.8 M files.

**Q7. "Why is performance different across your four splits?"**
Spatial autocorrelation. Neighbouring patches share land cover, atmosphere, phenology and
sensor geometry, so a random split lets the model memorise regional signatures rather
than learn generalisable spectral relationships. The stricter the geographic separation,
the closer the number gets to real operational performance.

**Q8. "Which classes fail, and why?"**
The rare ones — `Beaches, dunes, sands` (0.27%) and `Coastal wetlands` (0.29%) versus
`Arable land` at 39%. 145× imbalance. We use inverse-frequency class weights and tune
per-label thresholds on validation. *Show F3.*

**Q9. "What would you do differently with more time?"**
Classifier chains to exploit label correlation; feature ablation to isolate the red-edge
contribution; Sentinel-1 fusion (reBEN shows S1+S2 beats S2 alone); multiple seeds with
significance testing. All specified in `PLAN.md`.

**Q10. "Show me it running."**
Have ready: `docker ps` showing the cluster, the Spark UI at :8080, the HDFS UI at :9870,
and `dashboard.html` open in a browser. **Rehearse this once on Day 5** — do not discover
a broken container in front of the examiner.

### 5.3 Live-demo checklist

- [ ] `docker compose up -d` brings the cluster healthy from cold in under 3 minutes
- [ ] `hdfs dfsadmin -report` shows live DataNodes
- [ ] Spark history server shows a completed run with real executor CPU time
- [ ] `dashboard.html` opens by double-click, no server
- [ ] F1, F2, F3 printed or on a slide
- [ ] A one-page cheat sheet: dataset size, patch count, class count, your four metrics,
      the optimism gap, the speedup from parallel label training

Keep the rest of Day 5 free. Something will have broken.

---

## 6. Failure protocol

| If this breaks | Do this |
|---|---|
| Streaming extractor fights you >1 day | Extract 3 tiles conventionally (~2 GB), proceed on that. Ugly but submittable. |
| Day 1 gate fails (NDVI check) | **Stop everything.** Nothing downstream is valid. Fix band mapping first. |
| Full extraction fails overnight | Resume from `state/completed_tiles.json`. Submit the 10% subset result if needed. |
| Spark OOM | Reduce to 1 worker with more memory; train labels sequentially instead of in parallel. |
| Behind schedule on Day 3 | Drop S3. S1/S2/S4 still make the argument — S4 is the one that matters. |
| Behind schedule on Day 4 | Drop F3 and the dashboard. **F1 and F2 are non-negotiable.** |

---

## 7. After the 20th

The experiments are the perishable part — they need the machine, the archive, and a
working cluster. Writing does not. Once the deadline passes, return to `PLAN.md` for:

- the full Scala multi-label library (C1) — classifier chains, ECC, 2BR
- E2, E4, E5, E7
- Moran's I explanatory analysis
- 3 seeds and significance testing
- the paper, targeting IGARSS / JSTARS / *Remote Sensing* (MDPI) / BiDS

The 5-day sprint produces the **result**. The paper is written afterwards, unhurried.
