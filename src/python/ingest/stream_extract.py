"""
SPRINT.md Sec 1.3 -- streaming extractor. Never extracts the tar to disk.

    zstandard stream_reader -> tarfile mode='r|' (sequential) -> accumulate a
    patch's 12 bands in memory -> push to a bounded queue -> N-1 worker
    processes compute features -> a writer thread groups rows by archive
    folder and writes one Parquet shard per folder, checkpointing to
    state/completed_tiles.json for resume.

Run from the repo root:
    venv/Scripts/python.exe src/python/ingest/stream_extract.py --subset-fraction 0.1 --seed 42
"""
import argparse
import json
import multiprocessing as mp
import os
import queue as pyqueue
import sys
import tarfile
import threading
import time
from collections import defaultdict

import pandas as pd
import rasterio
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, BAND_ORDER  # noqa: E402
from patch_meta import parse_patch_id  # noqa: E402

REPO_ROOT = "C:/Users/sanji/Desktop/bda_project"
DEFAULT_ARCHIVE = f"{REPO_ROOT}/BigEarthNet-S2.tar.zst"
DEFAULT_METADATA = f"{REPO_ROOT}/metadata.parquet"

# Frozen alphabetical class order -- lbl_00..lbl_18. Must match all 19 labels
# found in metadata.parquet exactly (checked at startup).
CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
    "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
    "Industrial or commercial units", "Inland waters", "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]
assert len(CLASSES) == 19

DEBUG_MAX_MEMBERS = int(os.environ.get("STREAM_MAX_MEMBERS", "0")) or None  # test-only cap


def _worker_loop(task_q, result_q, meta_lookup):
    """Runs in a worker process. meta_lookup: patch_id -> (country, split, labels_frozenset)."""
    while True:
        batch = task_q.get()
        if batch is None:
            break
        for patch_id, folder, band_bytes in batch:
            try:
                band_arrays = {}
                for band in BAND_ORDER:
                    with rasterio.io.MemoryFile(band_bytes[band]) as mf:
                        with mf.open() as ds:
                            band_arrays[band] = ds.read(1)
                feats, _ = extract_features(band_arrays)
                country, split_official, labels = meta_lookup[patch_id]
                parsed = parse_patch_id(patch_id)
                row = {
                    "patch_id": patch_id, "tile": parsed["tile"], "gx": parsed["gx"],
                    "gy": parsed["gy"], "month": parsed["month"], "year": parsed["year"],
                    "country": country, "split_official": split_official,
                }
                for i, cls in enumerate(CLASSES):
                    row[f"lbl_{i:02d}"] = 1 if cls in labels else 0
                row.update(feats)
                result_q.put(("row", folder, row))
            except Exception as e:  # noqa: BLE001 -- surface any failure without killing the worker
                result_q.put(("error", folder, f"{patch_id}: {e!r}"))
    result_q.put(("worker_done", None, None))


def _flush_folder(folder, rows, output_dir):
    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{folder}.parquet")
    df.to_parquet(out_path, index=False)


def _writer_thread(result_q, folder_expected_q, output_dir, completed_path, completed, n_workers):
    os.makedirs(output_dir, exist_ok=True)
    folder_rows = defaultdict(list)
    folder_expected = {}
    workers_done = 0
    total_written = 0
    t0 = time.time()

    while True:
        try:
            while True:
                folder, expected = folder_expected_q.get_nowait()
                folder_expected[folder] = expected
                # rows for this folder may have already fully arrived before we
                # learned its expected count -- flush now instead of waiting for
                # a next row that will never come.
                if folder in folder_rows and len(folder_rows[folder]) >= expected:
                    _flush_folder(folder, folder_rows.pop(folder), output_dir)
                    completed.add(folder)
                    with open(completed_path, "w") as f:
                        json.dump(sorted(completed), f)
        except pyqueue.Empty:
            pass

        try:
            kind, folder, payload = result_q.get(timeout=1.0)
        except pyqueue.Empty:
            if workers_done >= n_workers:
                break
            continue

        if kind == "worker_done":
            workers_done += 1
            continue
        if kind == "error":
            print(f"[worker error] {payload}", flush=True)
            continue

        folder_rows[folder].append(payload)
        total_written += 1
        if total_written % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [writer] {total_written:,} patches processed, "
                  f"{total_written / elapsed:.1f} patches/sec", flush=True)

        expected = folder_expected.get(folder)
        if expected is not None and len(folder_rows[folder]) >= expected:
            _flush_folder(folder, folder_rows.pop(folder), output_dir)
            completed.add(folder)
            with open(completed_path, "w") as f:
                json.dump(sorted(completed), f)

    for folder, rows in folder_rows.items():
        print(f"[writer] WARNING: folder {folder} never reached its expected count "
              f"({len(rows)}/{folder_expected.get(folder)}) -- flushing what we have", flush=True)
        _flush_folder(folder, rows, output_dir)
        completed.add(folder)
    with open(completed_path, "w") as f:
        json.dump(sorted(completed), f)
    print(f"[writer] done. total patches written this run: {total_written:,}", flush=True)


def _build_target_ids(meta_df, subset_fraction, seed):
    if subset_fraction >= 1.0:
        return set(meta_df.index)
    tiles = pd.Series({pid: parse_patch_id(pid)["tile"] for pid in meta_df.index})
    sampled = tiles.groupby(tiles).apply(lambda g: g.sample(frac=subset_fraction, random_state=seed))
    return set(sampled.index.get_level_values(-1))


def _finish_folder(folder, count, folder_expected_q, completed):
    if folder is None or folder in completed or count == 0:
        return
    folder_expected_q.put((folder, count))


def stream_and_extract(archive_path, metadata_path, output_dir, state_dir,
                        subset_fraction, seed, n_workers, queue_maxsize, batch_size):
    meta_df = pd.read_parquet(metadata_path)
    if "patch_id" not in meta_df.columns:
        for alt in ("name", "patch_name"):
            if alt in meta_df.columns:
                meta_df["patch_id"] = meta_df[alt]
    meta_df = meta_df.set_index("patch_id")

    found_labels = set()
    for labels in meta_df["labels"]:
        found_labels.update(labels)
    if found_labels != set(CLASSES):
        raise ValueError(
            f"metadata labels don't match the frozen CLASSES order -- "
            f"extra={found_labels - set(CLASSES)} missing={set(CLASSES) - found_labels}"
        )

    target_ids = _build_target_ids(meta_df, subset_fraction, seed)
    print(f"Target patch set: {len(target_ids):,} of {len(meta_df):,} "
          f"(subset_fraction={subset_fraction}, seed={seed})")

    meta_lookup = {
        pid: (row["country"], row["split"], frozenset(row["labels"]))
        for pid, row in meta_df.loc[sorted(target_ids)].iterrows()
    }

    os.makedirs(state_dir, exist_ok=True)
    completed_path = os.path.join(
        state_dir,
        "completed_tiles_full.json" if subset_fraction >= 1.0 else f"completed_tiles_frac{int(subset_fraction * 100)}.json",
    )
    completed = set()
    if os.path.exists(completed_path):
        completed = set(json.load(open(completed_path)))
        print(f"Resuming: {len(completed)} folders already completed, skipping their patches")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=queue_maxsize)
    result_q = ctx.Queue()
    folder_expected_q = pyqueue.Queue()

    workers = [ctx.Process(target=_worker_loop, args=(task_q, result_q, meta_lookup)) for _ in range(n_workers)]
    for w in workers:
        w.start()

    writer = threading.Thread(
        target=_writer_thread,
        args=(result_q, folder_expected_q, output_dir, completed_path, completed, n_workers),
    )
    writer.start()

    current_folder = None
    folder_is_done = False
    folder_patch_count = 0
    current_patch_id = None
    current_bands = {}
    batch = []
    total_members = 0
    total_queued = 0
    t0 = time.time()

    with open(archive_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    total_members += 1
                    if DEBUG_MAX_MEMBERS and total_members > DEBUG_MAX_MEMBERS:
                        print(f"[test cap] stopping after {DEBUG_MAX_MEMBERS} members")
                        break
                    if not member.isreg():
                        continue
                    parts = member.name.split("/")
                    if len(parts) < 3:
                        continue
                    folder, patch_id, filename = parts[-3], parts[-2], parts[-1]

                    if folder != current_folder:
                        _finish_folder(current_folder, folder_patch_count, folder_expected_q, completed)
                        current_folder = folder
                        folder_patch_count = 0
                        folder_is_done = folder in completed

                    if folder_is_done or patch_id not in target_ids:
                        continue

                    if patch_id != current_patch_id:
                        current_patch_id = patch_id
                        current_bands = {}

                    band = filename.rsplit("_", 1)[-1].replace(".tif", "")
                    if band not in BAND_ORDER:
                        continue
                    current_bands[band] = tar.extractfile(member).read()

                    if len(current_bands) == len(BAND_ORDER):
                        batch.append((patch_id, folder, current_bands))
                        folder_patch_count += 1
                        total_queued += 1
                        current_bands = {}
                        current_patch_id = None
                        if len(batch) >= batch_size:
                            task_q.put(batch)
                            batch = []
                        if total_queued % 5000 == 0:
                            elapsed = time.time() - t0
                            print(f"  [reader] {total_queued:,} patches queued, "
                                  f"{total_queued / elapsed:.1f} patches/sec", flush=True)

    if batch:
        task_q.put(batch)
    _finish_folder(current_folder, folder_patch_count, folder_expected_q, completed)

    for _ in range(n_workers):
        task_q.put(None)
    for w in workers:
        w.join()
    writer.join()
    print(f"[reader] streamed {total_members:,} members, queued {total_queued:,} patches "
          f"in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-path", default=DEFAULT_ARCHIVE)
    p.add_argument("--metadata-path", default=DEFAULT_METADATA)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--state-dir", default=f"{REPO_ROOT}/state")
    p.add_argument("--subset-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 3))
    p.add_argument("--queue-maxsize", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        tag = "full" if args.subset_fraction >= 1.0 else f"subset_{int(args.subset_fraction * 100)}pct"
        output_dir = f"{REPO_ROOT}/features/{tag}"

    stream_and_extract(
        archive_path=args.archive_path,
        metadata_path=args.metadata_path,
        output_dir=output_dir,
        state_dir=args.state_dir,
        subset_fraction=args.subset_fraction,
        seed=args.seed,
        n_workers=args.workers,
        queue_maxsize=args.queue_maxsize,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
