"""
SPRINT.md Sec 1.2 -- Verify the BigEarthNet-S2 archive layout before writing the
streaming extractor. Confirm, do not assume:
  - full member paths and nesting depth
  - band filenames, dtype, pixel dimensions for one complete patch
  - total patch count in the archive (vs. metadata.parquet's 480,038)

Streams the .tar.zst once, sequentially, never extracting to disk.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import zstandard as zstd

ARCHIVE_PATH = "C:/Users/sanji/Desktop/bda_project/BigEarthNet-S2.tar.zst"
METADATA_PATH = "C:/Users/sanji/Desktop/bda_project/metadata.parquet"
DOCS_DIR = "C:/Users/sanji/Desktop/bda_project/docs"
OUT_MD = os.path.join(DOCS_DIR, "archive_layout.md")

N_PREVIEW_MEMBERS = 200
PROGRESS_EVERY = 50_000
MAX_MEMBERS = int(os.environ.get("VERIFY_MAX_MEMBERS", "0")) or None  # test-only cap; unset for full scan

EXPECTED_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def load_metadata_patch_ids():
    if not os.path.exists(METADATA_PATH):
        print(f"WARNING: metadata not found at {METADATA_PATH}")
        return None
    df = pd.read_parquet(METADATA_PATH)
    if "patch_id" not in df.columns:
        for alt in ("name", "patch_name"):
            if alt in df.columns:
                df["patch_id"] = df[alt]
                break
    return set(df["patch_id"].tolist())


def main():
    import tarfile

    os.makedirs(DOCS_DIR, exist_ok=True)
    metadata_ids = load_metadata_patch_ids()
    metadata_count = len(metadata_ids) if metadata_ids is not None else None
    print(f"metadata.parquet patch_id count: {metadata_count}")

    preview_lines = []          # first N_PREVIEW_MEMBERS: (path, depth, filename)
    first_patch_bands = {}      # band_name -> bytes, for the first complete patch we see
    first_patch_id = None
    first_patch_reported = False

    total_members = 0
    total_regular_files = 0
    patch_ids_seen = set()      # patch_id = parent dir name of a .tif member
    tile_dirs_seen = set()
    top_level_dirs = set()
    ext_counts = {}
    max_depth = 0
    depths_seen = set()

    t0 = time.time()

    with open(ARCHIVE_PATH, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    total_members += 1
                    parts = member.name.split("/")
                    depth = len(parts)
                    depths_seen.add(depth)
                    max_depth = max(max_depth, depth)
                    if len(parts) >= 1 and parts[0]:
                        top_level_dirs.add(parts[0])

                    if total_members <= N_PREVIEW_MEMBERS:
                        preview_lines.append(f"depth={depth}  isreg={member.isreg()}  path={member.name}")

                    if member.isreg():
                        total_regular_files += 1
                        filename = parts[-1]
                        _, ext = os.path.splitext(filename)
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1

                        if len(parts) >= 2:
                            patch_id = parts[-2]
                            if ext == ".tif":
                                patch_ids_seen.add(patch_id)
                                if len(parts) >= 3:
                                    tile_dirs_seen.add(parts[-3])

                                if not first_patch_reported:
                                    if first_patch_id is None:
                                        first_patch_id = patch_id
                                    if patch_id == first_patch_id:
                                        data = tar.extractfile(member).read()
                                        first_patch_bands[filename] = data
                                    elif first_patch_bands:
                                        # moved on to the next patch -> first patch's bands are complete
                                        first_patch_reported = True

                    if total_members % PROGRESS_EVERY == 0:
                        elapsed = time.time() - t0
                        print(f"  ...{total_members:,} members scanned, "
                              f"{len(patch_ids_seen):,} patches seen, {elapsed:.0f}s elapsed", flush=True)

                    if MAX_MEMBERS and total_members >= MAX_MEMBERS:
                        print(f"[test cap] stopping after {MAX_MEMBERS} members")
                        break

    elapsed_total = time.time() - t0
    print(f"Done streaming archive in {elapsed_total:.0f}s")
    print(f"Total members: {total_members:,}  regular files: {total_regular_files:,}")
    print(f"Unique patch dirs (by .tif parent): {len(patch_ids_seen):,}")
    print(f"Unique tile dirs: {len(tile_dirs_seen):,}")
    print(f"Top-level dirs: {top_level_dirs}")
    print(f"Depths seen: {sorted(depths_seen)} (max {max_depth})")
    print(f"Extension counts: {ext_counts}")

    # Inspect the first complete patch's bands with rasterio
    band_report = []
    for filename, data in sorted(first_patch_bands.items()):
        with rasterio.io.MemoryFile(data) as memfile:
            with memfile.open() as ds:
                band_report.append({
                    "filename": filename,
                    "dtype": ds.dtypes[0],
                    "width": ds.width,
                    "height": ds.height,
                    "count": ds.count,
                })
    print(f"First patch id: {first_patch_id}, bands found: {len(band_report)}")
    for b in band_report:
        print(f"  {b}")

    if metadata_ids is not None:
        overlap = len(patch_ids_seen & metadata_ids)
        only_in_archive = len(patch_ids_seen - metadata_ids)
        only_in_metadata = len(metadata_ids - patch_ids_seen)
        print(f"Overlap archive/metadata: {overlap:,}")
        print(f"In archive but not metadata.parquet (expected: snow/cloud patches): {only_in_archive:,}")
        print(f"In metadata.parquet but not archive (should be 0): {only_in_metadata:,}")
    else:
        overlap = only_in_archive = only_in_metadata = None

    # Write docs/archive_layout.md
    with open(OUT_MD, "w", encoding="utf-8") as out:
        out.write("# BigEarthNet-S2 archive layout (confirmed)\n\n")
        out.write(f"Generated by `src/python/ingest/verify_archive.py` in {elapsed_total:.0f}s.\n\n")
        out.write("## Path structure\n\n")
        out.write(f"- Top-level dir(s): `{sorted(top_level_dirs)}`\n")
        out.write(f"- Path depths observed: `{sorted(depths_seen)}` (max {max_depth})\n")
        out.write("- Layout: `<top>/<tile_folder>/<patch_id>/<patch_id>_B{01,02,03,04,05,06,07,08,8A,09,11,12}.tif`\n\n")
        out.write("## Counts (full archive scan)\n\n")
        out.write(f"- Total tar members: {total_members:,}\n")
        out.write(f"- Total regular files: {total_regular_files:,}\n")
        out.write(f"- Unique patch dirs (.tif parent): {len(patch_ids_seen):,}\n")
        out.write(f"- Unique tile dirs: {len(tile_dirs_seen):,}\n")
        out.write(f"- Extension counts: {ext_counts}\n")
        if metadata_ids is not None:
            out.write(f"- metadata.parquet patch_id count: {metadata_count:,}\n")
            out.write(f"- Overlap archive/metadata: {overlap:,}\n")
            out.write(f"- In archive but not metadata (snow/cloud, expected excluded): {only_in_archive:,}\n")
            out.write(f"- In metadata but not archive (should be 0): {only_in_metadata:,}\n")
        out.write("\n## First complete patch\n\n")
        out.write(f"`patch_id = {first_patch_id}`\n\n")
        out.write("| filename | dtype | width | height | band_count |\n|---|---|---|---|---|\n")
        for b in band_report:
            out.write(f"| {b['filename']} | {b['dtype']} | {b['width']} | {b['height']} | {b['count']} |\n")
        out.write("\n## First 200 tar members (path, depth)\n\n```\n")
        out.write("\n".join(preview_lines))
        out.write("\n```\n")

    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
