import tarfile
import zstandard as zstd
import json
import os
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

ARCHIVE_PATH = "C:/Users/sanji/Desktop/bda_project/BigEarthNet-S2.tar.zst"
METADATA_PATH = "C:/Users/sanji/Desktop/bda_project/metadata.parquet"
OUTPUT_DIR = "C:/Users/sanji/Desktop/bda_project/bigearthnet_subset"
TARGET_SIZE = 40000

def get_metadata():
    print(f"Loading metadata directly from {METADATA_PATH}...")
    df = pd.read_parquet(METADATA_PATH)
    
    # Ensure it has 'patch_id' and 'labels'
    if 'patch_id' not in df.columns:
        if 'name' in df.columns:
            df['patch_id'] = df['name']
        elif 'patch_name' in df.columns:
            df['patch_id'] = df['patch_name']
            
    print(f"Total patches found: {len(df)}")
    return df

def stratify_subset(df):
    print("Performing fast stratified split based on dominant label...")
    
    # 1. Derive dominant label for stratification
    def get_dominant(labels_val):
        if isinstance(labels_val, str):
            labels_val = json.loads(labels_val)
        if isinstance(labels_val, np.ndarray):
            labels_val = labels_val.tolist()
        return labels_val[0] if len(labels_val) > 0 else "Unknown"
        
    df['dominant_label'] = df['labels'].apply(get_dominant)
    
    # 2. Stratified sample
    # Calculate proportions
    proportions = df['dominant_label'].value_counts(normalize=True)
    
    # Sample based on proportions
    subset_df = pd.DataFrame()
    for label, prop in proportions.items():
        n_samples = int(TARGET_SIZE * prop)
        if n_samples > 0:
            label_df = df[df['dominant_label'] == label]
            sampled = label_df.sample(n=min(n_samples, len(label_df)), random_state=42)
            subset_df = pd.concat([subset_df, sampled])
            
    # Fill remaining to reach exactly TARGET_SIZE if rounding caused a slight deficit
    deficit = TARGET_SIZE - len(subset_df)
    if deficit > 0:
        remaining = df[~df['patch_id'].isin(subset_df['patch_id'])]
        extra = remaining.sample(n=deficit, random_state=42)
        subset_df = pd.concat([subset_df, extra])
        
    print(f"Subset created with {len(subset_df)} patches.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subset_df.to_parquet(f"{OUTPUT_DIR}/subset_40k_metadata.parquet")
    return set(subset_df['patch_id'].tolist())

def extract_subset(subset_ids):
    print(f"Extracting exactly {len(subset_ids)} patches to {OUTPUT_DIR}...")
    print("Streaming the .tar.zst archive...")
    
    extracted_count = 0
    with open(ARCHIVE_PATH, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                for member in tar:
                    parts = member.name.split('/')
                    patch_id = None
                    for p in parts:
                        if p in subset_ids:
                            patch_id = p
                            break
                            
                    if patch_id and member.isreg():
                        filename = parts[-1]
                        
                        # Only extract the 4 bands we actually need for the Spark ML pipeline!
                        if not any(filename.endswith(f"_{b}.tif") for b in ["B02", "B03", "B04", "B08"]):
                            continue
                            
                        out_dir = os.path.join(OUTPUT_DIR, patch_id)
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, filename)
                        
                        with open(out_path, 'wb') as out_f:
                            f_in = tar.extractfile(member)
                            if f_in:
                                out_f.write(f_in.read())
                                
                        if filename.endswith('.tif'):
                            extracted_count += 1
                            if extracted_count % 1000 == 0:
                                print(f"Extracted {extracted_count} TIF files...")

    print("Extraction complete!")
    if extracted_count == 0:
        print("ERROR: Extracted 0 files! The D: drive likely disconnected during the stream. Please copy the archive to C:.")
        exit(1)

if __name__ == "__main__":
    if not os.path.exists(ARCHIVE_PATH):
        print(f"ERROR: Could not find archive at {ARCHIVE_PATH}")
        exit(1)
        
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Could not find metadata at {METADATA_PATH}")
        exit(1)
        
    df = get_metadata()
    subset_ids = stratify_subset(df)
    extract_subset(subset_ids)
    print("SUCCESS: Stage 0 complete! You can now move on to Stage 1.")
