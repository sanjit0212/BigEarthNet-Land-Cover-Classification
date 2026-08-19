import rasterio
import numpy as np
import pandas as pd
import os

BANDS = ["B02", "B03", "B04", "B08"]  # blue, green, red, NIR
DATA_DIR = "C:/Users/sanji/Desktop/bda_project/bigearthnet_subset"

def extract_features(patch_id):
    arrs = {}
    for b in BANDS:
        file_path = f"{DATA_DIR}/{patch_id}/{patch_id}_{b}.tif"
        if not os.path.exists(file_path):
            file_path = f"{DATA_DIR}/{patch_id}/{b}.tif" # fallback
            
        try:
            with rasterio.open(file_path) as src:
                arrs[b] = src.read(1).astype(np.float32) / 10000.0  # reflectance scale
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # return dummy zeroes if file is missing (to not break the pipeline)
            arrs[b] = np.zeros((120, 120), dtype=np.float32)

    red, nir, green = arrs["B04"], arrs["B08"], arrs["B03"]
    
    # Avoid division by zero
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)

    row = {"patch_id": patch_id, "ndvi": float(np.mean(ndvi)), "ndwi": float(np.mean(ndwi))}
    
    for b in BANDS:
        row[f"{b.lower()}_mean"] = float(np.mean(arrs[b]))
        row[f"{b.lower()}_std"] = float(np.std(arrs[b]))
        
    return row

if __name__ == "__main__":
    print(f"Reading metadata from {DATA_DIR}...")
    meta = pd.read_parquet(f"{DATA_DIR}/subset_40k_metadata.parquet")
    
    print(f"Extracting features for {len(meta)} patches... This will be much faster now!")
    
    import multiprocessing
    from multiprocessing import Pool
    
    # Use all available CPU cores!
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing...")
    
    with Pool(num_cores) as p:
        rows = p.map(extract_features, meta["patch_id"].tolist())
        
    feat_df = pd.DataFrame(rows)

    final = feat_df.merge(meta[["patch_id", "dominant_label"]], on="patch_id")
    
    out_path = f"{DATA_DIR}/features_40k.parquet"
    final.to_parquet(out_path)
    
    print(f"Features saved to {out_path} with shape {final.shape}")
    print("\nTo push to HDFS, run these docker commands:")
    print(f"docker cp {out_path} namenode:/tmp/")
    print("docker exec namenode hdfs dfs -mkdir -p /bigearthnet/features")
    print("docker exec namenode hdfs dfs -put /tmp/features_40k.parquet /bigearthnet/features/")
