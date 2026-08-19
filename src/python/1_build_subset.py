import pandas as pd
import json

DATA_DIR = "C:/Users/sanji/Desktop/bda_project/bigearthnet_subset"

print(f"Loading metadata from {DATA_DIR}...")
# Assuming the user has a master metadata file or we just use the subset directly.
# The user prompt assumes 'subset_40k_metadata.parquet' already exists from a previous step,
# but we will create a robust script that adds the dominant label.
meta = pd.read_parquet(f"{DATA_DIR}/subset_40k_metadata.parquet")

def dominant_label(row):
    # If BigEarthNet gives per-class pixel coverage/area in 'label_areas', pick the max
    # If it's just a binary list of labels, we pick the first one as a simplified baseline
    
    if "label_areas" in row and isinstance(row["label_areas"], dict):
        coverage = row["label_areas"]
        return max(coverage, key=coverage.get)
    else:
        # Fallback: Just pick the first label in the multi-label array
        labels = row["labels"]
        if isinstance(labels, str):
            labels = json.loads(labels)
        return labels[0]

meta["dominant_label"] = meta.apply(dominant_label, axis=1)
meta.to_parquet(f"{DATA_DIR}/subset_40k_metadata.parquet")

print("Dominant labels derived and saved to Parquet!")
print(meta["dominant_label"].value_counts())
