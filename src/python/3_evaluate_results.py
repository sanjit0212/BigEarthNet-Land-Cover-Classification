import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "C:/Users/sanji/Desktop/bda_project/bigearthnet_subset"

print(f"Loading predictions from {DATA_DIR}...")
try:
    preds = pd.read_csv(f"{DATA_DIR}/predictions.csv")
except Exception as e:
    print(f"Error reading predictions: {e}")
    print("Ensure you have run the docker command to copy the results out of HDFS first!")
    exit(1)

# Map the numeric prediction back to the string label
label_map = preds[['label', 'dominant_label']].drop_duplicates().set_index('label')['dominant_label'].to_dict()
preds['predicted_label'] = preds['prediction'].map(label_map)

print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(preds["dominant_label"], preds["predicted_label"])
print(report)

print("Generating confusion matrix...")
cm = confusion_matrix(preds["dominant_label"], preds["predicted_label"])
cm_df = pd.DataFrame(cm)
cm_df.to_csv(f"{DATA_DIR}/confusion_matrix.csv", index=False)

# Export the classification report to CSV for PowerBI/Tableau as well
report_dict = classification_report(preds["dominant_label"], preds["predicted_label"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{DATA_DIR}/classification_report_metrics.csv", index=True)

print(f"Exported confusion_matrix.csv and classification_report_metrics.csv to {DATA_DIR}!")
print("These are ready to be connected to Power BI or Tableau.")
