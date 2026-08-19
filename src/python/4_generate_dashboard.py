import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "C:/Users/sanji/Desktop/bda_project/bigearthnet_subset"

print(f"Loading metrics from {DATA_DIR}...")
try:
    cm_df = pd.read_csv(f"{DATA_DIR}/confusion_matrix.csv")
    report_df = pd.read_csv(f"{DATA_DIR}/classification_report_metrics.csv", index_col=0)
except Exception as e:
    print(f"Error loading metrics: {e}")
    exit(1)

# Set up the figure for the dashboard (2 subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('BigEarthNet Land Cover Classification Dashboard (Spark Random Forest)', fontsize=16, fontweight='bold')

# --- 1. Plot Confusion Matrix ---
sns.heatmap(cm_df, annot=False, cmap='Blues', ax=ax1, cbar=True)
ax1.set_title('Confusion Matrix', fontsize=14)
ax1.set_xlabel('Predicted Label Index')
ax1.set_ylabel('True Label Index')

# --- 2. Plot F1-Scores for the classes ---
# Drop the averages to just show the class F1 scores
classes_only = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
# Sort by support (number of samples)
classes_only = classes_only.sort_values(by='support', ascending=False)

sns.barplot(x=classes_only['f1-score'], y=classes_only.index, ax=ax2, palette='viridis')
ax2.set_title('F1-Score by Class (Sorted by Frequency)', fontsize=14)
ax2.set_xlabel('F1-Score')
ax2.set_ylabel('Class')

plt.tight_layout()
dashboard_path = os.path.join(DATA_DIR, "training_dashboard.png")
plt.savefig(dashboard_path, dpi=300)
print(f"\nDashboard successfully saved to: {dashboard_path}")
plt.show()
