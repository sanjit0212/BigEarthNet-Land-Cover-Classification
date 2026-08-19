import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Create document
doc = Document()

# Add a Title Page
doc.add_heading('Big Data Analytics Project', 0)
doc.add_heading('End-to-End BigEarthNet Land Cover Classification using Hadoop and Apache Spark', 1)
doc.add_paragraph('\n')
doc.add_paragraph('A Distributed Machine Learning Pipeline', style='Subtitle')
doc.add_paragraph('\n\n\n')
doc.add_paragraph('Technologies Used: Apache Hadoop (HDFS), Apache Spark (MLlib), Scala, Python, Pandas, Rasterio, Scikit-Learn, Docker')
doc.add_page_break()

# Chapter 1
doc.add_heading('CHAPTER 1 — INTRODUCTION', level=1)
doc.add_paragraph('Background: Land cover classification is a fundamental task in remote sensing. With the massive influx of satellite imagery (such as the Sentinel-2 missions), processing these large-scale datasets on a single machine is computationally prohibitive.')
doc.add_paragraph('Problem: Satellite imagery archives like BigEarthNet are massive (80+ GB) and consist of hundreds of thousands of tiny individual files (TIFF patches). Traditional machine learning approaches struggle with loading, preprocessing, and training on such massive, heavily-distributed file architectures.')
doc.add_paragraph('Proposed Solution: This project implements a distributed Big Data pipeline. We overcome the "Small File Problem" by extracting aggregated spectral features locally in parallel and packaging them into a columnar Parquet format. We then deploy a distributed Hadoop and Spark cluster using Docker to train a Random Forest classifier at scale using Spark MLlib.')
doc.add_page_break()

# Chapter 2 & 3
doc.add_heading('CHAPTER 2 — PROBLEM STATEMENT', level=1)
doc.add_paragraph('How can we efficiently process and classify massive volumes of distributed satellite imagery patches into land cover categories using distributed Big Data infrastructure?')

doc.add_heading('CHAPTER 3 — OBJECTIVES', level=1)
doc.add_heading('Primary Objectives:', level=2)
doc.add_paragraph('1. Construct an automated feature extraction pipeline using Rasterio for Sentinel-2 multi-band imagery.', style='List Number')
doc.add_paragraph('2. Deploy a distributed Hadoop Distributed File System (HDFS) and Apache Spark cluster using Docker.', style='List Number')
doc.add_paragraph('3. Train a distributed Random Forest model using Spark MLlib to classify land cover categories.', style='List Number')
doc.add_heading('Secondary Objectives:', level=2)
doc.add_paragraph('1. Handle the HDFS "Small File Problem" via Parquet aggregation.', style='List Number')
doc.add_paragraph('2. Conduct Hyperparameter Tuning across Spark worker nodes.', style='List Number')
doc.add_paragraph('3. Generate visual evaluation dashboards.', style='List Number')
doc.add_page_break()

# Chapter 4 & 5
doc.add_heading('CHAPTER 4 — LITERATURE / EXISTING APPROACHES', level=1)
doc.add_paragraph('Existing approaches often rely on deep learning (CNNs) trained on single, massive GPU workstations. While effective, they are bottlenecked by vertical scaling. This project differs by focusing on horizontal scaling: relying on distributed computing (Hadoop/Spark) for Big Data processing of spectral features (NDVI, NDWI).')

doc.add_heading('CHAPTER 5 — PROPOSED SYSTEM', level=1)
doc.add_paragraph('1. Data Ingestion: Sentinel-2 patches are sampled (40,000 images).')
doc.add_paragraph('2. Feature Extraction (Python/Rasterio): Calculate NDVI, NDWI, Mean, and Standard Deviation for Bands B02, B03, B04, and B08. Saved to a single massive Parquet file.')
doc.add_paragraph('3. Distributed Storage: The Parquet file is uploaded to the HDFS NameNode.')
doc.add_paragraph('4. Distributed ML Training (Scala/Spark): Spark Master coordinates with Spark Workers to ingest the Parquet data, construct vector assemblies, and train a Random Forest model using Cross-Validation.')
doc.add_paragraph('5. Evaluation (Scikit-Learn/Seaborn): Extract Spark predictions from HDFS via getmerge and visualize using Python dashboards.')
doc.add_page_break()

# Chapter 6 & 7
doc.add_heading('CHAPTER 6 — TECHNOLOGIES AND TOOLS', level=1)
table = doc.add_table(rows=1, cols=3, style='Table Grid')
hdr = table.rows[0].cells
hdr[0].text = 'Technology / Tool'
hdr[1].text = 'Purpose'
hdr[2].text = 'Where Used'

techs = [
    ('Python & Pandas', 'Data preprocessing and evaluation', 'Local Scripts'),
    ('Rasterio', 'Read/Process .tif satellite bands', 'Feature Extraction'),
    ('Docker Compose', 'Deploy distributed infrastructure', 'Cluster Setup'),
    ('Hadoop HDFS', 'Distributed Storage', 'NameNode / DataNodes'),
    ('Scala & SBT', 'Build and run Spark Jobs', 'Model Training'),
    ('Apache Spark MLlib', 'Distributed Machine Learning', 'Model Training'),
    ('Scikit-Learn & Seaborn', 'Metric evaluation & Visuals', 'Dashboard Generation')
]
for t1, t2, t3 in techs:
    r = table.add_row().cells
    r[0].text, r[1].text, r[2].text = t1, t2, t3

doc.add_heading('CHAPTER 7 — DATASET / INPUT DATA', level=1)
doc.add_paragraph('Dataset: BigEarthNet Subset (Sentinel-2)')
doc.add_paragraph('Size: 40,000 image patches (Sampled from the original ~590k patches)')
doc.add_paragraph('Format: Multi-band .tif files -> Aggregated to .parquet')
doc.add_paragraph('Features Extracted (10): ndvi, ndwi, b02_mean, b02_std, b03_mean, b03_std, b04_mean, b04_std, b08_mean, b08_std')
doc.add_paragraph('Target: 19 Corine Land Cover categories (e.g., Arable land, Coniferous forest, Urban fabric)')
doc.add_page_break()

# Chapter 8, 9, 10
doc.add_heading('CHAPTER 8 — METHODOLOGY & IMPLEMENTATION', level=1)
doc.add_paragraph('Step 1: Subset Generation (0_create_subset_from_archive.py, 1_build_subset.py)')
doc.add_paragraph('Extracts a random 40k sample of BigEarthNet to avoid running out of local disk space.')

doc.add_paragraph('Step 2: Distributed Feature Extraction (2_extract_features.py)')
doc.add_paragraph('Uses Python multiprocessing to compute Normalized Difference Vegetation Index (NDVI) and Normalized Difference Water Index (NDWI) from Near-Infrared (B08), Red (B04), and Green (B03) bands to generate highly-correlated vegetation/water markers. It produces features_40k.parquet.')

doc.add_paragraph('Step 3: Spark MLlib Modeling (LandCoverClassifier.scala)')
doc.add_paragraph('1. Ingests HDFS Parquet data.\n2. Uses StringIndexer to convert labels to numeric categories.\n3. Uses VectorAssembler to combine the 10 features into a feature vector.\n4. Trains a RandomForestClassifier using CrossValidator (3 folds, ParamGrid: Trees=[50,100,150], Depth=[5,10,15]).\n5. Saves predictions to HDFS.')

doc.add_paragraph('Step 4: Evaluation Dashboard (3_evaluate_results.py, generate_dashboard.py)')
doc.add_paragraph('Parses the Spark predictions output and uses sklearn.metrics to generate a classification report and confusion matrix. Uses seaborn to plot the dashboard.')
doc.add_page_break()

# Chapter 11 & 12
doc.add_heading('CHAPTER 11 — EXPERIMENTAL SETUP', level=1)
doc.add_paragraph('Hardware (Local Simulation): Windows Host, Multi-core CPU')
doc.add_paragraph('Distributed Environment: Docker Engine')
doc.add_paragraph('- Hadoop: 1 NameNode, 3 DataNodes (bde2020 images)')
doc.add_paragraph('- Spark: 1 Spark Master, 1 Spark Worker (2 Cores, 4GB RAM)')
doc.add_paragraph('Train/Test Split: 75% / 25% (30,000 train, 10,000 test) using Seed=42')

doc.add_heading('CHAPTER 12 — RESULTS', level=1)
doc.add_paragraph('The Spark Random Forest pipeline achieved an overall accuracy of 68% across all 19 classes using only 10 extracted spectral features.')
doc.add_paragraph('Model Training Runtime: ~30 Minutes (distributed on Spark Worker)')

doc.add_heading('Classification Metrics (Top Classes):')
res_table = doc.add_table(rows=1, cols=5, style='Table Grid')
r_hdr = res_table.rows[0].cells
r_hdr[0].text, r_hdr[1].text, r_hdr[2].text, r_hdr[3].text, r_hdr[4].text = 'Class', 'Precision', 'Recall', 'F1-Score', 'Support'
data = [
    ('Marine waters', '0.94', '0.97', '0.96', '1128'),
    ('Arable land', '0.64', '0.84', '0.72', '3484'),
    ('Coniferous forest', '0.69', '0.77', '0.73', '1981'),
    ('Broad-leaved forest', '0.65', '0.49', '0.56', '1523')
]
for d1, d2, d3, d4, d5 in data:
    row = res_table.add_row().cells
    row[0].text, row[1].text, row[2].text, row[3].text, row[4].text = d1, d2, d3, d4, d5

doc.add_page_break()

# Result Images
doc.add_heading('13. RESULT IMAGES', level=1)
doc.add_paragraph('Figure 1: Training Dashboard Output (Confusion Matrix and F1-Scores by Class)')

img_path = r"C:\Users\sanji\Desktop\bda_project\bigearthnet_subset\training_dashboard.png"
if os.path.exists(img_path):
    doc.add_picture(img_path, width=Inches(6.0))
    p = doc.add_paragraph('Figure 1. The confusion matrix (left) and F1-score chart (right) demonstrating the model performance on the hold-out test set.')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
else:
    doc.add_paragraph('[Image missing - training_dashboard.png not found at expected path]')

doc.add_page_break()

# Chapter 13 & 16
doc.add_heading('CHAPTER 13 — RESULT ANALYSIS / DISCUSSION', level=1)
doc.add_paragraph('The results indicate that the Random Forest model performs exceptionally well on distinct, heavily represented classes (e.g., Marine Waters F1=0.96, Arable Land F1=0.72). However, performance drops significantly for minority classes (e.g., Coastal wetlands, Urban fabric) where the Support size was less than 20 images. This demonstrates the extreme class imbalance of real-world remote sensing data.')

doc.add_heading('CHAPTER 16 — LIMITATIONS', level=1)
doc.add_paragraph('1. Class Imbalance: The model overfits to majority classes (Arable Land) and ignores rare classes (Urban Fabric) to maximize overall accuracy.')
doc.add_paragraph('2. Feature Engineering: Hand-crafted features (NDVI, NDWI) capture spectral signatures well, but lose spatial and textural features that a Deep Learning CNN (like ResNet) would capture.')
doc.add_paragraph('3. Compute Limitation: The Spark worker was limited to 2 cores, preventing training on the full 80GB dataset.')

doc.add_heading('CHAPTER 17 — FUTURE WORK', level=1)
doc.add_paragraph('- Scale to Full Dataset: Ingest the full 590k image dataset into a cloud cluster (AWS EMR/Databricks).')
doc.add_paragraph('- Advanced Modeling: Use Spark MLlib to implement class weighting or SMOTE to handle class imbalance.')

doc.add_heading('CHAPTER 18 — CONCLUSION', level=1)
doc.add_paragraph('This project successfully demonstrated the implementation of a full-scale Big Data pipeline for Earth Observation data. By bridging the gap between raw Python feature extraction, Dockerized Hadoop distributed storage, and Scala Spark MLlib training, it provides a highly scalable blueprint for modern remote sensing analysis.')

doc.save('C:/Users/sanji/Desktop/bda_project/Complete_Project_Documentation.docx')
print("Documentation successfully generated.")
