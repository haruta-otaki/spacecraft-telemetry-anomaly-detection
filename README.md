# Spacecraft Telemetry Anomaly Detection

This project implements an unsupervised machine learning approach to detect anomalies in multivariate spacecraft telemetry data. Overlapping temporal windows and MiniRocket feature transformation were used to capture complex temporal patterns in the time series data. Furthermore, an ensemble model was implemented to evaluate each channel, incorporating the following anomaly detection algorithms: One-Class Support Vector Machine, Isolation Forest, and Local
Outlier Factor. 

The [multivariate spacecraft telemetry data from the Soil Moisture Active Passive satellite](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl) was used to train and test the models featured in this project, containing point and contextual anomalies across nine distinct telemetry channels. 

The ensemble model achieved an average recall of 0.55 and F1 score of 0.21 across all channels; this is due to the threshold optimisation applied on the decision function outputted by the unsupervised machine learning algorithms. Although this introduces optimistic bias, the technique was implemented because of the small labelled test data for spacecraft anomalies as well as to explore recall-precision trade-offs (as prioritizing recall, reducing the likelihood of missing true anomalies, is cruicial in anomaly detection systems). 

---

## Project Overview

Spacecraft telemetry data is high-dimensional, noisy, and dominated by nominal behavior; hence, this project explores an unsupervised learning approach that combines:

- Sliding-window time-series preprocessing
- MiniRocket-based feature transformation 
- Ensemble of classical unsupervised anomaly detection models

---

## Repository Structure

```text
project-root/
│
├── src/
│   ├── preprocessing/        # Data loading and feature preparation
│   │   ├── load_data.py
│   │   ├── windowing.py
│   │   ├── normalization.py
│   │   └── minirocket_transform.py
│   │
│   ├── models/               # Unsupervised anomaly detection models
│   │   ├── one_class_svm.py
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   └── ensemble.py
│   │
│   ├── evaluation/           # Metrics, thresholding, and reporting
│   │   ├── metrics.py
│   │   ├── thresholding.py
│   │   └── reporting.py
│   │
│   └── utils/                # Shared utilities
│       └── io.py
│
├── data/
│   ├── raw/                  # Original SMAP / MSL datasets
│   │   ├── SMAP/
│   │   └── MSL/
│   │
│   ├── processed/            # Intermediate artifacts
│   │   ├── windowed/
│   │   ├── minirocket/
│   │   └── scaled/
│   │
│   └── README.md             # Dataset notes
│
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis
│
├── results/
│   ├── figures/              # Generated plots
│   └── metrics/              # CSV metrics per model/channel
│
├── requirements.txt
├── .gitignore
├── README.md
└── setup.py (optional)
```

---

## Pipeline Summary

### 1. Preprocessing
- Channel-wise extraction from raw telemetry, exclusively using telemetry data from
the SMAP satellite, as the SMAP and MSL datasets differ in sensor dimensionality (respectively containing 25 and 55 features). 

- Sliding-window segmentation to provide a context-aware and robust representation of anomalous behaviour, reducing sensitivity to short-lived fluctuations while ensuring that each timestamp is assessed within the context of its surrounding temporal neighbourhood rather than in isolation.

- Feature scaling and normalization, specifically Z-score scaling, to ensure that all
features were on comparable scales. 

### 2. Feature Extraction
- MiniRocket transforms time-series windows into fixed-length feature vectors, enabling fast training and inference while preserving temporal structure.

### 3. Models Implemented
- One-Class SVM 
- Isolation Forest
- Local Outlier Factor (LOF)
- Ensemble model

The ensumble model is designed to combine multiple detectors to leverage the complementary strengths of the three algorithms: One-Class SVM defines the boundary of normal data, Isolation Forest isolates anomalies via recursive partitioning, and LOF measures local density deviations. The ensemble applies a “maximum voting” strategy with a threshold of one vote, flagging a test example as anomalous if any model predicts it as suchprioritizing recall.

### 4. Evaluation Metrics
- Precision (anomalous class)
- Recall (anomalous class)
- Fβ-score 
---

## Results

- Metrics are stored as CSV files in `results/metrics/`
- Comparative bar plots are automatically generated and saved to `results/figures/`

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. Run preprocessing and feature extraction
2. Train models from `src/models/`
3. Evaluate and generate reports from `src/evaluation/`

---

## Future Work

This project is still in progress; currently, deep learning architectures, such as Long Short-Term Memory networks (LSTM) and transformer models are being incorporated in hopes of a better evaluation score. 
