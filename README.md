# TrailMakers_TrailMakers_k8s-failure-prediction

A robust AI/ML solution for predicting Kubernetes cluster failures, developed by TrailMakers for Hackathon Phase 1. This project leverages XGBoost and Isolation Forest to detect node/pod failures, resource exhaustion, network issues, and service disruptions.

---

## Repository Structure

| Directory       | Contents                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `/src`          | `k8s_failure_prediction.ipynb` -  |
| `/models`       | Trained model files: `scaler.pkl`, `feature_selector.pkl`, `k8s_failure_predictor.pkl`, `anomaly_detector.pkl`. |
| `/data`         | `kubernetes_metrics_dataset.csv` - |
| `/docs`         | `README.md file `                  |
| `/presentation` | `TrailMakers_Kubernetes_ppt.pptx` - Presentation slides.<br>`TrailMakers_Demo_video.mp4`  |

---

## Project Overview

### Objective
Develop a predictive model to identify Kubernetes cluster issues, including node failures, pod crashes, resource exhaustion, network latency, and service disruptions, using a Kaggle-sourced dataset.

### Approach
1. **Data Preprocessing**:
   - Loaded `kubernetes_metrics_dataset.csv` 
   - Filled missing `issue_type` with `No_Issue` (70% prevalence).
   - Sorted chronologically and clipped negative latencies to 0.

2. **Feature Engineering**:
   - Engineered 25 features (e.g., `cpu_usage_percent_rolling`, `network_io_delta`).
   - Selected 15 key features via Random Forest (e.g., `node_ready_status`, `pod_restart_count`).

3. **Modeling**:
   - Trained XGBoost with 5-fold `TimeSeriesSplit`, `scale_pos_weight=22.9` for imbalance.
   - Integrated Isolation Forest (contamination=0.01) for a hybrid model.
   - Defined custom thresholds (e.g., `node_failure`: 0.7) and tested with noise (std=0.05).

4. **Evaluation**:
   - Measured F1, precision, recall, and ROC-AUC on clean and noisy test sets.
   - Explained predictions with top features per sample.

---

## Key Metrics
- **Primary Metrics**: Weighted F1 Score, Precision, Recall, ROC-AUC.
- **Top Features**:
  - `node_ready_status` (Importance: 0.1667)
  - `pod_restart_count` (0.0994)
  - `network_transmit_bytes` (0.0980)

---

## Model Performance
| Stage               | Metric                  | Value  |
|---------------------|-------------------------|--------|
| **Cross-Validation**| Average F1 Score        | 0.9985 |
|                    | Scores                  | [0.9969, 0.9985, 0.9985, 0.9992, 0.9992] |
| **Test Set (Clean)**| XGBoost F1 Score        | 0.999  |
|                    | XGBoost ROC-AUC         | 1.000  |
|                    | Hybrid F1 Score         | 0.99   |
| **Test Set (Noisy)**| Hybrid F1 Score (std=0.05) | 0.97   |
|                    | Minority Class Example  | `scheduler_failure`: 1.00 → 0.60 |

### Observations
- Near-perfect performance on clean data (F1: 0.999), with robust hybrid results (F1: 0.99).
- Noise test showed a drop to 0.97 F1, with minority classes like `scheduler_failure` more affected.
- Thresholds achieved F1 scores of 0.993–1.0, optimized for clean data detection.

---

## Setup Instructions
1. **Prerequisites**:
   - Python 3.8+
   - Install dependencies:
     ```bash
     pip install pandas numpy scikit-learn xgboost matplotlib seaborn


---

## External Links
- **Notebook**: [Kaggle Notebook](https://www.kaggle.com/code/abhishekyelmamdiii/k8s-failure-prediction-ipynb/)
- **Dataset**: [Kubernetes Health Dataset](https://www.kaggle.com/datasets/abhishekyelmamdiii/kubernetes-health-ds1)
- **Models**: All models are within GitHub’s size limit (3.70 MiB total) and stored in `/models`; no external hosting required currently.
 
---

## Additional Details
- **Dataset Source**: The dataset (`kubernetes_metrics_dataset.csv`) was sourced from Kaggle, containing 10,000 samples of Kubernetes cluster metrics spanning February 7 to May 22, 2025. Test data consists of 2000 samples derived from an 80/20 chronological split.
- **Overfitting Insight**: A noise test (Gaussian noise, std=0.05) indicated moderate overfitting, with F1 dropping from 0.999 to 0.96. Minority classes like `scheduler_failure` saw significant performance drops , suggesting sensitivity to real-world variability. Future improvements include validation on raw, noisy Kubernetes data.
- **Threshold Analysis**: Custom thresholds were defined for 9 issue types (e.g., `memory_pressure`: 0.4, `node_failure`: 0.7) and tested on the clean test set, achieving F1 scores ranging from 0.993 to 1.0. This optimized detection but may need adjustment for noisier environments.

---

Team
**TrailMakers:**
`Abhishek Yelmamdi (@ABHISHEK-YELMAMDI)`
`Rutuja Bhagat`
`Bhoomika R P`

---

