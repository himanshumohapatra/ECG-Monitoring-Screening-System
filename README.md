# ECG Monitoring and Predictive System

A software-based ECG monitoring and predictive system developed using the **MIT-BIH Arrhythmia Database** and machine learning techniques.

The project analyses recorded ECG data, extracts physiological and morphological features, and classifies ECG windows into five categories using a trained machine learning model.

### Key Features

- **Dataset:** MIT-BIH Arrhythmia Database.
- **Records:** 48 ECG records.
- **Beat-level dataset:** 218,988 rows.
- **Segmentation:** 20-second ECG windows.
- **Five-class classification:**
  - Normal heartbeat
  - Arrhythmia
  - Tachycardia
  - Bradycardia
  - Other abnormal ECG
- **Feature engineering:** 58 selected physiological and morphological features.
- **Machine learning:** Logistic Regression, Random Forest, SVM, XGBoost and MLP comparison.
- **Model selection:** Development-based grouped cross-validation.
- **Final model:** Logistic Regression with 58 enriched features.
- **Evaluation:** Independent grouped test dataset.
- **Interface:** Interactive Streamlit dashboard.
- **Simulation:** Recorded MIT-BIH ECG signal replay.

### Final Model Performance

| Metric | Test result |
|---|---:|
| Accuracy | 69.09% |
| Balanced Accuracy | 66.04% |
| Macro F1-score | 66.20% |
| Weighted F1-score | 69.33% |

### Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Joblib

### Project Purpose

The objective is to develop a software-based ECG analysis and predictive screening system that demonstrates ECG signal visualization, machine learning-based classification, and an interactive monitoring interface.

**Note:** This is an academic software simulation and experimental ECG screening project. It is not a clinically validated diagnostic system.
