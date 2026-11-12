# ============================================================
# STEP 10B — FINAL TEST EVALUATION
#
# Frozen candidate:
#   Logistic Regression
#   Full_enriched feature set (58 features)
#
# Training:
#   All development data only
#
# Final evaluation:
#   Held-out final test set only
#
# IMPORTANT:
#   - Do not tune model settings using final test results.
#   - Test groups must not overlap development groups.
#   - This is an experimental ECG screening model, not a
#     clinical diagnostic system.
# ============================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

STEP8B_DIR = (
    PROJECT_DIR
    / "step8b_five_class_dataset"
)

STEP9B_DIR = (
    PROJECT_DIR
    / "step9b_morphology_features"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "step10_final_test_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Development dataset with 58 model features
DEVELOPMENT_CSV = (
    STEP9B_DIR
    / "five_class_development_morphology_enriched.csv"
)

# All clean windows with morphology features
ALL_ENRICHED_CSV = (
    STEP9B_DIR
    / "five_class_windows_morphology_enriched.csv"
)

# Held-out test split created in Step 8B
FINAL_TEST_SPLIT_CSV = (
    STEP8B_DIR
    / "five_class_final_test.csv"
)


# ============================================================
# 2. TARGET, GROUPING, AND WINDOW KEY COLUMNS
# ============================================================

TARGET_COL = "preliminary_class"
GROUP_COL = "group_id"

WINDOW_KEY = [
    "record_id",
    "channel",
    "window_index",
]


# ============================================================
# 3. CLASS ORDER
# ============================================================

CLASS_ORDER = [
    "Normal heartbeat",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal ECG",
]


# ============================================================
# 4. EXACT 58 FEATURES FROM STEP 9D
# ============================================================

BASELINE_FEATURES = [
    "total_beats",
    "valid_rate_count",
    "valid_rate_fraction",
    "median_hr_bpm",
    "mean_hr_bpm",
    "min_hr_bpm",
    "max_hr_bpm",
    "median_rr_ms",
    "mean_rr_ms",
]


MORPHOLOGY_FEATURES = [
    "RR_interval_ms_median",
    "RR_interval_ms_mean",
    "RR_interval_ms_std",
    "RR_interval_ms_min",
    "RR_interval_ms_max",

    "Heart_rate_bpm_median",
    "Heart_rate_bpm_mean",
    "Heart_rate_bpm_std",
    "Heart_rate_bpm_min",
    "Heart_rate_bpm_max",

    "R_amplitude_mV_median",
    "R_amplitude_mV_mean",
    "R_amplitude_mV_std",

    "P_amplitude_mV_median",
    "P_amplitude_mV_mean",
    "P_amplitude_mV_std",

    "P_duration_ms_median",
    "P_duration_ms_mean",

    "QRS_duration_ms_median",
    "QRS_duration_ms_mean",

    "PR_interval_ms_median",
    "PR_interval_ms_mean",

    "QT_interval_ms_median",
    "QT_interval_ms_mean",

    "T_amplitude_mV_median",
    "T_amplitude_mV_mean",

    "T_duration_ms_median",
    "T_duration_ms_mean",
]


AVAILABILITY_FEATURES = [
    "RR_interval_ms_valid_count",
    "RR_interval_ms_valid_fraction",

    "Heart_rate_bpm_valid_count",
    "Heart_rate_bpm_valid_fraction",

    "R_amplitude_mV_valid_count",
    "R_amplitude_mV_valid_fraction",

    "P_amplitude_mV_valid_count",
    "P_amplitude_mV_valid_fraction",

    "P_duration_ms_valid_count",
    "P_duration_ms_valid_fraction",

    "QRS_duration_ms_valid_count",
    "QRS_duration_ms_valid_fraction",

    "PR_interval_ms_valid_count",
    "PR_interval_ms_valid_fraction",

    "QT_interval_ms_valid_count",
    "QT_interval_ms_valid_fraction",

    "T_amplitude_mV_valid_count",
    "T_amplitude_mV_valid_fraction",

    "T_duration_ms_valid_count",
    "T_duration_ms_valid_fraction",

    "aggregated_beat_count",
]


FEATURES = (
    BASELINE_FEATURES
    + MORPHOLOGY_FEATURES
    + AVAILABILITY_FEATURES
)


# ============================================================
# 5. EXPECTED DATASET SIZES
# ============================================================

EXPECTED_DEVELOPMENT_ROWS = 3544
EXPECTED_TEST_ROWS = 812

EXPECTED_DEVELOPMENT_GROUPS = 38
EXPECTED_TEST_GROUPS = 9


# ============================================================
# 6. DISPLAY HEADER
# ============================================================

print("\n" + "=" * 75)
print("STEP 10B — FINAL TEST EVALUATION")
print("=" * 75)

print("\nFrozen model:")
print("  Logistic Regression")
print("  Full_enriched feature set")
print(f"  Number of features: {len(FEATURES)}")

print("\nEvaluation policy:")
print("  Fit on development data only")
print("  Evaluate on final test data only")
print("  No model tuning using final test results")


# ============================================================
# 7. CHECK INPUT FILES
# ============================================================

required_files = [
    DEVELOPMENT_CSV,
    ALL_ENRICHED_CSV,
    FINAL_TEST_SPLIT_CSV,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required input files found.")


# ============================================================
# 8. LOAD DEVELOPMENT AND SPLIT INFORMATION
# ============================================================

print("\nLoading development dataset...")

development_df = pd.read_csv(
    DEVELOPMENT_CSV
)

print(
    "Development dataset shape:",
    development_df.shape,
)


print("\nLoading Step 8B final test split...")

test_split_df = pd.read_csv(
    FINAL_TEST_SPLIT_CSV
)

print(
    "Final test split shape:",
    test_split_df.shape,
)


# ============================================================
# 9. VALIDATE REQUIRED COLUMNS
# ============================================================

required_development_columns = (
    WINDOW_KEY
    + [
        TARGET_COL,
        GROUP_COL,
    ]
    + FEATURES
)

missing_development_columns = [
    col
    for col in required_development_columns
    if col not in development_df.columns
]

if missing_development_columns:

    raise ValueError(
        "\nDevelopment dataset is missing columns:\n"
        + "\n".join(missing_development_columns)
    )


required_test_split_columns = (
    WINDOW_KEY
    + [
        TARGET_COL,
        GROUP_COL,
    ]
)

missing_test_split_columns = [
    col
    for col in required_test_split_columns
    if col not in test_split_df.columns
]

if missing_test_split_columns:

    raise ValueError(
        "\nFinal test split is missing columns:\n"
        + "\n".join(missing_test_split_columns)
    )


# ============================================================
# 10. VALIDATE DEVELOPMENT DATA
# ============================================================

development_df = development_df.dropna(
    subset=[
        TARGET_COL,
        GROUP_COL,
    ]
).copy()

development_df[TARGET_COL] = (
    development_df[TARGET_COL].astype(str)
)

development_df[GROUP_COL] = (
    development_df[GROUP_COL].astype(str)
)

print("\nDevelopment rows:", len(development_df))
print(
    "Development groups:",
    development_df[GROUP_COL].nunique(),
)

print("\nDevelopment class counts:")
print(
    development_df[TARGET_COL]
    .value_counts()
    .reindex(CLASS_ORDER, fill_value=0)
)


# ============================================================
# 11. IDENTIFY HELD-OUT TEST GROUPS
# ============================================================

test_split_df = test_split_df.dropna(
    subset=[
        TARGET_COL,
        GROUP_COL,
    ]
).copy()

test_split_df[TARGET_COL] = (
    test_split_df[TARGET_COL].astype(str)
)

test_split_df[GROUP_COL] = (
    test_split_df[GROUP_COL].astype(str)
)

test_group_ids = sorted(
    test_split_df[GROUP_COL].unique()
)

print("\nHeld-out test group IDs:")
for group_id in test_group_ids:
    print(" ", group_id)

print(
    "\nNumber of held-out test groups:",
    len(test_group_ids),
)


# ============================================================
# 12. LOAD ALL ENRICHED WINDOWS
#
# This file contains all 4,356 clean windows.
# We use test group IDs from the Step 8B split to identify
# the 812 held-out test windows.
# ============================================================

print("\nLoading all enriched clean windows...")

all_enriched_df = pd.read_csv(
    ALL_ENRICHED_CSV
)

print(
    "All enriched windows shape:",
    all_enriched_df.shape,
)


required_all_columns = (
    WINDOW_KEY
    + [
        TARGET_COL,
        GROUP_COL,
    ]
    + FEATURES
)

missing_all_columns = [
    col
    for col in required_all_columns
    if col not in all_enriched_df.columns
]

if missing_all_columns:

    raise ValueError(
        "\nAll-enriched dataset is missing columns:\n"
        + "\n".join(missing_all_columns)
    )


all_enriched_df[TARGET_COL] = (
    all_enriched_df[TARGET_COL].astype(str)
)

all_enriched_df[GROUP_COL] = (
    all_enriched_df[GROUP_COL].astype(str)
)


# ============================================================
# 13. SELECT FINAL TEST WINDOWS BY GROUP ID
# ============================================================

test_df = all_enriched_df[
    all_enriched_df[GROUP_COL].isin(test_group_ids)
].copy()

print("\nSelected final test dataset shape:", test_df.shape)

print(
    "Selected final test groups:",
    test_df[GROUP_COL].nunique(),
)


# ============================================================
# 14. VALIDATE TEST SPLIT SIZE AND GROUPS
# ============================================================

if len(development_df) != EXPECTED_DEVELOPMENT_ROWS:

    raise ValueError(
        "Unexpected development row count. "
        f"Expected {EXPECTED_DEVELOPMENT_ROWS}, "
        f"found {len(development_df)}."
    )


if len(test_df) != EXPECTED_TEST_ROWS:

    raise ValueError(
        "Unexpected final test row count. "
        f"Expected {EXPECTED_TEST_ROWS}, "
        f"found {len(test_df)}."
    )


if (
    development_df[GROUP_COL].nunique()
    != EXPECTED_DEVELOPMENT_GROUPS
):

    raise ValueError(
        "Unexpected development group count. "
        f"Expected {EXPECTED_DEVELOPMENT_GROUPS}, "
        f"found {development_df[GROUP_COL].nunique()}."
    )


if (
    test_df[GROUP_COL].nunique()
    != EXPECTED_TEST_GROUPS
):

    raise ValueError(
        "Unexpected final test group count. "
        f"Expected {EXPECTED_TEST_GROUPS}, "
        f"found {test_df[GROUP_COL].nunique()}."
    )


# ============================================================
# 15. GROUP LEAKAGE CHECK
# ============================================================

development_groups = set(
    development_df[GROUP_COL].unique()
)

final_test_groups = set(
    test_df[GROUP_COL].unique()
)

overlap_groups = (
    development_groups
    .intersection(final_test_groups)
)

if overlap_groups:

    raise ValueError(
        "\nGROUP LEAKAGE DETECTED!\n"
        f"Overlapping groups: {overlap_groups}"
    )

print("\nGroup leakage check: PASSED")
print("Development and final test groups are separate.")


# ============================================================
# 16. VALIDATE TEST LABELS AGAINST STEP 8B SPLIT
# ============================================================

test_key_labels = test_split_df[
    WINDOW_KEY + [TARGET_COL, GROUP_COL]
].copy()

enriched_test_key_labels = test_df[
    WINDOW_KEY + [TARGET_COL, GROUP_COL]
].copy()


test_key_labels = test_key_labels.sort_values(
    WINDOW_KEY
).reset_index(drop=True)

enriched_test_key_labels = (
    enriched_test_key_labels
    .sort_values(WINDOW_KEY)
    .reset_index(drop=True)
)


if len(test_key_labels) != len(enriched_test_key_labels):

    raise ValueError(
        "Test split row count does not match enriched test rows."
    )


if not test_key_labels[WINDOW_KEY].equals(
    enriched_test_key_labels[WINDOW_KEY]
):

    raise ValueError(
        "Test window keys do not match between Step 8B "
        "and Step 9B enriched data."
    )


if not test_key_labels[TARGET_COL].equals(
    enriched_test_key_labels[TARGET_COL]
):

    raise ValueError(
        "Test labels do not match between Step 8B "
        "and Step 9B enriched data."
    )


print("Test window and label consistency check: PASSED")


# ============================================================
# 17. VALIDATE FEATURE SET
# ============================================================

if len(FEATURES) != 58:

    raise ValueError(
        f"Expected 58 features, found {len(FEATURES)}."
    )


print("\nFeature validation: PASSED")
print(f"Using exactly {len(FEATURES)} frozen features.")


# ============================================================
# 18. PREPARE X AND y
# ============================================================

X_train = development_df[FEATURES].copy()

y_train = development_df[TARGET_COL].copy()

X_test = test_df[FEATURES].copy()

y_test = test_df[TARGET_COL].copy()


# ============================================================
# 19. DEFINE EXACT FROZEN MODEL PIPELINE
#
# Matches Step 9D:
#   SimpleImputer(strategy="median")
#   StandardScaler()
#   LogisticRegression(
#       max_iter=3000,
#       class_weight="balanced",
#       random_state=42
#   )
# ============================================================

frozen_model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        ),
    ),

    (
        "scaler",
        StandardScaler(),
    ),

    (
        "model",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


# ============================================================
# 20. FIT ON DEVELOPMENT DATA ONLY
# ============================================================

print("\n" + "=" * 75)
print("FITTING FROZEN MODEL")
print("=" * 75)

print("Training rows:", len(X_train))
print("Training groups:", development_df[GROUP_COL].nunique())
print("Final test rows are NOT used during fitting.")

frozen_model.fit(
    X_train,
    y_train,
)

print("\nModel fitting completed.")


# ============================================================
# 21. SAVE FITTED MODEL
#
# Save this fitted pipeline for later dashboard packaging.
# ============================================================

model_path = (
    OUTPUT_DIR
    / "step10_frozen_full_enriched_logistic_model.joblib"
)

joblib.dump(
    frozen_model,
    model_path,
)

print("\nFitted model saved:")
print(model_path)


# ============================================================
# 22. FINAL TEST PREDICTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

print(
    "Evaluating on",
    len(X_test),
    "held-out test windows..."
)

y_pred = frozen_model.predict(
    X_test
)


# ============================================================
# 23. OVERALL TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_pred,
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

test_macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

test_weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

test_macro_precision = precision_score(
    y_test,
    y_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0,
)

test_macro_recall = recall_score(
    y_test,
    y_pred,
    labels=CLASS_ORDER,
    average="macro",
    zero_division=0,
)


metrics = {
    "model": "Logistic Regression",
    "feature_set": "Full_enriched",
    "n_features": len(FEATURES),

    "development_rows": len(X_train),
    "development_groups": development_df[GROUP_COL].nunique(),

    "final_test_rows": len(X_test),
    "final_test_groups": test_df[GROUP_COL].nunique(),

    "accuracy": test_accuracy,
    "balanced_accuracy": test_balanced_accuracy,
    "macro_precision": test_macro_precision,
    "macro_recall": test_macro_recall,
    "macro_f1": test_macro_f1,
    "weighted_f1": test_weighted_f1,
}


metrics_df = pd.DataFrame(
    [metrics]
)


# ============================================================
# 24. PER-CLASS CLASSIFICATION REPORT
# ============================================================

class_report_dict = classification_report(
    y_test,
    y_pred,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0,
)

class_report_df = pd.DataFrame(
    class_report_dict
).transpose()


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASS_ORDER,
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{name}"
        for name in CLASS_ORDER
    ],
    columns=[
        f"Predicted_{name}"
        for name in CLASS_ORDER
    ],
)


# ============================================================
# 26. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df[
    WINDOW_KEY
    + [
        GROUP_COL,
        TARGET_COL,
    ]
].copy()

prediction_df["predicted_class"] = y_pred

prediction_df["correct"] = (
    prediction_df[TARGET_COL]
    == prediction_df["predicted_class"]
)

prediction_df["confidence"] = (
    frozen_model.predict_proba(X_test).max(axis=1)
)


# ============================================================
# 27. SAVE OUTPUT FILES
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "step10_final_test_metrics.csv"
)

class_report_path = (
    OUTPUT_DIR
    / "step10_final_test_classification_report.csv"
)

confusion_matrix_path = (
    OUTPUT_DIR
    / "step10_final_test_confusion_matrix.csv"
)

predictions_path = (
    OUTPUT_DIR
    / "step10_final_test_predictions.csv"
)

feature_list_path = (
    OUTPUT_DIR
    / "step10_frozen_feature_list.txt"
)

run_report_path = (
    OUTPUT_DIR
    / "step10_final_test_run_report.txt"
)


metrics_df.to_csv(
    metrics_path,
    index=False,
)

class_report_df.to_csv(
    class_report_path,
)

cm_df.to_csv(
    confusion_matrix_path,
)

prediction_df.to_csv(
    predictions_path,
    index=False,
)


with open(
    feature_list_path,
    "w",
    encoding="utf-8",
) as f:

    f.write("STEP 10 — FROZEN FULL ENRICHED FEATURE LIST\n")
    f.write("=" * 60 + "\n\n")

    for feature in FEATURES:
        f.write(feature + "\n")


# ============================================================
# 28. SAVE RUN REPORT
# ============================================================

with open(
    run_report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "STEP 10B — FINAL TEST EVALUATION\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write("Frozen model: Logistic Regression\n")
    f.write("Frozen feature set: Full_enriched\n")
    f.write(f"Number of features: {len(FEATURES)}\n\n")

    f.write("Pipeline:\n")
    f.write("  SimpleImputer(strategy='median')\n")
    f.write("  StandardScaler()\n")
    f.write("  LogisticRegression(\n")
    f.write("      max_iter=3000,\n")
    f.write("      class_weight='balanced',\n")
    f.write("      random_state=42,\n")
    f.write("  )\n\n")

    f.write(f"Development rows: {len(X_train)}\n")
    f.write(
        "Development groups: "
        f"{development_df[GROUP_COL].nunique()}\n"
    )

    f.write(f"Final test rows: {len(X_test)}\n")
    f.write(
        "Final test groups: "
        f"{test_df[GROUP_COL].nunique()}\n"
    )

    f.write("Group leakage check: PASSED\n")
    f.write("Test split consistency check: PASSED\n")
    f.write("Final test used for model fitting: NO\n\n")

    f.write("Final test metrics:\n")
    f.write(metrics_df.to_string(index=False))

    f.write("\n\nPer-class classification report:\n")
    f.write(class_report_df.to_string())

    f.write("\n\nConfusion matrix:\n")
    f.write(cm_df.to_string())


# ============================================================
# 29. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 10B COMPLETED")
print("=" * 75)

print("\nFINAL TEST METRICS:")
print(metrics_df.to_string(index=False))

print("\nPER-CLASS CLASSIFICATION REPORT:")
print(class_report_df.to_string())

print("\nCONFUSION MATRIX:")
print(cm_df.to_string())

print("\nSaved outputs:")
print(metrics_path)
print(class_report_path)
print(confusion_matrix_path)
print(predictions_path)
print(feature_list_path)
print(run_report_path)
print(model_path)

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION FINISHED")
print("=" * 75)

print(
    "\nReminder: These are experimental ECG screening results. "
    "They are not clinical diagnostic performance claims."
)