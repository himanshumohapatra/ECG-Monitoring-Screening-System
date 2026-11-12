# ============================================================
# STEP 9 — FIVE-CLASS MODEL COMPARISON
# ============================================================
#
# Purpose:
#   Compare five machine-learning models for the new
#   five-class ECG window classification task.
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. SVM
#   4. XGBoost
#   5. MLP
#
# Dataset:
#   five_class_development_with_cv_folds.csv
#
# IMPORTANT:
#   - Uses the precomputed 5 grouped CV folds from Step 8B.
#   - Does NOT use five_class_final_test.csv.
#   - Does NOT modify the old beat-classification datasets.
#   - Does NOT use variables that directly created the target label.
#   - Does NOT use record/channel/group/fold identifiers as features.
#
# Outputs:
#   - Overall model comparison CSV
#   - Fold-level results CSV
#   - Per-class recall CSV
#   - Out-of-fold predictions CSV
#   - Confusion matrices
#   - Classification reports
#   - Model timing information
#   - Text report
#
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "step8b_five_class_dataset"
)

INPUT_FILE = os.path.join(
    INPUT_DIR,
    "five_class_development_with_cv_folds.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "step9_five_class_model_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. TARGET CLASSES
# ============================================================

TARGET_COLUMN = "preliminary_class"

CLASS_ORDER = [
    "Normal heartbeat",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal ECG"
]


# ============================================================
# 3. FEATURES
# ============================================================
#
# These are the window-level variables that are NOT directly
# used to define preliminary_class.
#
# We intentionally DO NOT use:
#
#   normal_fraction
#   arrhythmia_fraction
#   tachy_fraction
#   brady_fraction
#   other_abnormal_fraction
#   dominant_label
#   dominant_label_fraction
#
# because these variables directly contributed to the Step 8A
# rule-based target.
#
# We also exclude identifiers and split metadata:
#
#   record_id
#   channel
#   group_id
#   cv_fold
#   window_index
#   window_start_s
#   window_end_s
#   preliminary_class
#
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 9 — FIVE-CLASS MODEL COMPARISON")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file was not found:\n{INPUT_FILE}\n"
        "\nPlease check that Step 8B completed successfully."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

required_columns = (
    FEATURE_COLUMNS
    + [
        TARGET_COLUMN,
        "cv_fold",
        "group_id",
        "record_id",
        "channel",
    ]
)

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "\nThe following required columns are missing:\n"
        + "\n".join(missing_required)
    )

print("\nAll required columns are present.")


# ============================================================
# 6. CHECK TARGET
# ============================================================

print("\nTarget distribution:")

target_counts = df[TARGET_COLUMN].value_counts()

for class_name in CLASS_ORDER:
    count = target_counts.get(class_name, 0)
    percentage = count / len(df) * 100

    print(
        f"  {class_name:<25} "
        f"{count:>6,} "
        f"({percentage:6.2f}%)"
    )

unexpected_classes = sorted(
    set(df[TARGET_COLUMN].dropna().unique())
    - set(CLASS_ORDER)
)

if unexpected_classes:
    raise ValueError(
        "\nUnexpected target classes found:\n"
        + "\n".join(map(str, unexpected_classes))
    )

missing_classes = [
    c for c in CLASS_ORDER
    if c not in df[TARGET_COLUMN].unique()
]

if missing_classes:
    raise ValueError(
        "\nExpected target classes are missing:\n"
        + "\n".join(missing_classes)
    )


# ============================================================
# 7. CHECK CV FOLDS
# ============================================================

print("\n" + "=" * 70)
print("CV FOLD VALIDATION")
print("=" * 70)

fold_values = sorted(df["cv_fold"].dropna().unique())

print("\nCV folds found:")
print(fold_values)

if len(fold_values) != 5:
    raise ValueError(
        f"\nExpected exactly 5 CV folds, found {len(fold_values)}."
    )

if set(fold_values) != {1, 2, 3, 4, 5}:
    raise ValueError(
        "\nExpected CV folds to be numbered 1, 2, 3, 4, 5."
    )


# ============================================================
# 8. GROUP LEAKAGE CHECK
# ============================================================
#
# Every group must belong to exactly one validation fold.
#
# This is critical because multiple windows from the same
# record/group must never appear in both training and validation.
# ============================================================

group_fold_counts = (
    df.groupby("group_id")["cv_fold"]
    .nunique()
)

leaking_groups = group_fold_counts[
    group_fold_counts > 1
]

if len(leaking_groups) > 0:
    print("\nERROR: Group leakage detected!")

    print(leaking_groups)

    raise ValueError(
        "\nAt least one group appears in multiple CV folds."
    )

print("\nGroup leakage check: PASSED")

print(
    f"Unique groups: "
    f"{df['group_id'].nunique():,}"
)


# ============================================================
# 9. CHECK 201 + 202 GROUPING
# ============================================================

records_201_202 = df[
    df["record_id"].astype(str).isin(["201", "202"])
]

if len(records_201_202) > 0:

    groups_201_202 = (
        records_201_202["group_id"]
        .unique()
    )

    folds_201_202 = (
        records_201_202["cv_fold"]
        .unique()
    )

    print("\n201 + 202 grouping check:")

    print(
        f"Groups: {groups_201_202}"
    )

    print(
        f"CV folds: {folds_201_202}"
    )

    if len(groups_201_202) != 1:
        raise ValueError(
            "\nERROR: Records 201 and 202 are not in "
            "the same group."
        )

    if len(folds_201_202) != 1:
        raise ValueError(
            "\nERROR: Records 201 and 202 are not in "
            "the same CV fold."
        )

    print("201 + 202 grouping check: PASSED")


# ============================================================
# 10. FINAL TEST SET PROTECTION
# ============================================================
#
# This script deliberately never loads the final test file.
#
# We only verify that the development dataset contains no
# obvious final-test-only metadata.
# ============================================================

FINAL_TEST_FILE = os.path.join(
    INPUT_DIR,
    "five_class_final_test.csv"
)

print("\nFinal test protection:")
print(
    "five_class_final_test.csv will NOT be loaded or used."
)

if os.path.exists(FINAL_TEST_FILE):
    print(
        "Final test file detected on disk: YES"
    )
    print(
        "Final test file used: NO"
    )
else:
    print(
        "Final test file detected on disk: NO "
        "(this does not affect Step 9)"
    )


# ============================================================
# 11. PREPARE X AND Y
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y_text = df[TARGET_COLUMN].copy()

label_encoder = LabelEncoder()

label_encoder.fit(CLASS_ORDER)

y = label_encoder.transform(y_text)

encoded_class_order = label_encoder.transform(CLASS_ORDER)

print("\nEncoded class mapping:")

for class_name, encoded_value in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(
        f"  {encoded_value}: {class_name}"
    )


# ============================================================
# 12. FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

feature_summary = []

for feature in FEATURE_COLUMNS:

    missing_count = X[feature].isna().sum()
    missing_percentage = (
        missing_count / len(X) * 100
    )

    feature_summary.append({
        "feature": feature,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "min": X[feature].min(),
        "median": X[feature].median(),
        "mean": X[feature].mean(),
        "max": X[feature].max(),
    })

feature_summary_df = pd.DataFrame(feature_summary)

print(
    feature_summary_df.to_string(
        index=False
    )
)

feature_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_summary.csv"
    ),
    index=False
)


# ============================================================
# 13. MODEL DEFINITIONS
# ============================================================
#
# IMPORTANT:
#
# Scaling is required for:
#   - Logistic Regression
#   - SVM
#   - MLP
#
# Random Forest and XGBoost do not require feature scaling.
#
# Imputation is included inside every pipeline so that
# statistics are learned ONLY from the training portion of
# each fold.
#
# This prevents preprocessing leakage.
# ============================================================

models = {}


# ------------------------------------------------------------
# 13.1 Logistic Regression
# ------------------------------------------------------------

models["Logistic Regression"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 13.2 Random Forest
# ------------------------------------------------------------

models["Random Forest"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 13.3 SVM
# ------------------------------------------------------------

models["SVM"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# 13.4 XGBoost
# ------------------------------------------------------------
#
# We use sample weights for class imbalance because
# XGBClassifier does not use class_weight in the same way
# as Logistic Regression / RF / SVM.
#
# Sample weights are generated separately for each training
# fold.
# ------------------------------------------------------------

models["XGBoost"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=len(CLASS_ORDER),
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 13.5 MLP
# ------------------------------------------------------------

models["MLP"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=64,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42
            )
        )
    ]
)


print("\nModels prepared:")

for model_name in models:
    print(f"  - {model_name}")


# ============================================================
# 14. CLASS WEIGHTS FOR XGBOOST
# ============================================================

class_counts = (
    y_text.value_counts()
)

n_total = len(y_text)
n_classes = len(CLASS_ORDER)

class_weight_map = {}

for class_name in CLASS_ORDER:

    count = class_counts[class_name]

    class_weight_map[class_name] = (
        n_total /
        (n_classes * count)
    )

print("\nClass weights used for XGBoost:")

for class_name in CLASS_ORDER:
    print(
        f"  {class_name:<25} "
        f"{class_weight_map[class_name]:.4f}"
    )


# ============================================================
# 15. STORAGE FOR RESULTS
# ============================================================

overall_results = []

fold_results = []

per_class_results = []

all_oof_predictions = []

confusion_matrices = {}

classification_reports = {}


# ============================================================
# 16. RUN 5-FOLD GROUPED CV
# ============================================================

print("\n" + "=" * 70)
print("STARTING 5-FOLD GROUPED CROSS-VALIDATION")
print("=" * 70)

print(
    "\nIMPORTANT: "
    "The precomputed cv_fold column from Step 8B is used."
)

for model_name, model in models.items():

    print("\n")
    print("#" * 70)
    print(f"MODEL: {model_name}")
    print("#" * 70)

    model_start_time = time.perf_counter()

    model_oof_true = []
    model_oof_pred = []

    model_fold_training_times = []
    model_fold_prediction_times = []

    # --------------------------------------------------------
    # Five predefined folds
    # --------------------------------------------------------

    for fold in [1, 2, 3, 4, 5]:

        print("\n" + "-" * 70)
        print(f"FOLD {fold}")
        print("-" * 70)

        train_mask = df["cv_fold"] != fold
        val_mask = df["cv_fold"] == fold

        X_train = X.loc[train_mask]
        X_val = X.loc[val_mask]

        y_train = y[train_mask]
        y_val = y[val_mask]

        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]

        print(
            f"Training rows   : {len(X_train):,}"
        )

        print(
            f"Validation rows : {len(X_val):,}"
        )

        print(
            f"Training groups : "
            f"{train_df['group_id'].nunique():,}"
        )

        print(
            f"Validation groups : "
            f"{val_df['group_id'].nunique():,}"
        )

        # ----------------------------------------------------
        # Explicit group leakage check
        # ----------------------------------------------------

        train_groups = set(
            train_df["group_id"].unique()
        )

        val_groups = set(
            val_df["group_id"].unique()
        )

        overlap = train_groups.intersection(
            val_groups
        )

        if overlap:

            raise ValueError(
                f"\nGROUP LEAKAGE DETECTED in fold {fold}:\n"
                f"{sorted(overlap)}"
            )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_start = time.perf_counter()

        if model_name == "XGBoost":

            sample_weights = np.array([
                class_weight_map[
                    label_encoder.inverse_transform(
                        [label]
                    )[0]
                ]
                for label in y_train
            ])

            model.fit(
                X_train,
                y_train,
                model__sample_weight=sample_weights
            )

        else:

            model.fit(
                X_train,
                y_train
            )

        train_time = (
            time.perf_counter()
            - train_start
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_start = time.perf_counter()

        y_pred = model.predict(
            X_val
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val,
            y_pred
        )

        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recalls = recall_score(
            y_val,
            y_pred,
            labels=encoded_class_order,
            average=None,
            zero_division=0
        )

        print(
            f"\nAccuracy           : {accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy : {balanced_accuracy:.4f}"
        )

        print(
            f"Macro F1          : {macro_f1:.4f}"
        )

        print(
            f"Weighted F1       : {weighted_f1:.4f}"
        )

        print(
            f"Training time     : {train_time:.3f} s"
        )

        print(
            f"Prediction time   : {prediction_time:.3f} s"
        )

        # ----------------------------------------------------
        # Store fold metrics
        # ----------------------------------------------------

        fold_results.append({
            "model": model_name,
            "fold": fold,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "train_groups": train_df["group_id"].nunique(),
            "validation_groups": val_df["group_id"].nunique(),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "training_time_s": train_time,
            "prediction_time_s": prediction_time,
        })

        # ----------------------------------------------------
        # Per-class sensitivity
        # ----------------------------------------------------

        for class_name, recall_value in zip(
            CLASS_ORDER,
            recalls
        ):

            per_class_results.append({
                "model": model_name,
                "fold": fold,
                "class": class_name,
                "recall_sensitivity": recall_value
            })

        # ----------------------------------------------------
        # OOF predictions
        # ----------------------------------------------------

        model_oof_true.extend(
            y_val.tolist()
        )

        model_oof_pred.extend(
            y_pred.tolist()
        )

        # Save detailed OOF rows

        fold_prediction_df = val_df[
            [
                "record_id",
                "channel",
                "window_index",
                "group_id",
                "cv_fold",
            ]
        ].copy()

        fold_prediction_df["model"] = model_name

        fold_prediction_df["true_class"] = (
            label_encoder.inverse_transform(
                y_val
            )
        )

        fold_prediction_df["predicted_class"] = (
            label_encoder.inverse_transform(
                y_pred
            )
        )

        fold_prediction_df["correct"] = (
            y_val == y_pred
        )

        all_oof_predictions.append(
            fold_prediction_df
        )

        model_fold_training_times.append(
            train_time
        )

        model_fold_prediction_times.append(
            prediction_time
        )

    # ========================================================
    # OVERALL OOF RESULTS FOR THIS MODEL
    # ========================================================

    model_total_time = (
        time.perf_counter()
        - model_start_time
    )

    model_oof_true = np.array(
        model_oof_true
    )

    model_oof_pred = np.array(
        model_oof_pred
    )

    overall_accuracy = accuracy_score(
        model_oof_true,
        model_oof_pred
    )

    overall_balanced_accuracy = (
        balanced_accuracy_score(
            model_oof_true,
            model_oof_pred
        )
    )

    overall_macro_f1 = f1_score(
        model_oof_true,
        model_oof_pred,
        average="macro",
        zero_division=0
    )

    overall_weighted_f1 = f1_score(
        model_oof_true,
        model_oof_pred,
        average="weighted",
        zero_division=0
    )

    overall_recalls = recall_score(
        model_oof_true,
        model_oof_pred,
        labels=encoded_class_order,
        average=None,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(f"OVERALL OOF RESULT — {model_name}")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{overall_balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{overall_macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{overall_weighted_f1:.4f}"
    )

    print(
        f"Total training time: "
        f"{sum(model_fold_training_times):.3f} s"
    )

    print(
        f"Total prediction time: "
        f"{sum(model_fold_prediction_times):.3f} s"
    )

    print(
        f"Total model time: "
        f"{model_total_time:.3f} s"
    )

    print("\nPer-class sensitivity:")

    for class_name, recall_value in zip(
        CLASS_ORDER,
        overall_recalls
    ):

        print(
            f"  {class_name:<25} "
            f"{recall_value:.4f}"
        )

    # --------------------------------------------------------
    # Overall result row
    # --------------------------------------------------------

    result_row = {
        "model": model_name,
        "accuracy": overall_accuracy,
        "balanced_accuracy": overall_balanced_accuracy,
        "macro_f1": overall_macro_f1,
        "weighted_f1": overall_weighted_f1,
        "mean_fold_training_time_s": np.mean(
            model_fold_training_times
        ),
        "total_training_time_s": np.sum(
            model_fold_training_times
        ),
        "mean_fold_prediction_time_s": np.mean(
            model_fold_prediction_times
        ),
        "total_prediction_time_s": np.sum(
            model_fold_prediction_times
        ),
        "total_model_time_s": model_total_time,
    }

    # Add per-class sensitivity to result row

    for class_name, recall_value in zip(
        CLASS_ORDER,
        overall_recalls
    ):

        safe_name = (
            class_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result_row[
            f"recall_{safe_name}"
        ] = recall_value

    overall_results.append(
        result_row
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        model_oof_true,
        model_oof_pred,
        labels=encoded_class_order
    )

    confusion_matrices[
        model_name
    ] = cm

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        model_oof_true,
        model_oof_pred,
        labels=encoded_class_order,
        target_names=CLASS_ORDER,
        zero_division=0
    )

    classification_reports[
        model_name
    ] = report

    print("\nClassification report:")
    print(report)


# ============================================================
# 17. SAVE RESULT TABLES
# ============================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# Overall comparison
# ------------------------------------------------------------

overall_results_df = pd.DataFrame(
    overall_results
)

overall_results_df = (
    overall_results_df
    .sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "accuracy"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

overall_results_file = os.path.join(
    OUTPUT_DIR,
    "step9_overall_model_comparison.csv"
)

overall_results_df.to_csv(
    overall_results_file,
    index=False
)


# ------------------------------------------------------------
# Fold-level results
# ------------------------------------------------------------

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_file = os.path.join(
    OUTPUT_DIR,
    "step9_fold_results.csv"
)

fold_results_df.to_csv(
    fold_results_file,
    index=False
)


# ------------------------------------------------------------
# Per-class recall
# ------------------------------------------------------------

per_class_results_df = pd.DataFrame(
    per_class_results
)

per_class_results_file = os.path.join(
    OUTPUT_DIR,
    "step9_per_class_sensitivity.csv"
)

per_class_results_df.to_csv(
    per_class_results_file,
    index=False
)


# ------------------------------------------------------------
# OOF predictions
# ------------------------------------------------------------

oof_predictions_df = pd.concat(
    all_oof_predictions,
    ignore_index=True
)

oof_predictions_file = os.path.join(
    OUTPUT_DIR,
    "step9_out_of_fold_predictions.csv"
)

oof_predictions_df.to_csv(
    oof_predictions_file,
    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRICES
# ============================================================

print("\nSaving confusion matrices...")

for model_name, cm in confusion_matrices.items():

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )

    cm_csv_file = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{safe_model_name}.csv"
    )

    cm_df.to_csv(
        cm_csv_file
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Confusion Matrix — {model_name}"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(CLASS_ORDER)
    )

    plt.xticks(
        tick_positions,
        CLASS_ORDER,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_positions,
        CLASS_ORDER
    )

    threshold = (
        cm.max() / 2.0
        if cm.max() > 0
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white"
                if cm[i, j] > threshold
                else "black"
            )

    plt.ylabel(
        "True class"
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.tight_layout()

    cm_png_file = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{safe_model_name}.png"
    )

    plt.savefig(
        cm_png_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 19. SAVE CLASSIFICATION REPORTS
# ============================================================

reports_file = os.path.join(
    OUTPUT_DIR,
    "step9_classification_reports.txt"
)

with open(
    reports_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 9 — FIVE-CLASS MODEL COMPARISON\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for model_name, report in (
        classification_reports.items()
    ):

        f.write(
            f"MODEL: {model_name}\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            report
        )

        f.write(
            "\n\n"
        )


# ============================================================
# 20. CREATE MODEL RANKING
# ============================================================

ranking_df = overall_results_df.copy()

ranking_df.insert(
    0,
    "rank_by_macro_f1",
    np.arange(
        1,
        len(ranking_df) + 1
    )
)

ranking_file = os.path.join(
    OUTPUT_DIR,
    "step9_model_ranking.csv"
)

ranking_df.to_csv(
    ranking_file,
    index=False
)


# ============================================================
# 21. CREATE TEXT SUMMARY REPORT
# ============================================================

best_model = (
    overall_results_df.iloc[0]["model"]
)

best_macro_f1 = (
    overall_results_df.iloc[0]["macro_f1"]
)

best_balanced_accuracy = (
    overall_results_df.iloc[0]["balanced_accuracy"]
)

best_accuracy = (
    overall_results_df.iloc[0]["accuracy"]
)


report_file = os.path.join(
    OUTPUT_DIR,
    "step9_model_comparison_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 9 — FIVE-CLASS MODEL COMPARISON\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Dataset\n"
    )

    f.write(
        f"{INPUT_FILE}\n\n"
    )

    f.write(
        "Final test set status\n"
    )

    f.write(
        "five_class_final_test.csv was NOT loaded or used.\n\n"
    )

    f.write(
        "Target\n"
    )

    f.write(
        f"{TARGET_COLUMN}\n\n"
    )

    f.write(
        "Classes\n"
    )

    for class_name in CLASS_ORDER:
        f.write(
            f"- {class_name}\n"
        )

    f.write(
        "\nFeatures used\n"
    )

    for feature in FEATURE_COLUMNS:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nFeatures intentionally excluded because "
        "they directly contributed to the target:\n"
    )

    excluded_target_features = [
        "normal_fraction",
        "arrhythmia_fraction",
        "tachy_fraction",
        "brady_fraction",
        "other_abnormal_fraction",
        "dominant_label",
        "dominant_label_fraction",
    ]

    for feature in excluded_target_features:
        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nCross-validation\n"
    )

    f.write(
        "5 precomputed grouped CV folds from Step 8B.\n"
    )

    f.write(
        "Group leakage check: PASSED.\n"
    )

    f.write(
        "201 + 202 grouping check: PASSED.\n"
    )

    f.write(
        "\nModel ranking\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        overall_results_df[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Best model by macro F1: "
        f"{best_model}\n"
    )

    f.write(
        f"Best macro F1: "
        f"{best_macro_f1:.4f}\n"
    )

    f.write(
        f"Best balanced accuracy: "
        f"{best_balanced_accuracy:.4f}\n"
    )

    f.write(
        f"Best accuracy: "
        f"{best_accuracy:.4f}\n"
    )

    f.write(
        "\nIMPORTANT:\n"
    )

    f.write(
        "The model selected here is selected using grouped "
        "cross-validation only.\n"
    )

    f.write(
        "The final test set remains untouched and must only "
        "be evaluated after model selection is finalized.\n"
    )


# ============================================================
# 22. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)

print("\nOVERALL MODEL COMPARISON")
print("-" * 70)

display_columns = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "mean_fold_training_time_s",
    "mean_fold_prediction_time_s",
]

print(
    overall_results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)

print(
    f"BEST MODEL BY MACRO F1: {best_model}"
)

print(
    f"Macro F1          : {best_macro_f1:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{best_balanced_accuracy:.4f}"
)

print(
    f"Accuracy           : "
    f"{best_accuracy:.4f}"
)

print("=" * 70)


# ============================================================
# 23. OUTPUT FILE LIST
# ============================================================

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

output_files = sorted(
    os.listdir(OUTPUT_DIR)
)

for filename in output_files:
    print(
        f"  - {filename}"
    )


# ============================================================
# 24. FINAL SAFETY MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nThe Step 9 final test set was NOT used."
)

print(
    "The following file remains untouched:"
)

print(
    "five_class_final_test.csv"
)

print(
    "\nDo NOT evaluate the final test set yet."
)

print(
    "First inspect the Step 9 CV results and select the "
    "model/hyperparameters."
)

print(
    "\nAfter model selection, Step 10 can evaluate the "
    "locked final test set exactly once."
)

print("\nDone.")