# ============================================================
# STEP 7 - MODEL SELECTION AND PERFORMANCE ANALYSIS
# ============================================================
#
# Purpose:
#   Automatically analyze the results already produced by
#   Step 6 (ml_model_comparison).
#
# IMPORTANT:
#   - This script DOES NOT use final_test.csv.
#   - This script is only for model selection using the
#     development-set cross-validation results.
#   - final_test.csv must remain untouched until Step 9.
#
# Inputs:
#   ml_model_comparison/
#       summary_results.csv
#       fold_results.csv
#       per_class_recall.csv
#       label_mapping.csv
#       confusion_matrices/
#
# Outputs:
#   ml_model_comparison/
#       step7_model_selection/
#           model_selection_summary.csv
#           model_ranking.csv
#           core_vs_extended.csv
#           per_class_best_models.csv
#           model_class_performance.csv
#           rare_class_analysis.csv
#           confusion_matrix_analysis.csv
#           selected_model.txt
#           step7_model_selection_report.txt
#           step7_model_selection_report.md
#           model_ranking.png
#           macro_f1_comparison.png
#           balanced_accuracy_comparison.png
#           core_vs_extended_macro_f1.png
#           class_recall_heatmap.png
#
# ============================================================

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

RESULTS_DIR = PROJECT_DIR / "ml_model_comparison"

OUTPUT_DIR = RESULTS_DIR / "step7_model_selection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

SUMMARY_FILE = RESULTS_DIR / "summary_results.csv"
FOLD_FILE = RESULTS_DIR / "fold_results.csv"
RECALL_FILE = RESULTS_DIR / "per_class_recall.csv"
LABEL_FILE = RESULTS_DIR / "label_mapping.csv"

CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"


# Rare-class threshold.
# Classes with fewer than this many development examples
# are considered statistically unstable for model selection.
RARE_CLASS_THRESHOLD = 50


# Classes that are particularly important to inspect in the
# MIT-BIH beat classification task.
#
# This list is intentionally broad rather than treating every
# abnormal class as a disease diagnosis.
ABNORMAL_CLASSES = {
    "A",
    "a",
    "E",
    "F",
    "J",
    "L",
    "R",
    "S",
    "V",
    "e",
    "f",
    "j",
}


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def print_header(title):
    """Print a clear section header."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except Exception:
        return np.nan


def find_column(df, candidates, required=True):
    """
    Find the first matching column from a list of possible names.
    Matching is case-insensitive.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_model_name(value):
    """Normalize model names for easier comparison."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    replacements = {
        "LogisticRegression": "Logistic Regression",
        "RandomForest": "Random Forest",
        "RandomForestClassifier": "Random Forest",
        "SVC": "SVM",
        "Support Vector Machine": "SVM",
        "XGBClassifier": "XGBoost",
        "XGB": "XGBoost",
        "MLPClassifier": "MLP",
    }

    return replacements.get(value, value)


def normalize_feature_set(value):
    """Normalize Core/Extended labels."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() == "core":
        return "Core"

    if value.lower() == "extended":
        return "Extended"

    return value


def format_pct(value):
    """Format decimal or percentage-like value."""
    if pd.isna(value):
        return "N/A"

    value = float(value)

    # Metrics in the Step 6 output are decimals.
    if abs(value) <= 1.0:
        return f"{value * 100:.2f}%"

    return f"{value:.2f}%"


def format_seconds(value):
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.2f} s"


def detect_class_column(df):
    """
    Detect the class/label column in per_class_recall.csv.
    """
    candidates = [
        "class",
        "label",
        "beat_label",
        "true_label",
        "target",
        "class_label",
    ]

    return find_column(df, candidates, required=False)


def detect_recall_column(df):
    """
    Detect recall/sensitivity column.
    """
    candidates = [
        "recall",
        "sensitivity",
        "class_recall",
        "per_class_recall",
        "recall_sensitivity",
    ]

    return find_column(df, candidates, required=False)


# ------------------------------------------------------------
# 4. VERIFY INPUTS
# ------------------------------------------------------------

print_header("STEP 7 - MODEL SELECTION REPORT")

print(f"Project directory:")
print(f"  {PROJECT_DIR}")

print(f"\nModel comparison directory:")
print(f"  {RESULTS_DIR}")

print(f"\nOutput directory:")
print(f"  {OUTPUT_DIR}")


required_files = [
    SUMMARY_FILE,
    FOLD_FILE,
    RECALL_FILE,
]

missing_files = [
    path for path in required_files
    if not path.exists()
]

if missing_files:
    print("\nERROR: Required Step 6 files are missing:")

    for path in missing_files:
        print(f"  - {path}")

    print("\nMake sure Step 6 completed successfully before running Step 7.")
    sys.exit(1)


# ------------------------------------------------------------
# 5. EXPLICITLY PROTECT final_test.csv
# ------------------------------------------------------------

print_header("FINAL TEST PROTECTION CHECK")

FINAL_TEST = PROJECT_DIR / "ml_grouped_cv" / "final_test.csv"

print(
    "This Step 7 script will NOT load, evaluate, train on, "
    "or otherwise use final_test.csv."
)

if FINAL_TEST.exists():
    print("final_test.csv detected:")
    print(f"  {FINAL_TEST}")
    print("Status: PROTECTED / NOT USED")
else:
    print("final_test.csv is not present at the expected location.")
    print("Status: No final test file accessed.")


# ------------------------------------------------------------
# 6. LOAD STEP 6 RESULTS
# ------------------------------------------------------------

print_header("LOADING STEP 6 RESULTS")

summary_df = pd.read_csv(SUMMARY_FILE)
fold_df = pd.read_csv(FOLD_FILE)
recall_df = pd.read_csv(RECALL_FILE)

print(f"summary_results.csv:")
print(f"  Rows: {len(summary_df)}")
print(f"  Columns: {len(summary_df.columns)}")

print(f"\nfold_results.csv:")
print(f"  Rows: {len(fold_df)}")
print(f"  Columns: {len(fold_df.columns)}")

print(f"\nper_class_recall.csv:")
print(f"  Rows: {len(recall_df)}")
print(f"  Columns: {len(recall_df.columns)}")


# ------------------------------------------------------------
# 7. NORMALIZE SUMMARY COLUMNS
# ------------------------------------------------------------

print_header("PREPARING MODEL SUMMARY")

summary = summary_df.copy()

# Detect important columns.
model_col = find_column(
    summary,
    ["model", "model_name", "classifier"],
    required=False,
)

feature_col = find_column(
    summary,
    ["feature_set", "features", "feature_group", "feature_type"],
    required=False,
)

accuracy_col = find_column(
    summary,
    ["accuracy", "oof_accuracy"],
    required=False,
)

balanced_col = find_column(
    summary,
    [
        "balanced_accuracy",
        "balanced_acc",
        "oof_balanced_accuracy",
    ],
    required=False,
)

macro_f1_col = find_column(
    summary,
    [
        "macro_f1",
        "oof_macro_f1",
        "overall_macro_f1",
    ],
    required=False,
)

mean_fold_f1_col = find_column(
    summary,
    [
        "mean_fold_macro_f1",
        "mean_macro_f1",
        "cv_mean_macro_f1",
    ],
    required=False,
)

std_fold_f1_col = find_column(
    summary,
    [
        "std_fold_macro_f1",
        "std_macro_f1",
        "cv_std_macro_f1",
    ],
    required=False,
)

train_time_col = find_column(
    summary,
    [
        "mean_train_time_sec",
        "mean_training_time_sec",
        "training_time_sec",
        "train_time_sec",
    ],
    required=False,
)


# If model/feature information is not in summary, try fold file.
if model_col is None:
    model_col = find_column(
        fold_df,
        ["model", "model_name", "classifier"],
        required=True,
    )

if feature_col is None:
    feature_col = find_column(
        fold_df,
        ["feature_set", "features", "feature_group", "feature_type"],
        required=True,
    )


# Create standardized columns.
if model_col in summary.columns:
    summary["Model"] = summary[model_col].apply(normalize_model_name)

if feature_col in summary.columns:
    summary["Feature_Set"] = summary[feature_col].apply(
        normalize_feature_set
    )

if accuracy_col is not None:
    summary["Accuracy"] = pd.to_numeric(
        summary[accuracy_col],
        errors="coerce",
    )

if balanced_col is not None:
    summary["Balanced_Accuracy"] = pd.to_numeric(
        summary[balanced_col],
        errors="coerce",
    )

if macro_f1_col is not None:
    summary["Macro_F1"] = pd.to_numeric(
        summary[macro_f1_col],
        errors="coerce",
    )

if mean_fold_f1_col is not None:
    summary["Mean_Fold_Macro_F1"] = pd.to_numeric(
        summary[mean_fold_f1_col],
        errors="coerce",
    )

if std_fold_f1_col is not None:
    summary["Std_Fold_Macro_F1"] = pd.to_numeric(
        summary[std_fold_f1_col],
        errors="coerce",
    )

if train_time_col is not None:
    summary["Mean_Train_Time_Sec"] = pd.to_numeric(
        summary[train_time_col],
        errors="coerce",
    )


required_standardized = [
    "Model",
    "Feature_Set",
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
]

missing_standardized = [
    col
    for col in required_standardized
    if col not in summary.columns
]

if missing_standardized:
    print("\nERROR: Could not identify required metrics:")
    for col in missing_standardized:
        print(f"  - {col}")

    print("\nAvailable summary columns:")
    for col in summary.columns:
        print(f"  - {col}")

    sys.exit(1)


# ------------------------------------------------------------
# 8. CHECK FOR DUPLICATES
# ------------------------------------------------------------

model_keys = ["Model", "Feature_Set"]

duplicate_count = summary.duplicated(
    subset=model_keys,
    keep=False,
).sum()

if duplicate_count > 0:

    print(
        f"\nWARNING: Found {duplicate_count} duplicated "
        "model/feature rows in summary."
    )

    # Keep the row with the best Macro F1.
    summary = (
        summary
        .sort_values("Macro_F1", ascending=False)
        .drop_duplicates(
            subset=model_keys,
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "Duplicate rows were resolved by keeping the highest "
        "OOF Macro F1 result."
    )


# ------------------------------------------------------------
# 9. PREPARE PER-CLASS RECALL DATA
# ------------------------------------------------------------

print_header("PREPARING PER-CLASS RECALL ANALYSIS")

recalls = recall_df.copy()

recall_model_col = find_column(
    recalls,
    ["model", "model_name", "classifier"],
    required=False,
)

recall_feature_col = find_column(
    recalls,
    ["feature_set", "features", "feature_group", "feature_type"],
    required=False,
)

class_col = detect_class_column(recalls)
recall_value_col = detect_recall_column(recalls)


if recall_model_col is None:
    print(
        "\nWARNING: Model column could not be identified in "
        "per_class_recall.csv."
    )

if recall_feature_col is None:
    print(
        "WARNING: Feature-set column could not be identified in "
        "per_class_recall.csv."
    )

if class_col is None:
    print(
        "WARNING: Class column could not be identified in "
        "per_class_recall.csv."
    )

if recall_value_col is None:
    print(
        "WARNING: Recall column could not be identified in "
        "per_class_recall.csv."
    )


recall_ready = False

if all(
    x is not None
    for x in [
        recall_model_col,
        recall_feature_col,
        class_col,
        recall_value_col,
    ]
):

    recalls["Model"] = recalls[recall_model_col].apply(
        normalize_model_name
    )

    recalls["Feature_Set"] = recalls[recall_feature_col].apply(
        normalize_feature_set
    )

    recalls["Class"] = recalls[class_col].astype(str)

    recalls["Recall"] = pd.to_numeric(
        recalls[recall_value_col],
        errors="coerce",
    )

    recall_ready = True

    print(
        f"Prepared {len(recalls)} per-class recall records."
    )


# ------------------------------------------------------------
# 10. BUILD MODEL RANKING
# ------------------------------------------------------------

print_header("BUILDING MODEL RANKING")

ranking = summary.copy()


# ------------------------------------------------------------
# 11. CALCULATE ABNORMAL-CLASS SCORE
# ------------------------------------------------------------

if recall_ready:

    abnormal_recall = (
        recalls[
            recalls["Class"].isin(ABNORMAL_CLASSES)
        ]
        .groupby(
            ["Model", "Feature_Set"],
            as_index=False,
        )["Recall"]
        .mean()
        .rename(
            columns={
                "Recall": "Mean_Abnormal_Class_Recall"
            }
        )
    )

    ranking = ranking.merge(
        abnormal_recall,
        on=["Model", "Feature_Set"],
        how="left",
    )

else:

    ranking["Mean_Abnormal_Class_Recall"] = np.nan


# ------------------------------------------------------------
# 12. CALCULATE NON-RARE ABNORMAL RECALL
# ------------------------------------------------------------

#
# We need class support information.
#
# If support is available in per_class_recall.csv, use it.
#

support_col = None

if recall_ready:

    support_col = find_column(
        recalls,
        [
            "support",
            "class_support",
            "n_samples",
            "count",
            "true_count",
        ],
        required=False,
    )


if recall_ready and support_col is not None:

    recalls["Support"] = pd.to_numeric(
        recalls[support_col],
        errors="coerce",
    )

    stable_abnormal = recalls[
        (recalls["Class"].isin(ABNORMAL_CLASSES))
        & (recalls["Support"] >= RARE_CLASS_THRESHOLD)
    ]

    stable_abnormal_recall = (
        stable_abnormal
        .groupby(
            ["Model", "Feature_Set"],
            as_index=False,
        )["Recall"]
        .mean()
        .rename(
            columns={
                "Recall":
                "Mean_Stable_Abnormal_Recall"
            }
        )
    )

    ranking = ranking.merge(
        stable_abnormal_recall,
        on=["Model", "Feature_Set"],
        how="left",
    )

else:

    ranking["Mean_Stable_Abnormal_Recall"] = np.nan


# ------------------------------------------------------------
# 13. MODEL-SELECTION SCORING
# ------------------------------------------------------------

#
# IMPORTANT:
#
# We do NOT use raw Accuracy as the primary selection metric
# because the dataset is heavily imbalanced.
#
# Primary:
#   OOF Macro F1
#
# Secondary:
#   OOF Balanced Accuracy
#
# Additional:
#   Mean abnormal-class recall
#   Stable abnormal-class recall
#   Training time
#
# Accuracy is reported but not used as the primary ranking
# criterion.
#

ranking["Macro_F1_Rank"] = ranking[
    "Macro_F1"
].rank(
    ascending=False,
    method="min",
)

ranking["Balanced_Accuracy_Rank"] = ranking[
    "Balanced_Accuracy"
].rank(
    ascending=False,
    method="min",
)

if ranking[
    "Mean_Abnormal_Class_Recall"
].notna().any():

    ranking["Abnormal_Recall_Rank"] = ranking[
        "Mean_Abnormal_Class_Recall"
    ].rank(
        ascending=False,
        method="min",
    )

else:

    ranking["Abnormal_Recall_Rank"] = np.nan


if ranking[
    "Mean_Stable_Abnormal_Recall"
].notna().any():

    ranking["Stable_Abnormal_Recall_Rank"] = ranking[
        "Mean_Stable_Abnormal_Recall"
    ].rank(
        ascending=False,
        method="min",
    )

else:

    ranking["Stable_Abnormal_Recall_Rank"] = np.nan


# Primary ranking:
# 1. Macro F1
# 2. Balanced accuracy
# 3. Stable abnormal recall
# 4. Abnormal recall
#

sort_columns = [
    "Macro_F1",
    "Balanced_Accuracy",
]

ascending_order = [
    False,
    False,
]


if "Mean_Stable_Abnormal_Recall" in ranking.columns:

    sort_columns.append(
        "Mean_Stable_Abnormal_Recall"
    )

    ascending_order.append(False)


if "Mean_Abnormal_Class_Recall" in ranking.columns:

    sort_columns.append(
        "Mean_Abnormal_Class_Recall"
    )

    ascending_order.append(False)


ranking = ranking.sort_values(
    by=sort_columns,
    ascending=ascending_order,
).reset_index(drop=True)


ranking["Overall_Rank"] = np.arange(
    1,
    len(ranking) + 1,
)


# ------------------------------------------------------------
# 14. ADD MODEL-SELECTION REASON
# ------------------------------------------------------------

def build_selection_reason(row):

    model = row["Model"]
    feature_set = row["Feature_Set"]

    return (
        f"{model} ({feature_set}): "
        f"OOF Macro F1={format_pct(row['Macro_F1'])}, "
        f"Balanced Accuracy="
        f"{format_pct(row['Balanced_Accuracy'])}, "
        f"Accuracy={format_pct(row['Accuracy'])}."
    )


ranking["Selection_Summary"] = ranking.apply(
    build_selection_reason,
    axis=1,
)


# ------------------------------------------------------------
# 15. CORE VS EXTENDED ANALYSIS
# ------------------------------------------------------------

print_header("CORE VS EXTENDED FEATURE ANALYSIS")

core_extended = (
    summary[
        summary["Feature_Set"].isin(
            ["Core", "Extended"]
        )
    ]
    .copy()
)

if not core_extended.empty:

    core_extended = core_extended[
        [
            "Model",
            "Feature_Set",
            "Accuracy",
            "Balanced_Accuracy",
            "Macro_F1",
        ]
    ]

    pivot_macro = core_extended.pivot(
        index="Model",
        columns="Feature_Set",
        values="Macro_F1",
    )

    pivot_balanced = core_extended.pivot(
        index="Model",
        columns="Feature_Set",
        values="Balanced_Accuracy",
    )

    pivot_accuracy = core_extended.pivot(
        index="Model",
        columns="Feature_Set",
        values="Accuracy",
    )

    core_extended_summary = pd.DataFrame(
        index=pivot_macro.index
    )

    if "Core" in pivot_macro.columns:
        core_extended_summary["Core_Macro_F1"] = (
            pivot_macro["Core"]
        )

    if "Extended" in pivot_macro.columns:
        core_extended_summary["Extended_Macro_F1"] = (
            pivot_macro["Extended"]
        )

    if (
        "Core" in pivot_macro.columns
        and "Extended" in pivot_macro.columns
    ):

        core_extended_summary[
            "Macro_F1_Improvement"
        ] = (
            core_extended_summary[
                "Extended_Macro_F1"
            ]
            - core_extended_summary[
                "Core_Macro_F1"
            ]
        )

    if "Core" in pivot_balanced.columns:
        core_extended_summary[
            "Core_Balanced_Accuracy"
        ] = pivot_balanced["Core"]

    if "Extended" in pivot_balanced.columns:
        core_extended_summary[
            "Extended_Balanced_Accuracy"
        ] = pivot_balanced["Extended"]

    if (
        "Core" in pivot_balanced.columns
        and "Extended" in pivot_balanced.columns
    ):

        core_extended_summary[
            "Balanced_Accuracy_Improvement"
        ] = (
            core_extended_summary[
                "Extended_Balanced_Accuracy"
            ]
            - core_extended_summary[
                "Core_Balanced_Accuracy"
            ]
        )

    if "Core" in pivot_accuracy.columns:
        core_extended_summary[
            "Core_Accuracy"
        ] = pivot_accuracy["Core"]

    if "Extended" in pivot_accuracy.columns:
        core_extended_summary[
            "Extended_Accuracy"
        ] = pivot_accuracy["Extended"]

    if (
        "Core" in pivot_accuracy.columns
        and "Extended" in pivot_accuracy.columns
    ):

        core_extended_summary[
            "Accuracy_Improvement"
        ] = (
            core_extended_summary[
                "Extended_Accuracy"
            ]
            - core_extended_summary[
                "Core_Accuracy"
            ]
        )

    core_extended_summary = (
        core_extended_summary
        .reset_index()
        .rename(columns={"index": "Model"})
    )

else:

    core_extended_summary = pd.DataFrame()


# ------------------------------------------------------------
# 16. PER-CLASS BEST MODEL ANALYSIS
# ------------------------------------------------------------

print_header("PER-CLASS MODEL ANALYSIS")

if recall_ready:

    # Best model for each class based on recall.
    class_best = recalls.copy()

    class_best = class_best.sort_values(
        "Recall",
        ascending=False,
    )

    class_best = (
        class_best
        .groupby("Class", as_index=False)
        .first()
    )

    class_best = class_best[
        [
            "Class",
            "Model",
            "Feature_Set",
            "Recall",
        ]
    ].sort_values(
        "Class"
    )

else:

    class_best = pd.DataFrame()


# ------------------------------------------------------------
# 17. RARE CLASS ANALYSIS
# ------------------------------------------------------------

print_header("RARE CLASS ANALYSIS")

if recall_ready and "Support" in recalls.columns:

    rare_analysis = (
        recalls[
            recalls["Support"] < RARE_CLASS_THRESHOLD
        ]
        .copy()
    )

    rare_analysis = rare_analysis.sort_values(
        ["Class", "Recall"],
        ascending=[True, False],
    )

else:

    rare_analysis = pd.DataFrame()


# ------------------------------------------------------------
# 18. CONFUSION MATRIX FILE ANALYSIS
# ------------------------------------------------------------

print_header("CONFUSION MATRIX CHECK")

confusion_records = []

if CONFUSION_DIR.exists():

    confusion_files = list(
        CONFUSION_DIR.glob("*.csv")
    )

    print(
        f"Found {len(confusion_files)} confusion-matrix CSV files."
    )

    for file in confusion_files:

        name = file.stem

        try:

            cm = pd.read_csv(file)

            confusion_records.append(
                {
                    "File": file.name,
                    "Rows": cm.shape[0],
                    "Columns": cm.shape[1],
                    "Readable": True,
                }
            )

        except Exception as exc:

            confusion_records.append(
                {
                    "File": file.name,
                    "Rows": np.nan,
                    "Columns": np.nan,
                    "Readable": False,
                    "Error": str(exc),
                }
            )

else:

    print(
        "Confusion-matrix directory not found."
    )


confusion_analysis = pd.DataFrame(
    confusion_records
)


# ------------------------------------------------------------
# 19. DETERMINE PRELIMINARY WINNER
# ------------------------------------------------------------

print_header("PRELIMINARY MODEL SELECTION")

ranking = ranking.sort_values(
    [
        "Macro_F1",
        "Balanced_Accuracy",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


winner = ranking.iloc[0]

winner_model = winner["Model"]
winner_features = winner["Feature_Set"]

winner_macro_f1 = winner["Macro_F1"]
winner_balanced = winner["Balanced_Accuracy"]
winner_accuracy = winner["Accuracy"]


print("\nPreliminary best model:")
print(
    f"  Model: {winner_model}"
)
print(
    f"  Features: {winner_features}"
)
print(
    f"  OOF Macro F1: {format_pct(winner_macro_f1)}"
)
print(
    f"  OOF Balanced Accuracy: "
    f"{format_pct(winner_balanced)}"
)
print(
    f"  OOF Accuracy: "
    f"{format_pct(winner_accuracy)}"
)


# ------------------------------------------------------------
# 20. CHECK WHETHER WINNER IS CLEAR
# ------------------------------------------------------------

selection_confidence = "Moderate"

if len(ranking) >= 2:

    second = ranking.iloc[1]

    macro_f1_gap = (
        winner_macro_f1
        - second["Macro_F1"]
    )

    balanced_gap = (
        winner_balanced
        - second["Balanced_Accuracy"]
    )

    print(
        f"\nDifference from second-ranked model:"
    )

    print(
        f"  Macro F1 gap: "
        f"{format_pct(macro_f1_gap)}"
    )

    print(
        f"  Balanced accuracy gap: "
        f"{format_pct(balanced_gap)}"
    )

    # Small difference means the winner should be treated
    # cautiously.
    if abs(macro_f1_gap) < 0.01:

        selection_confidence = "Low"

    elif abs(macro_f1_gap) >= 0.03:

        selection_confidence = "Higher"


# ------------------------------------------------------------
# 21. BUILD HUMAN-READABLE MODEL TABLE
# ------------------------------------------------------------

model_table_columns = [
    "Overall_Rank",
    "Model",
    "Feature_Set",
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
]

optional_columns = [
    "Mean_Fold_Macro_F1",
    "Std_Fold_Macro_F1",
    "Mean_Abnormal_Class_Recall",
    "Mean_Stable_Abnormal_Recall",
    "Mean_Train_Time_Sec",
]

for col in optional_columns:

    if col in ranking.columns:
        model_table_columns.append(col)


model_table = ranking[
    model_table_columns
].copy()


# ------------------------------------------------------------
# 22. SAVE CSV RESULTS
# ------------------------------------------------------------

print_header("SAVING ANALYSIS FILES")

ranking.to_csv(
    OUTPUT_DIR / "model_ranking.csv",
    index=False,
)

model_table.to_csv(
    OUTPUT_DIR / "model_selection_summary.csv",
    index=False,
)

if not core_extended_summary.empty:

    core_extended_summary.to_csv(
        OUTPUT_DIR / "core_vs_extended.csv",
        index=False,
    )

if not class_best.empty:

    class_best.to_csv(
        OUTPUT_DIR / "per_class_best_models.csv",
        index=False,
    )

if recall_ready:

    recalls.to_csv(
        OUTPUT_DIR / "model_class_performance.csv",
        index=False,
    )

if not rare_analysis.empty:

    rare_analysis.to_csv(
        OUTPUT_DIR / "rare_class_analysis.csv",
        index=False,
    )

if not confusion_analysis.empty:

    confusion_analysis.to_csv(
        OUTPUT_DIR / "confusion_matrix_analysis.csv",
        index=False,
    )


# ------------------------------------------------------------
# 23. SAVE SELECTED MODEL
# ------------------------------------------------------------

selected_model_text = f"""
STEP 7 PRELIMINARY MODEL SELECTION
===================================

Selected model for Step 8:
    Model: {winner_model}
    Feature set: {winner_features}

OOF metrics:
    Accuracy: {format_pct(winner_accuracy)}
    Balanced accuracy: {format_pct(winner_balanced)}
    Macro F1: {format_pct(winner_macro_f1)}

Selection confidence:
    {selection_confidence}

IMPORTANT:
This is a development/CV-based model selection result.

final_test.csv was NOT used.

The selected model must still be evaluated exactly once
on final_test.csv in the final evaluation stage.

Do not describe this result as a final unbiased test result.
"""


with open(
    OUTPUT_DIR / "selected_model.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        selected_model_text.strip()
        + "\n"
    )


# ------------------------------------------------------------
# 24. CREATE MODEL RANKING CHART
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

plot_df = ranking.copy()

plot_df["Model_Features"] = (
    plot_df["Model"]
    + " - "
    + plot_df["Feature_Set"]
)

plot_df = plot_df.sort_values(
    "Macro_F1",
    ascending=True,
)

plt.barh(
    plot_df["Model_Features"],
    plot_df["Macro_F1"] * 100,
)

plt.xlabel("OOF Macro F1 (%)")
plt.ylabel("Model and feature set")
plt.title(
    "Model Comparison - OOF Macro F1"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_ranking.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# 25. BALANCED ACCURACY CHART
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

plot_df = ranking.copy()

plot_df["Model_Features"] = (
    plot_df["Model"]
    + " - "
    + plot_df["Feature_Set"]
)

plot_df = plot_df.sort_values(
    "Balanced_Accuracy",
    ascending=True,
)

plt.barh(
    plot_df["Model_Features"],
    plot_df["Balanced_Accuracy"] * 100,
)

plt.xlabel("OOF Balanced Accuracy (%)")
plt.ylabel("Model and feature set")
plt.title(
    "Model Comparison - OOF Balanced Accuracy"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "balanced_accuracy_comparison.png",
    dpi=200,
)

plt.close()


# ------------------------------------------------------------
# 26. CORE VS EXTENDED CHART
# ------------------------------------------------------------

if not core_extended_summary.empty:

    if (
        "Core_Macro_F1" in core_extended_summary.columns
        and "Extended_Macro_F1"
        in core_extended_summary.columns
    ):

        chart_df = core_extended_summary.copy()

        x = np.arange(len(chart_df))
        width = 0.35

        plt.figure(figsize=(11, 7))

        plt.bar(
            x - width / 2,
            chart_df["Core_Macro_F1"] * 100,
            width,
            label="Core",
        )

        plt.bar(
            x + width / 2,
            chart_df["Extended_Macro_F1"] * 100,
            width,
            label="Extended",
        )

        plt.xticks(
            x,
            chart_df["Model"],
            rotation=35,
            ha="right",
        )

        plt.ylabel("OOF Macro F1 (%)")
        plt.xlabel("Model")
        plt.title(
            "Core vs Extended ECG Features"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "core_vs_extended_macro_f1.png",
            dpi=200,
        )

        plt.close()


# ------------------------------------------------------------
# 27. PER-CLASS RECALL HEATMAP
# ------------------------------------------------------------

if recall_ready:

    try:

        heatmap_data = recalls.pivot_table(
            index="Class",
            columns=[
                "Model",
                "Feature_Set",
            ],
            values="Recall",
            aggfunc="mean",
        )

        # Sort classes alphabetically.
        heatmap_data = heatmap_data.sort_index()

        plt.figure(
            figsize=(
                max(12, heatmap_data.shape[1] * 1.2),
                max(7, heatmap_data.shape[0] * 0.45),
            )
        )

        plt.imshow(
            heatmap_data.values,
            aspect="auto",
            interpolation="nearest",
        )

        plt.colorbar(
            label="Recall / Sensitivity"
        )

        plt.xticks(
            range(heatmap_data.shape[1]),
            [
                f"{a} - {b}"
                for a, b
                in heatmap_data.columns
            ],
            rotation=70,
            ha="right",
        )

        plt.yticks(
            range(heatmap_data.shape[0]),
            heatmap_data.index,
        )

        plt.xlabel(
            "Model and feature set"
        )

        plt.ylabel(
            "Beat class"
        )

        plt.title(
            "Per-Class Recall / Sensitivity"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "class_recall_heatmap.png",
            dpi=200,
        )

        plt.close()

    except Exception as exc:

        print(
            "\nWARNING: Could not create recall heatmap:"
        )

        print(exc)


# ------------------------------------------------------------
# 28. BUILD MARKDOWN REPORT
# ------------------------------------------------------------

print_header("CREATING FINAL REPORT")

report_lines = []

report_lines.append(
    "# Step 7 - Model Selection Report"
)

report_lines.append("")

report_lines.append(
    "## Purpose"
)

report_lines.append("")

report_lines.append(
    "This report analyzes the development-set cross-validation "
    "results generated by Step 6. It is used to select a "
    "candidate model for the next training stage."
)

report_lines.append("")

report_lines.append(
    "**Important:** `final_test.csv` was not used."
)

report_lines.append("")

report_lines.append(
    "## 1. Model Selection Principle"
)

report_lines.append("")

report_lines.append(
    "Because the ECG beat classes are strongly imbalanced, "
    "raw accuracy is not used as the primary model-selection "
    "criterion."
)

report_lines.append("")

report_lines.append(
    "The main metrics considered are:"
)

report_lines.append("")

report_lines.append(
    "- OOF Macro F1"
)

report_lines.append(
    "- OOF Balanced Accuracy"
)

report_lines.append(
    "- Per-class recall/sensitivity"
)

report_lines.append(
    "- Abnormal-class performance"
)

report_lines.append(
    "- Core versus Extended feature performance"
)

report_lines.append("")

report_lines.append(
    "## 2. Model Ranking"
)

report_lines.append("")

report_lines.append(
    "| Rank | Model | Features | Accuracy | "
    "Balanced Accuracy | Macro F1 |"
)

report_lines.append(
    "|---:|---|---|---:|---:|---:|"
)

for _, row in ranking.iterrows():

    report_lines.append(
        f"| {int(row['Overall_Rank'])} | "
        f"{row['Model']} | "
        f"{row['Feature_Set']} | "
        f"{format_pct(row['Accuracy'])} | "
        f"{format_pct(row['Balanced_Accuracy'])} | "
        f"{format_pct(row['Macro_F1'])} |"
    )


report_lines.append("")

report_lines.append(
    "## 3. Preliminary Selected Model"
)

report_lines.append("")

report_lines.append(
    f"**Model:** {winner_model}"
)

report_lines.append("")

report_lines.append(
    f"**Feature set:** {winner_features}"
)

report_lines.append("")

report_lines.append(
    f"**OOF Macro F1:** "
    f"{format_pct(winner_macro_f1)}"
)

report_lines.append("")

report_lines.append(
    f"**OOF Balanced Accuracy:** "
    f"{format_pct(winner_balanced)}"
)

report_lines.append("")

report_lines.append(
    f"**OOF Accuracy:** "
    f"{format_pct(winner_accuracy)}"
)

report_lines.append("")

report_lines.append(
    f"**Selection confidence:** "
    f"{selection_confidence}"
)

report_lines.append("")

report_lines.append(
    "This is a preliminary development/CV selection. "
    "It is not the final unbiased test result."
)

report_lines.append("")


# ------------------------------------------------------------
# 29. INTERPRETATION OF WINNER
# ------------------------------------------------------------

report_lines.append(
    "## 4. Interpretation"
)

report_lines.append("")

if winner_accuracy > winner_macro_f1:

    report_lines.append(
        "The accuracy is substantially higher than the Macro F1. "
        "This is expected in a strongly imbalanced ECG dataset "
        "because common classes can dominate overall accuracy."
    )

    report_lines.append("")

report_lines.append(
    "Therefore, the selected model should not be described "
    "as successful merely because it has high accuracy."
)

report_lines.append("")

report_lines.append(
    "The more important question is whether the model can "
    "correctly recognize multiple ECG beat classes, including "
    "less common abnormal classes."
)

report_lines.append("")


# ------------------------------------------------------------
# 30. CORE VS EXTENDED REPORT
# ------------------------------------------------------------

report_lines.append(
    "## 5. Core vs Extended Features"
)

report_lines.append("")

if not core_extended_summary.empty:

    report_lines.append(
        "| Model | Core Macro F1 | Extended Macro F1 | "
        "Improvement |"
    )

    report_lines.append(
        "|---|---:|---:|---:|"
    )

    for _, row in core_extended_summary.iterrows():

        core_value = row.get(
            "Core_Macro_F1",
            np.nan,
        )

        extended_value = row.get(
            "Extended_Macro_F1",
            np.nan,
        )

        improvement = row.get(
            "Macro_F1_Improvement",
            np.nan,
        )

        report_lines.append(
            f"| {row['Model']} | "
            f"{format_pct(core_value)} | "
            f"{format_pct(extended_value)} | "
            f"{format_pct(improvement)} |"
        )

else:

    report_lines.append(
        "Core-versus-Extended comparison could not be generated."
    )

report_lines.append("")


# ------------------------------------------------------------
# 31. PER-CLASS REPORT
# ------------------------------------------------------------

report_lines.append(
    "## 6. Per-Class Recall / Sensitivity"
)

report_lines.append("")

if not class_best.empty:

    report_lines.append(
        "The table below identifies the model/feature combination "
        "with the highest observed recall for each class."
    )

    report_lines.append("")

    report_lines.append(
        "| Class | Best Model | Features | Recall |"
    )

    report_lines.append(
        "|---|---|---|---:|"
    )

    for _, row in class_best.iterrows():

        report_lines.append(
            f"| {row['Class']} | "
            f"{row['Model']} | "
            f"{row['Feature_Set']} | "
            f"{format_pct(row['Recall'])} |"
        )

else:

    report_lines.append(
        "Per-class recall data could not be interpreted."
    )

report_lines.append("")


# ------------------------------------------------------------
# 32. RARE CLASS WARNING
# ------------------------------------------------------------

report_lines.append(
    "## 7. Rare-Class Warning"
)

report_lines.append("")

report_lines.append(
    f"Classes with fewer than "
    f"{RARE_CLASS_THRESHOLD} development examples are "
    "treated as statistically unstable for detailed "
    "model-selection conclusions."
)

report_lines.append("")

if not rare_analysis.empty:

    rare_classes = sorted(
        rare_analysis["Class"].unique()
    )

    report_lines.append(
        "Rare/low-support classes detected:"
    )

    report_lines.append("")

    report_lines.append(
        ", ".join(rare_classes)
    )

    report_lines.append("")

    report_lines.append(
        "Recall values for these classes can change substantially "
        "depending on which record/group appears in a CV fold."
    )

else:

    report_lines.append(
        "No rare-class support information was available."
    )

report_lines.append("")


# ------------------------------------------------------------
# 33. CV LIMITATION REPORT
# ------------------------------------------------------------

report_lines.append(
    "## 8. Cross-Validation Limitations"
)

report_lines.append("")

report_lines.append(
    "The MIT-BIH beat classes are highly imbalanced."
)

report_lines.append("")

report_lines.append(
    "Some classes occur in very few beats and/or very few "
    "independent record groups. Because the project correctly "
    "keeps related ECG records together during splitting, "
    "some CV folds may contain very few or no examples of "
    "certain rare classes in the training data."
)

report_lines.append("")

report_lines.append(
    "This is a limitation of the available data rather than "
    "evidence of data leakage."
)

report_lines.append("")


# ------------------------------------------------------------
# 34. DISEASE INTERPRETATION WARNING
# ------------------------------------------------------------

report_lines.append(
    "## 9. Clinical Interpretation Boundary"
)

report_lines.append("")

report_lines.append(
    "The current machine-learning task is ECG beat classification."
)

report_lines.append("")

report_lines.append(
    "A predicted beat label such as N, V, A, L, R, etc. should "
    "not automatically be interpreted as a definitive patient "
    "disease diagnosis."
)

report_lines.append("")

report_lines.append(
    "A later system layer may aggregate beat-level predictions "
    "into an abnormal-pattern or alert indication, but that "
    "requires separate validation before making clinical claims."
)

report_lines.append("")


# ------------------------------------------------------------
# 35. STEP 8 RECOMMENDATION
# ------------------------------------------------------------

report_lines.append(
    "## 10. Recommendation for Step 8"
)

report_lines.append("")

report_lines.append(
    f"Proceed to Step 8 using:"
)

report_lines.append("")

report_lines.append(
    f"**{winner_model} + {winner_features} features**"
)

report_lines.append("")

report_lines.append(
    "The next step should retrain this selected model using "
    "the complete development dataset while keeping "
    "`final_test.csv` completely untouched."
)

report_lines.append("")

report_lines.append(
    "After the model is finalized, Step 9 should evaluate it "
    "exactly once on the held-out final test set."
)

report_lines.append("")


# ------------------------------------------------------------
# 36. FILE OUTPUTS
# ------------------------------------------------------------

report_lines.append(
    "## 11. Generated Files"
)

report_lines.append("")

report_lines.append(
    "- `model_ranking.csv`"
)

report_lines.append(
    "- `model_selection_summary.csv`"
)

report_lines.append(
    "- `core_vs_extended.csv`"
)

report_lines.append(
    "- `per_class_best_models.csv`"
)

report_lines.append(
    "- `model_class_performance.csv`"
)

report_lines.append(
    "- `rare_class_analysis.csv`"
)

report_lines.append(
    "- `confusion_matrix_analysis.csv`"
)

report_lines.append(
    "- `selected_model.txt`"
)

report_lines.append(
    "- `model_ranking.png`"
)

report_lines.append(
    "- `balanced_accuracy_comparison.png`"
)

report_lines.append(
    "- `core_vs_extended_macro_f1.png`"
)

report_lines.append(
    "- `class_recall_heatmap.png`"
)

report_lines.append("")


# ------------------------------------------------------------
# 37. WRITE MARKDOWN REPORT
# ------------------------------------------------------------

markdown_report = "\n".join(
    report_lines
)

with open(
    OUTPUT_DIR / "step7_model_selection_report.md",
    "w",
    encoding="utf-8",
) as f:

    f.write(markdown_report)


# ------------------------------------------------------------
# 38. WRITE PLAIN-TEXT REPORT
# ------------------------------------------------------------

plain_report = markdown_report

plain_report = plain_report.replace(
    "**",
    "",
)

with open(
    OUTPUT_DIR / "step7_model_selection_report.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(plain_report)


# ------------------------------------------------------------
# 39. FINAL CONSOLE SUMMARY
# ------------------------------------------------------------

print_header("STEP 7 COMPLETE")

print(
    "\nPreliminary model selected for Step 8:"
)

print(
    f"  Model        : {winner_model}"
)

print(
    f"  Feature set  : {winner_features}"
)

print(
    f"  Macro F1     : {format_pct(winner_macro_f1)}"
)

print(
    f"  Balanced Acc : {format_pct(winner_balanced)}"
)

print(
    f"  Accuracy     : {format_pct(winner_accuracy)}"
)

print(
    f"  Confidence   : {selection_confidence}"
)

print(
    "\nIMPORTANT:"
)

print(
    "  final_test.csv was NOT used."
)

print(
    "  final_test.csv remains reserved for the final evaluation."
)

print(
    "\nReports and analysis files saved to:"
)

print(
    f"  {OUTPUT_DIR}"
)

print(
    "\nMain report:"
)

print(
    f"  {OUTPUT_DIR / 'step7_model_selection_report.md'}"
)

print(
    "\nSelected model:"
)

print(
    f"  {OUTPUT_DIR / 'selected_model.txt'}"
)

print(
    "\nStep 7 finished successfully."
)

print(
    "Next stage: Step 8 - retrain the selected model on the "
    "complete development dataset."
)

# ============================================================
# END OF STEP 7
# ============================================================