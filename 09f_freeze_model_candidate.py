# ============================================================
# STEP 9F — FREEZE SELECTED MODEL CANDIDATE
#
# Purpose:
#   Record the selected model and configuration after Step 9E.
#
# Important:
#   - Does NOT train a model
#   - Does NOT load or evaluate the final test set
#   - Uses Step 9E development OOF results only
# ============================================================

from pathlib import Path
import json
from datetime import datetime

import pandas as pd


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

STEP9E_DIR = PROJECT_DIR / "step9e_candidate_review"

COMPARISON_FILE = (
    STEP9E_DIR / "step9e_candidate_comparison.csv"
)

OUTPUT_DIR = PROJECT_DIR / "step9f_model_freeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. FROZEN PROJECT CONFIGURATION
# ------------------------------------------------------------

CLASS_LABELS = [
    "Normal",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal",
]

PRIMARY_CANDIDATE = {
    "model_type": "Logistic Regression",
    "feature_set": "Full_enriched",
    "number_of_features": 58,
    "selection_basis": (
        "Highest pooled development OOF macro-F1 "
        "among the reviewed candidates."
    ),
}

COMPARATOR_CANDIDATE = {
    "model_type": "XGBoost",
    "feature_set": "Baseline",
    "number_of_features": 9,
    "selection_basis": (
        "Retained as a simpler baseline comparator."
    ),
}

CV_CONFIGURATION = {
    "evaluation_method": "Grouped 5-fold cross-validation",
    "development_groups": 38,
    "development_windows": 3544,
    "group_leakage_check": "Passed in previous development evaluation",
}

TEST_SET_POLICY = {
    "final_test_set_used_for_candidate_selection": False,
    "final_test_set_used_in_step9f": False,
    "final_test_set_reserved_for": (
        "Step 10 final evaluation"
    ),
}


# ------------------------------------------------------------
# 3. CHECK STEP 9E RESULTS EXIST
# ------------------------------------------------------------

print("=" * 75)
print("STEP 9F — FREEZE SELECTED MODEL CANDIDATE")
print("=" * 75)

print("\nReading Step 9E candidate comparison:")
print(COMPARISON_FILE)

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        "\nStep 9E comparison file not found:\n"
        f"{COMPARISON_FILE}\n\n"
        "Please complete Step 9E first."
    )

comparison_df = pd.read_csv(COMPARISON_FILE)

print("\nStep 9E comparison:")
print(comparison_df.to_string(index=False))


# ------------------------------------------------------------
# 4. VERIFY PRIMARY CANDIDATE
# ------------------------------------------------------------

primary_rows = comparison_df[
    comparison_df["candidate"].astype(str).str.contains(
        "Full enriched Logistic Regression",
        case=False,
        na=False,
    )
]

if len(primary_rows) != 1:
    raise ValueError(
        "\nCould not identify exactly one Full enriched "
        "Logistic Regression candidate in the Step 9E results."
    )

primary_result = primary_rows.iloc[0]


# ------------------------------------------------------------
# 5. VERIFY COMPARATOR
# ------------------------------------------------------------

comparator_rows = comparison_df[
    comparison_df["candidate"].astype(str).str.contains(
        "Baseline XGBoost",
        case=False,
        na=False,
    )
]

if len(comparator_rows) != 1:
    raise ValueError(
        "\nCould not identify exactly one Baseline XGBoost "
        "candidate in the Step 9E results."
    )

comparator_result = comparator_rows.iloc[0]


# ------------------------------------------------------------
# 6. DISPLAY FROZEN CANDIDATE RESULTS
# ------------------------------------------------------------

print("\n" + "-" * 75)
print("PRIMARY CANDIDATE — FROZEN")
print("-" * 75)

print(f"Model: {PRIMARY_CANDIDATE['model_type']}")
print(f"Feature set: {PRIMARY_CANDIDATE['feature_set']}")
print(f"Number of features: {PRIMARY_CANDIDATE['number_of_features']}")

print("\nStep 9E development OOF metrics:")

for metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]:
    if metric in primary_result.index:
        print(f"{metric:20s}: {primary_result[metric]:.4f}")


print("\n" + "-" * 75)
print("COMPARATOR — RETAINED")
print("-" * 75)

print(f"Model: {COMPARATOR_CANDIDATE['model_type']}")
print(f"Feature set: {COMPARATOR_CANDIDATE['feature_set']}")
print(f"Number of features: {COMPARATOR_CANDIDATE['number_of_features']}")

print("\nStep 9E development OOF metrics:")

for metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]:
    if metric in comparator_result.index:
        print(f"{metric:20s}: {comparator_result[metric]:.4f}")


# ------------------------------------------------------------
# 7. CREATE FREEZE RECORD
# ------------------------------------------------------------

freeze_record = {
    "step": "9F",
    "title": "Selected model candidate freeze",
    "freeze_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    "primary_candidate": {
        **PRIMARY_CANDIDATE,
        "step9e_oof_metrics": {
            "accuracy": float(primary_result["accuracy"]),
            "balanced_accuracy": float(
                primary_result["balanced_accuracy"]
            ),
            "macro_f1": float(primary_result["macro_f1"]),
            "weighted_f1": float(primary_result["weighted_f1"]),
        },
    },

    "comparator_candidate": {
        **COMPARATOR_CANDIDATE,
        "step9e_oof_metrics": {
            "accuracy": float(comparator_result["accuracy"]),
            "balanced_accuracy": float(
                comparator_result["balanced_accuracy"]
            ),
            "macro_f1": float(comparator_result["macro_f1"]),
            "weighted_f1": float(comparator_result["weighted_f1"]),
        },
    },

    "class_labels": CLASS_LABELS,
    "cross_validation": CV_CONFIGURATION,
    "test_set_policy": TEST_SET_POLICY,

    "interpretation": (
        "The primary candidate was selected using development "
        "out-of-fold results. These results are not final test "
        "performance and do not establish clinical validity."
    ),
}


# ------------------------------------------------------------
# 8. SAVE JSON CONFIGURATION
# ------------------------------------------------------------

json_file = OUTPUT_DIR / "step9f_frozen_model_config.json"

with open(
    json_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        freeze_record,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# 9. SAVE HUMAN-READABLE REPORT
# ------------------------------------------------------------

report_file = OUTPUT_DIR / "step9f_model_freeze_report.txt"

with open(
    report_file,
    "w",
    encoding="utf-8",
) as file:

    file.write("STEP 9F — MODEL CANDIDATE FREEZE\n")
    file.write("=" * 75 + "\n\n")

    file.write("PRIMARY CANDIDATE\n")
    file.write("-" * 75 + "\n")
    file.write(f"Model: {PRIMARY_CANDIDATE['model_type']}\n")
    file.write(f"Feature set: {PRIMARY_CANDIDATE['feature_set']}\n")
    file.write(
        f"Number of features: "
        f"{PRIMARY_CANDIDATE['number_of_features']}\n"
    )

    file.write("\nStep 9E OOF metrics:\n")
    file.write(
        f"Accuracy: {primary_result['accuracy']:.4f}\n"
    )
    file.write(
        f"Balanced accuracy: "
        f"{primary_result['balanced_accuracy']:.4f}\n"
    )
    file.write(
        f"Macro-F1: {primary_result['macro_f1']:.4f}\n"
    )
    file.write(
        f"Weighted-F1: {primary_result['weighted_f1']:.4f}\n"
    )

    file.write("\n\nCOMPARATOR CANDIDATE\n")
    file.write("-" * 75 + "\n")
    file.write(f"Model: {COMPARATOR_CANDIDATE['model_type']}\n")
    file.write(f"Feature set: {COMPARATOR_CANDIDATE['feature_set']}\n")
    file.write(
        f"Number of features: "
        f"{COMPARATOR_CANDIDATE['number_of_features']}\n"
    )

    file.write("\nStep 9E OOF metrics:\n")
    file.write(
        f"Accuracy: {comparator_result['accuracy']:.4f}\n"
    )
    file.write(
        f"Balanced accuracy: "
        f"{comparator_result['balanced_accuracy']:.4f}\n"
    )
    file.write(
        f"Macro-F1: {comparator_result['macro_f1']:.4f}\n"
    )
    file.write(
        f"Weighted-F1: {comparator_result['weighted_f1']:.4f}\n"
    )

    file.write("\n\nEVALUATION POLICY\n")
    file.write("-" * 75 + "\n")
    file.write(
        "Candidate selection used development OOF predictions only.\n"
    )
    file.write(
        "The final test set was not loaded or evaluated in Step 9F.\n"
    )
    file.write(
        "The final test set is reserved for Step 10.\n"
    )

    file.write("\n\nLIMITATIONS\n")
    file.write("-" * 75 + "\n")
    file.write(
        "This is an experimental ECG screening prototype.\n"
    )
    file.write(
        "The model outputs engineering screening labels, "
        "not definitive clinical diagnoses.\n"
    )
    file.write(
        "Development OOF metrics do not establish clinical validity.\n"
    )


# ------------------------------------------------------------
# 10. SAVE A SIMPLE CSV SUMMARY
# ------------------------------------------------------------

summary_rows = [
    {
        "role": "Primary candidate",
        "model": PRIMARY_CANDIDATE["model_type"],
        "feature_set": PRIMARY_CANDIDATE["feature_set"],
        "n_features": PRIMARY_CANDIDATE["number_of_features"],
        "accuracy": primary_result["accuracy"],
        "balanced_accuracy": primary_result["balanced_accuracy"],
        "macro_f1": primary_result["macro_f1"],
        "weighted_f1": primary_result["weighted_f1"],
    },
    {
        "role": "Comparator",
        "model": COMPARATOR_CANDIDATE["model_type"],
        "feature_set": COMPARATOR_CANDIDATE["feature_set"],
        "n_features": COMPARATOR_CANDIDATE["number_of_features"],
        "accuracy": comparator_result["accuracy"],
        "balanced_accuracy": comparator_result["balanced_accuracy"],
        "macro_f1": comparator_result["macro_f1"],
        "weighted_f1": comparator_result["weighted_f1"],
    },
]

summary_df = pd.DataFrame(summary_rows)

summary_file = OUTPUT_DIR / "step9f_frozen_candidate_summary.csv"
summary_df.to_csv(summary_file, index=False)


# ------------------------------------------------------------
# 11. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 9F COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFrozen model configuration:")
print(json_file)

print("\nHuman-readable freeze report:")
print(report_file)

print("\nCSV summary:")
print(summary_file)

print("\nFinal test set was NOT loaded or evaluated.")
print("The selected candidate is now recorded for Step 10.")