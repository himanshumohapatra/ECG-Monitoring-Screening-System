# ============================================================
# STEP 9E — OOF CANDIDATE REVIEW
# Development OOF predictions only.
# The final test set is NOT loaded or evaluated.
# ============================================================

from pathlib import Path
import warnings

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

INPUT_FILE = (
    PROJECT_DIR
    / "step9d_feature_ablation"
    / "step9d_oof_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "step9e_candidate_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. STANDARD CLASS LABELS
# ------------------------------------------------------------

EXPECTED_LABELS = [
    "Normal",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal",
]


# Normalize spelling, capitalization, and common variants.
LABEL_MAP = {
    "normal": "Normal",
    "normal heartbeat": "Normal",

    "arrhythmia": "Arrhythmia",

    "tachycardia": "Tachycardia",

    "bradycardia": "Bradycardia",

    "other abnormal": "Other abnormal",
    "other abnormal ecg": "Other abnormal",
}


# ------------------------------------------------------------
# 3. CANDIDATES
# ------------------------------------------------------------

CANDIDATES = [
    {
        "candidate_name": "Primary — Full enriched Logistic Regression",
        "model_aliases": [
            "Logistic Regression",
            "LogisticRegression",
            "Logistic",
        ],
        "feature_set_aliases": [
            "Full enriched",
            "full_enriched",
            "Full Enriched",
        ],
    },
    {
        "candidate_name": "Comparator — Baseline XGBoost",
        "model_aliases": [
            "XGBoost",
            "XGB",
            "XGBClassifier",
        ],
        "feature_set_aliases": [
            "Baseline",
            "baseline",
            "Baseline features",
        ],
    },
]


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def normalize_text(value):
    """Normalize text for matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )


def normalize_labels(series, column_name):
    """Convert CSV label variants into standard class labels."""

    original = series.astype(str).str.strip()
    lookup = {
        normalize_text(key): value
        for key, value in LABEL_MAP.items()
    }

    normalized = original.map(
        lambda value: lookup.get(normalize_text(value))
    )

    if normalized.isna().any():
        unknown = sorted(
            original[normalized.isna()].unique().tolist()
        )

        raise ValueError(
            f"\nUnmapped labels in '{column_name}': {unknown}\n"
            f"Recognized labels: {list(LABEL_MAP.keys())}"
        )

    return normalized


def find_matching_value(series, aliases):
    """Find the actual CSV value matching one of the aliases."""

    available_values = (
        series.dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    normalized_aliases = {
        normalize_text(alias)
        for alias in aliases
    }

    for value in available_values:
        if normalize_text(value) in normalized_aliases:
            return value

    return None


def calculate_metrics(y_true, y_pred):
    """Calculate overall classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=EXPECTED_LABELS,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=EXPECTED_LABELS,
            average="weighted",
            zero_division=0,
        ),
    }


# ------------------------------------------------------------
# 5. LOAD OOF PREDICTIONS
# ------------------------------------------------------------

print("=" * 75)
print("STEP 9E — OOF CANDIDATE REVIEW")
print("=" * 75)

print("\nReading development OOF predictions:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


# ------------------------------------------------------------
# 6. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "group_id",
    "cv_fold",
    "preliminary_class",
    "feature_set",
    "model",
    "oof_prediction",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

model_col = "model"
feature_col = "feature_set"
actual_col = "preliminary_class"
predicted_col = "oof_prediction"

print("\nColumns selected:")
print(f"Model column: {model_col}")
print(f"Feature-set column: {feature_col}")
print(f"Actual-label column: {actual_col}")
print(f"Predicted-label column: {predicted_col}")


# ------------------------------------------------------------
# 7. NORMALIZE LABELS
# ------------------------------------------------------------

print("\nNormalizing actual and predicted class labels...")

df[actual_col] = normalize_labels(
    df[actual_col],
    actual_col,
)

df[predicted_col] = normalize_labels(
    df[predicted_col],
    predicted_col,
)

print("\nActual labels:")
print(sorted(df[actual_col].unique()))

print("\nPredicted labels:")
print(sorted(df[predicted_col].unique()))

if sorted(df[actual_col].unique()) != sorted(EXPECTED_LABELS):
    raise ValueError(
        "\nActual labels do not match the expected five classes."
    )

if sorted(df[predicted_col].unique()) != sorted(EXPECTED_LABELS):
    raise ValueError(
        "\nPredicted labels do not match the expected five classes."
    )


# ------------------------------------------------------------
# 8. VALIDATE OOF STRUCTURE
# ------------------------------------------------------------

if df[
    [
        "group_id",
        "cv_fold",
        actual_col,
        predicted_col,
    ]
].isna().any().any():

    raise ValueError(
        "\nMissing values detected in required OOF fields."
    )

print("\nOOF validation passed.")
print(f"Total OOF rows: {len(df):,}")
print(f"Unique groups: {df['group_id'].nunique()}")
print(f"CV folds: {sorted(df['cv_fold'].unique())}")


# ------------------------------------------------------------
# 9. REVIEW CANDIDATES
# ------------------------------------------------------------

all_metrics = []

for candidate in CANDIDATES:

    candidate_name = candidate["candidate_name"]

    print("\n" + "-" * 75)
    print(candidate_name)
    print("-" * 75)

    matched_model = find_matching_value(
        df[model_col],
        candidate["model_aliases"],
    )

    matched_feature_set = find_matching_value(
        df[feature_col],
        candidate["feature_set_aliases"],
    )

    if matched_model is None:
        print("\nAvailable model names:")
        print(sorted(df[model_col].astype(str).unique()))

        raise ValueError(
            f"\nCould not match model aliases: "
            f"{candidate['model_aliases']}"
        )

    if matched_feature_set is None:
        print("\nAvailable feature-set names:")
        print(sorted(df[feature_col].astype(str).unique()))

        raise ValueError(
            f"\nCould not match feature-set aliases: "
            f"{candidate['feature_set_aliases']}"
        )

    candidate_df = df[
        (df[model_col].astype(str) == matched_model)
        &
        (df[feature_col].astype(str) == matched_feature_set)
    ].copy()

    if candidate_df.empty:
        raise ValueError(
            f"\nNo OOF predictions found for {candidate_name}."
        )

    print(f"Matched model: {matched_model}")
    print(f"Matched feature set: {matched_feature_set}")
    print(f"OOF rows: {len(candidate_df):,}")

    y_true = candidate_df[actual_col]
    y_pred = candidate_df[predicted_col]

    metrics = calculate_metrics(y_true, y_pred)

    # Determine feature count from candidate identity.
    if "Full enriched" in candidate_name:
        n_features = 58
    else:
        n_features = 9

    metrics_row = {
        "candidate": candidate_name,
        "model": matched_model,
        "feature_set": matched_feature_set,
        "n_features": n_features,
        "n_oof_rows": len(candidate_df),
        **metrics,
    }

    all_metrics.append(metrics_row)

    print("\nOverall OOF metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:20s}: {metric_value:.4f}")

    # Classification report
    report_text = classification_report(
        y_true,
        y_pred,
        labels=EXPECTED_LABELS,
        target_names=EXPECTED_LABELS,
        zero_division=0,
    )

    print("\nPer-class classification report:")
    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=EXPECTED_LABELS,
        target_names=EXPECTED_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    safe_name = (
        candidate_name
        .replace(" ", "_")
        .replace("—", "-")
        .replace(":", "")
    )

    report_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=EXPECTED_LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=EXPECTED_LABELS,
        columns=EXPECTED_LABELS,
    )

    cm_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
    )

    # Save candidate-specific OOF predictions
    candidate_df[
        [
            "group_id",
            "cv_fold",
            actual_col,
            predicted_col,
        ]
    ].to_csv(
        OUTPUT_DIR / f"{safe_name}_oof_predictions.csv",
        index=False,
    )


# ------------------------------------------------------------
# 10. SAVE CANDIDATE COMPARISON
# ------------------------------------------------------------

comparison_df = pd.DataFrame(all_metrics)

comparison_file = (
    OUTPUT_DIR / "step9e_candidate_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False,
)

display_columns = [
    "candidate",
    "n_features",
    "n_oof_rows",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

print("\n" + "=" * 75)
print("CANDIDATE COMPARISON")
print("=" * 75)

print(
    comparison_df[display_columns]
    .sort_values("macro_f1", ascending=False)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 11. PRELIMINARY RANKING
# ------------------------------------------------------------

best_candidate = comparison_df.sort_values(
    "macro_f1",
    ascending=False,
).iloc[0]

print("\n" + "=" * 75)
print("PRELIMINARY CANDIDATE RANKING")
print("=" * 75)

print(
    "Best candidate by pooled OOF macro-F1: "
    f"{best_candidate['candidate']}"
)

print(f"Macro-F1: {best_candidate['macro_f1']:.4f}")

print(
    "Balanced accuracy: "
    f"{best_candidate['balanced_accuracy']:.4f}"
)

print(
    "\nImportant: this is development OOF evaluation only. "
    "It is not final test-set evaluation or clinical validation."
)


# ------------------------------------------------------------
# 12. SAVE RUN REPORT
# ------------------------------------------------------------

run_report_file = (
    OUTPUT_DIR / "step9e_review_report.txt"
)

with open(
    run_report_file,
    "w",
    encoding="utf-8",
) as file:

    file.write("STEP 9E — OOF CANDIDATE REVIEW\n")
    file.write("=" * 75 + "\n\n")

    file.write(f"Input file: {INPUT_FILE}\n")
    file.write(f"Input shape: {df.shape}\n")
    file.write(f"OOF rows: {len(df):,}\n")
    file.write(f"Unique groups: {df['group_id'].nunique()}\n")
    file.write(f"CV folds: {sorted(df['cv_fold'].unique())}\n\n")

    file.write("Candidate comparison:\n")

    file.write(
        comparison_df[display_columns]
        .sort_values("macro_f1", ascending=False)
        .to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "Best candidate by pooled OOF macro-F1: "
        f"{best_candidate['candidate']}\n"
    )

    file.write(
        "\nThe final test set was not loaded or evaluated.\n"
    )

print("\nSaved candidate comparison:")
print(comparison_file)

print("\nSaved review report:")
print(run_report_file)

print("\nSTEP 9E COMPLETED SUCCESSFULLY.")