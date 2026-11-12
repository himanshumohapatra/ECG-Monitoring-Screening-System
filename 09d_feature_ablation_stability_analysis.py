# ============================================================
# STEP 9D — FEATURE ABLATION AND STABILITY ANALYSIS
#
# Feature sets:
#   1. Baseline
#   2. Baseline + morphology measurements
#   3. Baseline + availability/count features
#   4. Full enriched
#
# Models:
#   Logistic Regression, Random Forest, SVM RBF, XGBoost, MLP
#
# Evaluation:
#   Existing 5 grouped CV folds
#   Out-of-fold metrics
#   Per-class recall
#   Fold stability
#
# IMPORTANT:
#   - Development dataset only
#   - Final test set is NOT loaded
#   - XGBoost labels are encoded inside each training fold
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
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
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

INPUT_CSV = (
    PROJECT_DIR
    / "step9b_morphology_features"
    / "five_class_development_morphology_enriched.csv"
)

OUTPUT_DIR = PROJECT_DIR / "step9d_feature_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TARGET AND GROUPING COLUMNS
# ============================================================

TARGET_COL = "preliminary_class"
FOLD_COL = "cv_fold"
GROUP_COL = "group_id"

CLASS_ORDER = [
    "Normal heartbeat",
    "Arrhythmia",
    "Tachycardia",
    "Bradycardia",
    "Other abnormal ECG",
]


# ============================================================
# 3. BASELINE FEATURES — SAME 9 FEATURES AS STEP 9C
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


# ============================================================
# 4. MORPHOLOGY / RHYTHM MEASUREMENT FEATURES
# ============================================================

MORPHOLOGY_FEATURES = [
    # RR interval
    "RR_interval_ms_median",
    "RR_interval_ms_mean",
    "RR_interval_ms_std",
    "RR_interval_ms_min",
    "RR_interval_ms_max",

    # Heart rate
    "Heart_rate_bpm_median",
    "Heart_rate_bpm_mean",
    "Heart_rate_bpm_std",
    "Heart_rate_bpm_min",
    "Heart_rate_bpm_max",

    # R-wave amplitude
    "R_amplitude_mV_median",
    "R_amplitude_mV_mean",
    "R_amplitude_mV_std",

    # P-wave amplitude
    "P_amplitude_mV_median",
    "P_amplitude_mV_mean",
    "P_amplitude_mV_std",

    # P-wave duration
    "P_duration_ms_median",
    "P_duration_ms_mean",

    # QRS duration
    "QRS_duration_ms_median",
    "QRS_duration_ms_mean",

    # PR interval
    "PR_interval_ms_median",
    "PR_interval_ms_mean",

    # QT interval
    "QT_interval_ms_median",
    "QT_interval_ms_mean",

    # T-wave amplitude
    "T_amplitude_mV_median",
    "T_amplitude_mV_mean",

    # T-wave duration
    "T_duration_ms_median",
    "T_duration_ms_mean",
]


# ============================================================
# 5. AVAILABILITY / COUNT FEATURES
# ============================================================

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


# ============================================================
# 6. FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "Baseline": BASELINE_FEATURES,

    "Baseline_plus_morphology": (
        BASELINE_FEATURES + MORPHOLOGY_FEATURES
    ),

    "Baseline_plus_availability": (
        BASELINE_FEATURES + AVAILABILITY_FEATURES
    ),

    "Full_enriched": (
        BASELINE_FEATURES
        + MORPHOLOGY_FEATURES
        + AVAILABILITY_FEATURES
    ),
}


# ============================================================
# 7. LOAD DEVELOPMENT DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("STEP 9D — FEATURE ABLATION AND STABILITY ANALYSIS")
print("=" * 70)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_CSV}\n"
        "Please confirm Step 9B completed successfully."
    )

df = pd.read_csv(INPUT_CSV)

print(f"\nInput file: {INPUT_CSV}")
print(f"Development dataset shape: {df.shape}")

required_columns = [
    TARGET_COL,
    FOLD_COL,
    GROUP_COL,
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

all_expected_features = sorted(
    set(
        BASELINE_FEATURES
        + MORPHOLOGY_FEATURES
        + AVAILABILITY_FEATURES
    )
)

missing_features = [
    col for col in all_expected_features
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "\nSome expected Step 9B features are missing:\n"
        + "\n".join(missing_features)
    )

df = df.dropna(
    subset=[TARGET_COL, FOLD_COL, GROUP_COL]
).copy()

df[TARGET_COL] = df[TARGET_COL].astype(str)
df[FOLD_COL] = df[FOLD_COL].astype(int)

print(f"Rows after required-column validation: {len(df)}")
print(f"CV folds: {sorted(df[FOLD_COL].unique())}")
print(f"Groups: {df[GROUP_COL].nunique()}")

print("\nClass counts:")
print(df[TARGET_COL].value_counts())


# ============================================================
# 8. VALIDATE GROUPED CV FOLDS
# ============================================================

group_fold_counts = (
    df.groupby(GROUP_COL)[FOLD_COL]
    .nunique()
)

if (group_fold_counts > 1).any():
    bad_groups = group_fold_counts[
        group_fold_counts > 1
    ].index.tolist()

    raise ValueError(
        "Group leakage detected. These groups occur in "
        f"multiple folds: {bad_groups}"
    )

print("\nGroup leakage check: PASSED")


# ============================================================
# 9. MODEL DEFINITIONS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]),

    "SVM RBF": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ]),

    "MLP": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )),
    ]),
}


# ============================================================
# 10. RUN GROUPED CROSS-VALIDATION
# ============================================================

folds = sorted(df[FOLD_COL].unique())

overall_results = []
fold_results = []
class_recall_results = []
all_oof_predictions = []

total_runs = (
    len(FEATURE_SETS)
    * len(models)
    * len(folds)
)

run_number = 0


for feature_set_name, feature_columns in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(f"FEATURE SET: {feature_set_name}")
    print(f"Number of features: {len(feature_columns)}")
    print("=" * 70)

    X = df[feature_columns].copy()
    y = df[TARGET_COL].copy()

    for model_name, model_pipeline in models.items():

        print(f"\nModel: {model_name}")

        oof_predictions = pd.Series(
            index=df.index,
            dtype="object",
        )

        for fold in folds:

            run_number += 1

            train_mask = df[FOLD_COL] != fold
            valid_mask = df[FOLD_COL] == fold

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]

            X_valid = X.loc[valid_mask]
            y_valid = y.loc[valid_mask]

            groups_train = set(
                df.loc[train_mask, GROUP_COL]
            )

            groups_valid = set(
                df.loc[valid_mask, GROUP_COL]
            )

            overlap = groups_train.intersection(
                groups_valid
            )

            if overlap:
                raise ValueError(
                    f"Group leakage in fold {fold}: {overlap}"
                )

            fitted_model = clone(model_pipeline)

            # ------------------------------------------------
            # XGBOOST LABEL ENCODING FIX
            #
            # XGBoost requires integer labels starting from 0.
            # Encode labels using the training fold only.
            # Convert predictions back to original class names.
            # ------------------------------------------------

            if model_name == "XGBoost":

                label_encoder = LabelEncoder()

                y_train_encoded = label_encoder.fit_transform(
                    y_train
                )

                fitted_model.fit(
                    X_train,
                    y_train_encoded,
                )

                predictions_encoded = fitted_model.predict(
                    X_valid
                )

                predictions = label_encoder.inverse_transform(
                    predictions_encoded.astype(int)
                )

            else:

                fitted_model.fit(
                    X_train,
                    y_train,
                )

                predictions = fitted_model.predict(
                    X_valid
                )

            # ------------------------------------------------
            # Store out-of-fold predictions
            # ------------------------------------------------

            oof_predictions.loc[
                valid_mask
            ] = predictions

            # ------------------------------------------------
            # Fold metrics
            # ------------------------------------------------

            fold_accuracy = accuracy_score(
                y_valid,
                predictions,
            )

            fold_balanced_accuracy = balanced_accuracy_score(
                y_valid,
                predictions,
            )

            fold_macro_f1 = f1_score(
                y_valid,
                predictions,
                average="macro",
                zero_division=0,
            )

            fold_weighted_f1 = f1_score(
                y_valid,
                predictions,
                average="weighted",
                zero_division=0,
            )

            fold_results.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "fold": fold,
                "n_features": len(feature_columns),
                "n_train": len(y_train),
                "n_valid": len(y_valid),
                "accuracy": fold_accuracy,
                "balanced_accuracy": fold_balanced_accuracy,
                "macro_f1": fold_macro_f1,
                "weighted_f1": fold_weighted_f1,
            })

            print(
                f"  Fold {fold}: "
                f"Macro-F1={fold_macro_f1:.4f}, "
                f"Balanced Acc={fold_balanced_accuracy:.4f}"
            )

            print(
                f"  Progress: {run_number}/{total_runs}"
            )

        # ====================================================
        # OVERALL OUT-OF-FOLD METRICS
        # ====================================================

        valid_oof_mask = oof_predictions.notna()

        y_true_oof = y.loc[valid_oof_mask]
        y_pred_oof = oof_predictions.loc[valid_oof_mask]

        overall_accuracy = accuracy_score(
            y_true_oof,
            y_pred_oof,
        )

        overall_balanced_accuracy = balanced_accuracy_score(
            y_true_oof,
            y_pred_oof,
        )

        overall_macro_f1 = f1_score(
            y_true_oof,
            y_pred_oof,
            average="macro",
            zero_division=0,
        )

        overall_weighted_f1 = f1_score(
            y_true_oof,
            y_pred_oof,
            average="weighted",
            zero_division=0,
        )

        overall_results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "n_features": len(feature_columns),
            "accuracy": overall_accuracy,
            "balanced_accuracy": overall_balanced_accuracy,
            "macro_f1": overall_macro_f1,
            "weighted_f1": overall_weighted_f1,
        })

        # ====================================================
        # PER-CLASS RECALL
        # ====================================================

        class_recalls = recall_score(
            y_true_oof,
            y_pred_oof,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        for class_name, class_recall in zip(
            CLASS_ORDER,
            class_recalls,
        ):
            class_recall_results.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "class": class_name,
                "recall": class_recall,
            })

        # ====================================================
        # SAVE OOF PREDICTIONS
        # ====================================================

        prediction_output = df[
            [
                GROUP_COL,
                FOLD_COL,
                TARGET_COL,
        ]].copy()

        prediction_output["feature_set"] = feature_set_name
        prediction_output["model"] = model_name
        prediction_output["oof_prediction"] = oof_predictions

        all_oof_predictions.append(
            prediction_output.reset_index(drop=True)
        )


# ============================================================
# 11. CONVERT RESULTS TO DATAFRAMES
# ============================================================

overall_df = pd.DataFrame(overall_results)
fold_df = pd.DataFrame(fold_results)
class_recall_df = pd.DataFrame(class_recall_results)

oof_df = pd.concat(
    all_oof_predictions,
    ignore_index=True,
)


# ============================================================
# 12. SAVE PRIMARY RESULTS
# ============================================================

overall_path = (
    OUTPUT_DIR / "step9d_overall_oof_results.csv"
)

fold_path = (
    OUTPUT_DIR / "step9d_per_fold_results.csv"
)

recall_path = (
    OUTPUT_DIR / "step9d_per_class_recall.csv"
)

oof_path = (
    OUTPUT_DIR / "step9d_oof_predictions.csv"
)

overall_df.to_csv(
    overall_path,
    index=False,
)

fold_df.to_csv(
    fold_path,
    index=False,
)

class_recall_df.to_csv(
    recall_path,
    index=False,
)

oof_df.to_csv(
    oof_path,
    index=False,
)


# ============================================================
# 13. COMPARE EACH FEATURE SET AGAINST BASELINE
# ============================================================

baseline_metrics = overall_df[
    overall_df["feature_set"] == "Baseline"
].copy()

baseline_metrics = baseline_metrics[
    [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
].rename(columns={
    "accuracy": "baseline_accuracy",
    "balanced_accuracy": "baseline_balanced_accuracy",
    "macro_f1": "baseline_macro_f1",
    "weighted_f1": "baseline_weighted_f1",
})

comparison_df = overall_df.merge(
    baseline_metrics,
    on="model",
    how="left",
)

comparison_df["delta_accuracy"] = (
    comparison_df["accuracy"]
    - comparison_df["baseline_accuracy"]
)

comparison_df["delta_balanced_accuracy"] = (
    comparison_df["balanced_accuracy"]
    - comparison_df["baseline_balanced_accuracy"]
)

comparison_df["delta_macro_f1"] = (
    comparison_df["macro_f1"]
    - comparison_df["baseline_macro_f1"]
)

comparison_df["delta_weighted_f1"] = (
    comparison_df["weighted_f1"]
    - comparison_df["baseline_weighted_f1"]
)

comparison_path = (
    OUTPUT_DIR / "step9d_comparison_vs_baseline.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# 14. FOLD STABILITY SUMMARY
# ============================================================

stability_df = (
    fold_df
    .groupby(
        ["feature_set", "model"],
        as_index=False,
    )
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),

        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean",
        ),

        balanced_accuracy_std=(
            "balanced_accuracy",
            "std",
        ),

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
)

stability_path = (
    OUTPUT_DIR / "step9d_fold_stability_summary.csv"
)

stability_df.to_csv(
    stability_path,
    index=False,
)


# ============================================================
# 15. SAVE FEATURE SET DEFINITIONS
# ============================================================

feature_list_path = (
    OUTPUT_DIR / "step9d_feature_set_definitions.txt"
)

with open(
    feature_list_path,
    "w",
    encoding="utf-8",
) as f:

    for feature_set_name, feature_columns in FEATURE_SETS.items():

        f.write(f"\n{feature_set_name}\n")
        f.write(
            f"Number of features: {len(feature_columns)}\n"
        )
        f.write("-" * 60 + "\n")

        for feature in feature_columns:
            f.write(feature + "\n")


# ============================================================
# 16. SAVE RUN REPORT
# ============================================================

report_path = (
    OUTPUT_DIR / "step9d_run_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "STEP 9D — FEATURE ABLATION AND STABILITY ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(f"Input file: {INPUT_CSV}\n")
    f.write(f"Development rows: {len(df)}\n")
    f.write(f"Groups: {df[GROUP_COL].nunique()}\n")
    f.write(f"Folds: {folds}\n")
    f.write("Final test set loaded: NO\n")
    f.write("Group leakage check: PASSED\n\n")

    f.write("Feature sets:\n")

    for name, features in FEATURE_SETS.items():
        f.write(
            f"  {name}: {len(features)} features\n"
        )

    f.write("\n\nOverall OOF results:\n")
    f.write(overall_df.to_string(index=False))

    f.write("\n\nFold stability summary:\n")
    f.write(stability_df.to_string(index=False))

    f.write("\n\nChanges compared with baseline:\n")
    f.write(comparison_df.to_string(index=False))


# ============================================================
# 17. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9D COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOverall OOF results:")
print(
    overall_df.sort_values(
        by="macro_f1",
        ascending=False,
    ).to_string(index=False)
)

print("\nChanges compared with baseline:")
print(
    comparison_df[
        [
            "feature_set",
            "model",
            "macro_f1",
            "delta_macro_f1",
            "balanced_accuracy",
            "delta_balanced_accuracy",
        ]
    ].sort_values(
        by="delta_macro_f1",
        ascending=False,
    ).to_string(index=False)
)

print("\nFold stability summary:")
print(
    stability_df.sort_values(
        by="macro_f1_mean",
        ascending=False,
    ).to_string(index=False)
)

print("\nSaved files:")
print(overall_path)
print(fold_path)
print(recall_path)
print(oof_path)
print(comparison_path)
print(stability_path)
print(feature_list_path)
print(report_path)

print("\nFinal test set was NOT loaded or evaluated.")