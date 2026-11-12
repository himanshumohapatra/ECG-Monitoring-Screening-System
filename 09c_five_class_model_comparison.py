# ============================================================
# 09c_five_class_model_comparison.py
#
# Step 9C:
# Compare baseline rate/quality features against morphology-
# enriched features using the EXISTING grouped CV folds.
#
# IMPORTANT:
# - Uses development data only.
# - Does NOT load or evaluate the final test set.
# - Preserves the existing cv_fold assignments.
# - Fits imputation and scaling separately inside each fold.
# - Checks for group leakage between train and validation.
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\Himanshu Mohapatra\OneDrive\Desktop\Project_Main"
)

INPUT_FILE = (
    PROJECT_DIR
    / "step9b_morphology_features"
    / "five_class_development_morphology_enriched.csv"
)

OUTPUT_DIR = PROJECT_DIR / "step9c_model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "preliminary_class"
FOLD_COLUMN = "cv_fold"
GROUP_COLUMN = "group_id"

RANDOM_STATE = 42
EXPECTED_FOLDS = [1, 2, 3, 4, 5]

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*X does not have valid feature names.*",
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ------------------------------------------------------------
# 2. BASELINE FEATURES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. COLUMNS THAT MUST NOT BE USED AS MODEL FEATURES
# ------------------------------------------------------------

EXCLUDED_COLUMNS = {
    # Target and CV/group identifiers
    TARGET_COLUMN,
    FOLD_COLUMN,
    GROUP_COLUMN,

    # Record/window identifiers and metadata
    "record_id",
    "channel",
    "window_index",
    "window_start_s",
    "window_end_s",

    # Target-construction fractions / labels
    "normal_fraction",
    "arrhythmia_fraction",
    "tachy_fraction",
    "brady_fraction",
    "other_abnormal_fraction",
    "dominant_label",
    "dominant_label_fraction",
}


# ------------------------------------------------------------
# 4. LOAD DEVELOPMENT DATA ONLY
# ------------------------------------------------------------

print("=" * 70)
print("STEP 9C: BASELINE vs MORPHOLOGY-ENRICHED MODEL COMPARISON")
print("=" * 70)

print("\nLoading development dataset:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find Step 9B development file:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)
print("Available columns:", len(df.columns))

required_columns = [
    TARGET_COLUMN,
    FOLD_COLUMN,
    GROUP_COLUMN,
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "Required columns are missing: "
        f"{missing_required}"
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError(
        f"Missing values found in target column: {TARGET_COLUMN}"
    )

if df[FOLD_COLUMN].isna().any():
    raise ValueError(
        f"Missing values found in CV fold column: {FOLD_COLUMN}"
    )

if df[GROUP_COLUMN].isna().any():
    raise ValueError(
        f"Missing values found in group column: {GROUP_COLUMN}"
    )

actual_folds = sorted(df[FOLD_COLUMN].unique().tolist())

print("\nExisting CV folds:", actual_folds)

if actual_folds != EXPECTED_FOLDS:
    raise ValueError(
        "Unexpected CV fold assignments.\n"
        f"Expected: {EXPECTED_FOLDS}\n"
        f"Found:    {actual_folds}"
    )

print("\nTarget class counts:")
print(df[TARGET_COLUMN].value_counts().sort_index())


# ------------------------------------------------------------
# 5. VALIDATE BASELINE FEATURE COLUMNS
# ------------------------------------------------------------

missing_baseline = [
    col for col in BASELINE_FEATURES
    if col not in df.columns
]

if missing_baseline:
    raise ValueError(
        "The following baseline features are missing:\n"
        f"{missing_baseline}"
    )


# ------------------------------------------------------------
# 6. BUILD MORPHOLOGY-ENRICHED FEATURE SET
# ------------------------------------------------------------

# Use numeric columns only, excluding target, metadata, and
# target-construction columns.
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

morphology_candidate_features = [
    col for col in numeric_columns
    if col not in EXCLUDED_COLUMNS
    and col not in BASELINE_FEATURES
]

# The enriched set includes the baseline features plus all
# eligible numeric morphology/beat-aggregation features.
ENRICHED_FEATURES = (
    BASELINE_FEATURES
    + morphology_candidate_features
)

if len(ENRICHED_FEATURES) <= len(BASELINE_FEATURES):
    raise ValueError(
        "No additional morphology features were found. "
        "Please check the Step 9B output file."
    )

print("\n" + "-" * 70)
print("FEATURE SETS")
print("-" * 70)

print(f"\nBaseline feature count: {len(BASELINE_FEATURES)}")
for feature in BASELINE_FEATURES:
    print("  -", feature)

print(
    f"\nMorphology-enriched feature count: "
    f"{len(ENRICHED_FEATURES)}"
)

print(
    f"Additional morphology/aggregation features: "
    f"{len(morphology_candidate_features)}"
)

for feature in morphology_candidate_features:
    print("  -", feature)

# Save feature lists for reproducibility.
feature_list_path = OUTPUT_DIR / "step9c_feature_lists.json"

with open(feature_list_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "baseline_features": BASELINE_FEATURES,
            "additional_morphology_features":
                morphology_candidate_features,
            "enriched_features": ENRICHED_FEATURES,
            "excluded_columns": sorted(EXCLUDED_COLUMNS),
        },
        f,
        indent=4,
    )

print("\nSaved feature lists:")
print(feature_list_path)


# ------------------------------------------------------------
# 7. ENCODE TARGET LABELS
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_text = df[TARGET_COLUMN].astype(str).to_numpy()
y = label_encoder.fit_transform(y_text)

class_names = label_encoder.classes_.tolist()
number_of_classes = len(class_names)

print("\nEncoded classes:")
for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")

if number_of_classes != 5:
    raise ValueError(
        "Expected exactly five target classes, but found "
        f"{number_of_classes}: {class_names}"
    )


# ------------------------------------------------------------
# 8. STORE EXISTING FOLD ASSIGNMENTS
# ------------------------------------------------------------

fold_assignments = df[FOLD_COLUMN].astype(int).to_numpy()
groups = df[GROUP_COLUMN].astype(str).to_numpy()

print("\nFold sizes:")
for fold in EXPECTED_FOLDS:
    fold_mask = fold_assignments == fold

    print(
        f"  Fold {fold}: "
        f"{fold_mask.sum()} validation windows, "
        f"{len(np.unique(groups[fold_mask]))} validation groups"
    )


# ------------------------------------------------------------
# 9. DEFINE MODELS
# ------------------------------------------------------------

def create_models():
    """
    Create fresh model instances for each fold.

    Logistic Regression, SVM, and MLP use standardized features.
    Random Forest and XGBoost use imputed, unscaled features.
    """

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale": True,
            "use_sample_weights": False,
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
            "use_sample_weights": False,
        },

        "SVM RBF": {
            "model": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "scale": True,
            "use_sample_weights": False,
        },

        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=number_of_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
            "use_sample_weights": True,
        },

        "MLP": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            "scale": True,
            "use_sample_weights": False,
        },
    }

    return models


# ------------------------------------------------------------
# 10. CLASS WEIGHTS FOR XGBOOST
# ------------------------------------------------------------

def calculate_balanced_sample_weights(y_train):
    """
    Compute balanced sample weights using training-fold labels only.
    """

    classes, counts = np.unique(y_train, return_counts=True)

    total = len(y_train)
    number_of_present_classes = len(classes)

    class_weight_map = {
        class_id: total / (number_of_present_classes * count)
        for class_id, count in zip(classes, counts)
    }

    sample_weights = np.array(
        [class_weight_map[label] for label in y_train],
        dtype=float,
    )

    return sample_weights


# ------------------------------------------------------------
# 11. CHECK GROUP LEAKAGE FOR EXISTING FOLDS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CHECKING GROUP LEAKAGE")
print("-" * 70)

for fold in EXPECTED_FOLDS:
    train_mask = fold_assignments != fold
    val_mask = fold_assignments == fold

    train_groups = set(groups[train_mask])
    val_groups = set(groups[val_mask])

    overlap = train_groups.intersection(val_groups)

    if overlap:
        raise RuntimeError(
            f"Group leakage detected in fold {fold}: "
            f"{sorted(overlap)}"
        )

    print(
        f"Fold {fold}: PASS | "
        f"Train groups={len(train_groups)}, "
        f"Validation groups={len(val_groups)}"
    )


# ------------------------------------------------------------
# 12. RUN OUT-OF-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

EXPERIMENTS = {
    "Baseline": BASELINE_FEATURES,
    "Morphology-Enriched": ENRICHED_FEATURES,
}

all_oof_predictions = []
all_fold_metrics = []
all_confusion_matrices = []
all_classification_reports = []

overall_results = []

total_runs = (
    len(EXPERIMENTS)
    * 5
    * len(EXPECTED_FOLDS)
)

run_number = 0

for experiment_name, feature_columns in EXPERIMENTS.items():

    print("\n")
    print("=" * 70)
    print("EXPERIMENT:", experiment_name)
    print("Number of features:", len(feature_columns))
    print("=" * 70)

    X_all = df[feature_columns].copy()

    # Convert possible non-numeric values to NaN.
    X_all = X_all.apply(pd.to_numeric, errors="coerce")

    if X_all.isna().all(axis=0).any():
        all_missing = X_all.columns[X_all.isna().all(axis=0)].tolist()

        raise ValueError(
            "These features contain only missing values:\n"
            f"{all_missing}"
        )

    model_names = list(create_models().keys())

    # One OOF prediction per development row per model.
    oof_predictions = {
        model_name: np.full(len(df), -1, dtype=int)
        for model_name in model_names
    }

    oof_fold_seen = {
        model_name: np.zeros(len(df), dtype=int)
        for model_name in model_names
    }

    # --------------------------------------------------------
    # Fold loop: preserve the existing cv_fold assignments.
    # --------------------------------------------------------

    for fold in EXPECTED_FOLDS:

        train_mask = fold_assignments != fold
        val_mask = fold_assignments == fold

        X_train_raw = X_all.loc[train_mask].to_numpy(dtype=float)
        X_val_raw = X_all.loc[val_mask].to_numpy(dtype=float)

        y_train = y[train_mask]
        y_val = y[val_mask]

        train_groups = set(groups[train_mask])
        val_groups = set(groups[val_mask])

        if train_groups.intersection(val_groups):
            raise RuntimeError(
                f"Group leakage detected in fold {fold}."
            )

        # ----------------------------------------------------
        # Fit imputer ONLY on the training fold.
        # ----------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        X_train_imputed = imputer.fit_transform(X_train_raw)
        X_val_imputed = imputer.transform(X_val_raw)

        # Safety check: no NaNs should remain after imputation.
        if np.isnan(X_train_imputed).any():
            raise RuntimeError(
                f"NaN values remain in training data, fold {fold}."
            )

        if np.isnan(X_val_imputed).any():
            raise RuntimeError(
                f"NaN values remain in validation data, fold {fold}."
            )

        print(
            f"\nFold {fold}: "
            f"train={len(y_train)}, validation={len(y_val)}"
        )

        # ----------------------------------------------------
        # Fit and evaluate each model.
        # ----------------------------------------------------

        models = create_models()

        for model_name, model_info in models.items():

            run_number += 1

            model = model_info["model"]
            should_scale = model_info["scale"]
            use_sample_weights = model_info["use_sample_weights"]

            print(
                f"  [{run_number}/{total_runs}] "
                f"{experiment_name} | {model_name}"
            )

            X_train_model = X_train_imputed
            X_val_model = X_val_imputed

            # Fit scaler ONLY on training fold.
            if should_scale:
                scaler = StandardScaler()

                X_train_model = scaler.fit_transform(
                    X_train_imputed
                )

                X_val_model = scaler.transform(
                    X_val_imputed
                )

            # XGBoost receives balanced sample weights computed
            # from this training fold only.
            if use_sample_weights:
                sample_weights = calculate_balanced_sample_weights(
                    y_train
                )

                model.fit(
                    X_train_model,
                    y_train,
                    sample_weight=sample_weights,
                )
            else:
                model.fit(
                    X_train_model,
                    y_train,
                )

            y_pred = model.predict(X_val_model).astype(int)

            # Store OOF predictions at original row positions.
            validation_indices = np.where(val_mask)[0]

            oof_predictions[model_name][validation_indices] = y_pred
            oof_fold_seen[model_name][validation_indices] += 1

            # Fold metrics.
            fold_accuracy = accuracy_score(y_val, y_pred)

            fold_balanced_accuracy = balanced_accuracy_score(
                y_val,
                y_pred,
            )

            fold_macro_f1 = f1_score(
                y_val,
                y_pred,
                average="macro",
                zero_division=0,
            )

            fold_weighted_f1 = f1_score(
                y_val,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            all_fold_metrics.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(y_train),
                    "n_validation": len(y_val),
                    "n_train_groups": len(train_groups),
                    "n_validation_groups": len(val_groups),
                    "accuracy": fold_accuracy,
                    "balanced_accuracy": fold_balanced_accuracy,
                    "macro_f1": fold_macro_f1,
                    "weighted_f1": fold_weighted_f1,
                }
            )

            print(
                f"      Accuracy={fold_accuracy:.4f} | "
                f"Balanced Accuracy={fold_balanced_accuracy:.4f} | "
                f"Macro F1={fold_macro_f1:.4f}"
            )

    # --------------------------------------------------------
    # Validate OOF predictions: exactly one prediction per row.
    # --------------------------------------------------------

    for model_name in model_names:

        if not np.all(oof_fold_seen[model_name] == 1):
            bad_rows = np.where(
                oof_fold_seen[model_name] != 1
            )[0]

            raise RuntimeError(
                f"{model_name} did not receive exactly one OOF "
                f"prediction for every row. Problem rows: "
                f"{bad_rows[:20].tolist()}"
            )

        if np.any(oof_predictions[model_name] < 0):
            raise RuntimeError(
                f"Missing OOF predictions for {model_name}."
            )

    # --------------------------------------------------------
    # Overall OOF metrics for this experiment.
    # --------------------------------------------------------

    for model_name in model_names:

        y_pred_oof = oof_predictions[model_name]

        accuracy = accuracy_score(y, y_pred_oof)

        balanced_accuracy = balanced_accuracy_score(
            y,
            y_pred_oof,
        )

        macro_f1 = f1_score(
            y,
            y_pred_oof,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y,
            y_pred_oof,
            average="weighted",
            zero_division=0,
        )

        report_dict = classification_report(
            y,
            y_pred_oof,
            labels=np.arange(number_of_classes),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(
            y,
            y_pred_oof,
            labels=np.arange(number_of_classes),
        )

        overall_results.append(
            {
                "experiment": experiment_name,
                "model": model_name,
                "n_features": len(feature_columns),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

        # Save confusion matrix as a tidy CSV.
        cm_df = pd.DataFrame(
            cm,
            index=[
                f"Actual_{name}"
                for name in class_names
            ],
            columns=[
                f"Predicted_{name}"
                for name in class_names
            ],
        )

        cm_df.to_csv(
            OUTPUT_DIR
            / (
                f"confusion_matrix_"
                f"{experiment_name.replace('-', '_').replace(' ', '_')}_"
                f"{model_name.replace(' ', '_')}.csv"
            )
        )

        # Save classification report.
        report_df = pd.DataFrame(report_dict).transpose()

        report_df.to_csv(
            OUTPUT_DIR
            / (
                f"classification_report_"
                f"{experiment_name.replace('-', '_').replace(' ', '_')}_"
                f"{model_name.replace(' ', '_')}.csv"
            )
        )

        # Save OOF predictions.
        prediction_df = pd.DataFrame(
            {
                "row_index": np.arange(len(df)),
                "group_id": df[GROUP_COLUMN].astype(str),
                "cv_fold": fold_assignments,
                "actual_class": y_text,
                "predicted_class": label_encoder.inverse_transform(
                    y_pred_oof
                ),
            }
        )

        prediction_df["correct"] = (
            prediction_df["actual_class"]
            == prediction_df["predicted_class"]
        )

        prediction_df.to_csv(
            OUTPUT_DIR
            / (
                f"oof_predictions_"
                f"{experiment_name.replace('-', '_').replace(' ', '_')}_"
                f"{model_name.replace(' ', '_')}.csv"
            ),
            index=False,
        )

        # Store a compact per-class recall table.
        class_recall_rows = []

        for class_index, class_name in enumerate(class_names):

            class_recall = report_dict[class_name]["recall"]

            class_recall_rows.append(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "class": class_name,
                    "recall": class_recall,
                }
            )

        all_classification_reports.extend(class_recall_rows)

        print("\n" + "-" * 70)
        print(
            f"OVERALL OOF RESULTS: "
            f"{experiment_name} | {model_name}"
        )
        print("-" * 70)

        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
        print(f"Macro F1:          {macro_f1:.4f}")
        print(f"Weighted F1:       {weighted_f1:.4f}")

        print("\nPer-class recall:")
        for class_name in class_names:
            print(
                f"  {class_name}: "
                f"{report_dict[class_name]['recall']:.4f}"
            )

        print("\nConfusion matrix:")
        print(cm)


# ------------------------------------------------------------
# 13. SAVE SUMMARY RESULTS
# ------------------------------------------------------------

overall_results_df = pd.DataFrame(overall_results)

overall_results_df = overall_results_df.sort_values(
    by=["experiment", "macro_f1"],
    ascending=[True, False],
)

fold_metrics_df = pd.DataFrame(all_fold_metrics)

class_recall_df = pd.DataFrame(all_classification_reports)

overall_results_path = (
    OUTPUT_DIR / "step9c_overall_oof_model_comparison.csv"
)

fold_metrics_path = (
    OUTPUT_DIR / "step9c_per_fold_metrics.csv"
)

class_recall_path = (
    OUTPUT_DIR / "step9c_per_class_recall.csv"
)

overall_results_df.to_csv(
    overall_results_path,
    index=False,
)

fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False,
)

class_recall_df.to_csv(
    class_recall_path,
    index=False,
)


# ------------------------------------------------------------
# 14. PRINT BASELINE vs ENRICHED COMPARISON
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("STEP 9C FINAL COMPARISON")
print("=" * 70)

display_columns = [
    "experiment",
    "model",
    "n_features",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

print(
    overall_results_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\n" + "-" * 70)
print("MACRO F1 COMPARISON: BASELINE vs MORPHOLOGY-ENRICHED")
print("-" * 70)

comparison_table = overall_results_df.pivot(
    index="model",
    columns="experiment",
    values="macro_f1",
)

if (
    "Baseline" in comparison_table.columns
    and "Morphology-Enriched" in comparison_table.columns
):

    comparison_table["macro_f1_change"] = (
        comparison_table["Morphology-Enriched"]
        - comparison_table["Baseline"]
    )

    comparison_table = comparison_table.sort_values(
        by="macro_f1_change",
        ascending=False,
    )

    print(
        comparison_table.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    best_model = comparison_table["macro_f1_change"].idxmax()
    best_change = comparison_table.loc[
        best_model,
        "macro_f1_change",
    ]

    print(
        f"\nLargest macro-F1 change: {best_model} "
        f"({best_change:+.4f})"
    )


# ------------------------------------------------------------
# 15. SAVE RUN REPORT
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "step9c_run_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("STEP 9C: BASELINE vs MORPHOLOGY-ENRICHED MODEL COMPARISON\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Input file:\n{INPUT_FILE}\n\n")

    f.write(f"Development rows: {len(df)}\n")
    f.write(f"Number of classes: {number_of_classes}\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"CV folds: {EXPECTED_FOLDS}\n\n")

    f.write(f"Baseline feature count: {len(BASELINE_FEATURES)}\n")
    f.write(
        "Morphology-enriched feature count: "
        f"{len(ENRICHED_FEATURES)}\n\n"
    )

    f.write("Final OOF results:\n")
    f.write(
        overall_results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    f.write("\n\n")
    f.write("Final test set was NOT loaded or evaluated.\n")
    f.write("Existing CV fold assignments were preserved.\n")
    f.write("Imputation and scaling were fitted inside each training fold.\n")
    f.write("Group leakage checks passed for all folds.\n")


# ------------------------------------------------------------
# 16. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9C COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nMain output files:")
print("1.", overall_results_path)
print("2.", fold_metrics_path)
print("3.", class_recall_path)
print("4.", report_path)

print("\nAdditional outputs:")
print("- OOF predictions for every experiment/model")
print("- Confusion matrices for every experiment/model")
print("- Classification reports for every experiment/model")
print("- Feature lists used in both experiments")

print("\nFinal test set was not loaded or evaluated.")