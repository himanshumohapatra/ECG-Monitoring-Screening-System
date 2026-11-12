# ============================================================
# STEP 6 — MODEL COMPARISON USING EXISTING GROUPED CV FOLDS
# MIT-BIH Arrhythmia Database
#
# IMPORTANT:
# - Uses development_with_cv_folds.csv only
# - Does NOT use final_test.csv
# - Uses the CV folds already created in Step 5B
# - Compares Core vs Extended features
# - Models:
#       1. Logistic Regression
#       2. Random Forest
#       3. SVM (LinearSVC)
#       4. XGBoost
#       5. MLP
# - Handles missing values using median imputation
# - Adds missingness indicators
# - Calculates:
#       Accuracy
#       Balanced Accuracy
#       Macro F1
#       Per-class Recall / Sensitivity
# - Measures training and prediction time
# - Creates confusion matrices
# - Saves all results into ml_model_comparison
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import time
import warnings
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

BASE_DIR = (
    r"C:\Users\Himanshu Mohapatra"
    r"\OneDrive\Desktop\Project_Main"
)

CV_DIR = os.path.join(
    BASE_DIR,
    "ml_grouped_cv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml_model_comparison"
)

DEVELOPMENT_FILE = os.path.join(
    CV_DIR,
    "development_with_cv_folds.csv"
)

EXPECTED_FOLDS = [1, 2, 3, 4, 5]


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. PACKAGE CHECK
# ============================================================

print("=" * 80)
print("STEP 6 — MODEL COMPARISON")
print("=" * 80)

print("\nChecking required packages...")

print(
    f"  numpy: {np.__version__}"
)

print(
    f"  pandas: {pd.__version__}"
)

try:
    import sklearn

    print(
        f"  scikit-learn: {sklearn.__version__}"
    )

except Exception as error:

    print(
        "ERROR importing scikit-learn:",
        error
    )

    sys.exit(1)


try:
    import matplotlib

    print(
        f"  matplotlib: {matplotlib.__version__}"
    )

except Exception as error:

    print(
        "ERROR importing matplotlib:",
        error
    )

    sys.exit(1)


try:
    import xgboost

    print(
        f"  xgboost: {xgboost.__version__}"
    )

except Exception as error:

    print(
        "ERROR importing xgboost:",
        error
    )

    sys.exit(1)


print("\nPackage check completed.")


# ============================================================
# 5. CHECK INPUT FILE
# ============================================================

if not os.path.exists(
    DEVELOPMENT_FILE
):

    print("\nERROR:")
    print(
        "Development file was not found:"
    )

    print(
        DEVELOPMENT_FILE
    )

    sys.exit(1)


print("\nInput file found:")
print(
    DEVELOPMENT_FILE
)


# ============================================================
# 6. LOAD DATA
# ============================================================

print(
    "\nLoading development dataset..."
)

df = pd.read_csv(
    DEVELOPMENT_FILE,
    low_memory=False
)

print(
    "\nDataset loaded successfully."
)

print(
    f"Rows   : {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)


# ============================================================
# 7. BASIC CHECKS
# ============================================================

required_columns = [
    "record_id",
    "group_id",
    "cv_fold",
    "beat_label",
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:

    print("\nERROR:")
    print(
        "Missing required columns:"
    )

    for col in missing_required:
        print(
            f"  - {col}"
        )

    sys.exit(1)


print(
    "\nBasic dataset information:"
)

print(
    f"Records : {df['record_id'].nunique()}"
)

print(
    f"Groups  : {df['group_id'].nunique()}"
)

print(
    f"CV folds: "
    f"{sorted(df['cv_fold'].dropna().unique())}"
)

print(
    f"Classes : {df['beat_label'].nunique()}"
)


# ============================================================
# 8. CV FOLD CHECK
# ============================================================

actual_folds = sorted(
    df["cv_fold"]
    .dropna()
    .astype(int)
    .unique()
)

if actual_folds != EXPECTED_FOLDS:

    print("\nERROR:")
    print(
        "Expected CV folds:"
    )

    print(
        EXPECTED_FOLDS
    )

    print(
        "Found:"
    )

    print(
        actual_folds
    )

    sys.exit(1)


# ============================================================
# 9. GROUP LEAKAGE CHECK
# ============================================================

print(
    "\nChecking group leakage..."
)

group_fold_counts = (
    df.groupby(
        "group_id"
    )["cv_fold"]
    .nunique()
)

leaking_groups = (
    group_fold_counts[
        group_fold_counts > 1
    ]
)

if len(leaking_groups) > 0:

    print(
        "\nERROR:"
    )

    print(
        "Group leakage detected!"
    )

    print(
        leaking_groups
    )

    sys.exit(1)


print(
    "Group leakage check PASSED."
)

print(
    "Every group belongs to exactly one CV fold."
)


# ============================================================
# 10. CHECK 201 / 202
# ============================================================

records_201_202 = df[
    df["record_id"].isin(
        [201, 202]
    )
]

if len(records_201_202) > 0:

    groups_201_202 = (
        records_201_202[
            "group_id"
        ]
        .astype(str)
        .unique()
    )

    folds_201_202 = (
        records_201_202[
            "cv_fold"
        ]
        .astype(int)
        .unique()
    )

    print(
        "\n201/202 subject grouping check:"
    )

    print(
        "Groups:",
        groups_201_202
    )

    print(
        "Folds:",
        folds_201_202
    )

    if len(folds_201_202) != 1:

        print(
            "\nERROR:"
        )

        print(
            "Records 201 and 202 are "
            "split across different folds."
        )

        sys.exit(1)

    print(
        "201 and 202 remain together."
    )


# ============================================================
# 11. CLASS LABELS
# ============================================================

print(
    "\nPreparing class labels..."
)

labels = sorted(
    df["beat_label"]
    .astype(str)
    .unique()
)

n_classes = len(labels)

print(
    f"Number of classes: {n_classes}"
)

print(
    "Classes:"
)

print(
    labels
)


# Global mapping.
#
# This mapping is used for evaluation.
# XGBoost receives a fold-specific mapping when necessary.
# ============================================================

label_to_int = {
    label: index
    for index, label
    in enumerate(labels)
}

int_to_label = {
    index: label
    for label, index
    in label_to_int.items()
}

df["y_encoded"] = (
    df["beat_label"]
    .astype(str)
    .map(label_to_int)
)

if df["y_encoded"].isna().any():

    print(
        "\nERROR:"
    )

    print(
        "Some beat labels could not be encoded."
    )

    sys.exit(1)

df["y_encoded"] = (
    df["y_encoded"]
    .astype(int)
)


# ============================================================
# 12. FEATURE SETS
# ============================================================

CORE_FEATURES = [
    "RR_interval_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms",
]

EXTENDED_FEATURES = [
    "RR_interval_ms",
    "R_amplitude_mV",
    "P_amplitude_mV",
    "P_duration_ms",
    "QRS_duration_ms",
    "PR_interval_ms",
    "QT_interval_ms",
    "T_amplitude_mV",
    "T_duration_ms",
]

FEATURE_SETS = {
    "Core": CORE_FEATURES,
    "Extended": EXTENDED_FEATURES,
}


# ============================================================
# 13. FEATURE CHECK
# ============================================================

print(
    "\nChecking feature columns..."
)

all_features = sorted(
    set(
        CORE_FEATURES +
        EXTENDED_FEATURES
    )
)

missing_features = [
    feature
    for feature in all_features
    if feature not in df.columns
]

if missing_features:

    print(
        "\nERROR:"
    )

    print(
        "Missing feature columns:"
    )

    for feature in missing_features:
        print(
            f"  - {feature}"
        )

    sys.exit(1)


print(
    "All required feature columns are present."
)


# ============================================================
# 14. MISSINGNESS REPORT
# ============================================================

print(
    "\nFeature missingness:"
)

for feature_set_name, features in FEATURE_SETS.items():

    print(
        f"\n{feature_set_name} features:"
    )

    for feature in features:

        missing_count = (
            df[feature]
            .isna()
            .sum()
        )

        missing_percent = (
            missing_count /
            len(df) *
            100
        )

        print(
            f"  {feature:20s} "
            f"{missing_count:8,d} "
            f"({missing_percent:7.3f}%)"
        )


# ============================================================
# 15. PREPROCESSOR
# ============================================================

def make_preprocessor(
    features,
    scale=False
):

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]

    if scale:

        steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    return Pipeline(
        steps
    )


# ============================================================
# 16. MODEL CREATORS
# ============================================================

def create_logistic(features):

    preprocessor = make_preprocessor(
        features,
        scale=True
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

def create_random_forest(features):

    preprocessor = make_preprocessor(
        features,
        scale=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ------------------------------------------------------------
# Linear SVM
# ------------------------------------------------------------

def create_svm(features):

    preprocessor = make_preprocessor(
        features,
        scale=True
    )

    model = LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        dual="auto",
        random_state=RANDOM_STATE
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ------------------------------------------------------------
# XGBoost
#
# num_class is NOT used to solve the rare-class issue.
# The target is locally remapped for each fold.
# ------------------------------------------------------------

def create_xgboost(
    features,
    local_num_classes
):

    preprocessor = make_preprocessor(
        features,
        scale=False
    )

    model = XGBClassifier(
        objective="multi:softmax",

        num_class=local_num_classes,

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ------------------------------------------------------------
# MLP
# ------------------------------------------------------------

def create_mlp(features):

    preprocessor = make_preprocessor(
        features,
        scale=True
    )

    model = MLPClassifier(
        hidden_layer_sizes=(
            128,
            64
        ),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        batch_size=256,

        learning_rate_init=0.001,

        max_iter=100,

        early_stopping=True,

        validation_fraction=0.1,

        n_iter_no_change=10,

        random_state=RANDOM_STATE
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# 17. MODEL LIST
# ============================================================

MODEL_CREATORS = {
    "Logistic Regression":
        create_logistic,

    "Random Forest":
        create_random_forest,

    "SVM":
        create_svm,

    "XGBoost":
        create_xgboost,

    "MLP":
        create_mlp,
}


# ============================================================
# 18. MLP SAMPLE-WEIGHT SUPPORT
# ============================================================

try:

    mlp_fit_signature = inspect.signature(
        MLPClassifier.fit
    )

    mlp_supports_sample_weight = (
        "sample_weight"
        in mlp_fit_signature.parameters
    )

except Exception:

    mlp_supports_sample_weight = False


print(
    "\nMLP sample-weight support:"
)

print(
    mlp_supports_sample_weight
)


# ============================================================
# 19. SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(y):

    y_array = np.asarray(
        y
    )

    unique_classes, counts = np.unique(
        y_array,
        return_counts=True
    )

    n_samples = len(
        y_array
    )

    n_classes_present = len(
        unique_classes
    )

    weight_map = {}

    for cls, count in zip(
        unique_classes,
        counts
    ):

        weight_map[cls] = (
            n_samples /
            (
                n_classes_present *
                count
            )
        )

    weights = np.array(
        [
            weight_map[value]
            for value in y_array
        ],
        dtype=np.float64
    )

    return weights


# ============================================================
# 20. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
    }


# ============================================================
# 21. RESULT STORAGE
# ============================================================

fold_results = []

per_class_recall_results = []

all_oof_predictions = {}


# ============================================================
# 22. START CV
# ============================================================

print("\n")
print("=" * 80)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("=" * 80)


for feature_set_name, features in FEATURE_SETS.items():

    print("\n")
    print("#" * 80)

    print(
        f"FEATURE SET: {feature_set_name}"
    )

    print(
        "#" * 80
    )

    print(
        "Features:"
    )

    for feature in features:

        print(
            f"  - {feature}"
        )


    for model_name, model_creator in MODEL_CREATORS.items():

        print("\n")
        print(
            "-" * 80
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            f"FEATURE SET: {feature_set_name}"
        )

        print(
            "-" * 80
        )


        oof_true = []
        oof_pred = []


        # ====================================================
        # FIVE EXISTING FOLDS
        # ====================================================

        for fold in EXPECTED_FOLDS:

            print("\n")
            print(
                f"Fold {fold}/5"
            )


            train_data = df[
                df["cv_fold"] != fold
            ].copy()

            validation_data = df[
                df["cv_fold"] == fold
            ].copy()


            print(
                f"Training rows   : "
                f"{len(train_data):,}"
            )

            print(
                f"Validation rows : "
                f"{len(validation_data):,}"
            )


            X_train = (
                train_data[
                    features
                ].copy()
            )

            X_val = (
                validation_data[
                    features
                ].copy()
            )


            y_train_global = (
                train_data[
                    "y_encoded"
                ]
                .astype(int)
                .to_numpy()
            )

            y_val_global = (
                validation_data[
                    "y_encoded"
                ]
                .astype(int)
                .to_numpy()
            )


            # ================================================
            # TRAINING CLASSES
            # ================================================

            training_classes = np.sort(
                np.unique(
                    y_train_global
                )
            )

            missing_training_classes = [
                class_index
                for class_index
                in range(n_classes)
                if class_index
                not in training_classes
            ]


            if (
                len(
                    missing_training_classes
                )
                > 0
            ):

                print(
                    "\nWARNING:"
                )

                print(
                    "The following classes are "
                    "absent from this fold's "
                    "training data:"
                )

                print(
                    [
                        int_to_label[
                            index
                        ]
                        for index
                        in missing_training_classes
                    ]
                )


            # ================================================
            # CREATE MODEL
            # ================================================

            if model_name == "XGBoost":

                # --------------------------------------------
                # XGBoost needs consecutive local class IDs.
                #
                # Example:
                #
                # Global:
                # A=1
                # E=2
                # F=3
                #
                # If E is absent:
                #
                # XGBoost receives:
                # A=0
                # F=1
                # --------------------------------------------

                local_class_map = {
                    global_class:
                        local_class
                    for local_class,
                    global_class
                    in enumerate(
                        training_classes
                    )
                }


                local_to_global = {
                    local_class:
                        global_class
                    for global_class,
                    local_class
                    in local_class_map.items()
                }


                y_train_local = np.array(
                    [
                        local_class_map[
                            value
                        ]
                        for value
                        in y_train_global
                    ],
                    dtype=int
                )


                local_num_classes = len(
                    training_classes
                )


                model = create_xgboost(
                    features,
                    local_num_classes
                )


                sample_weights = (
                    calculate_sample_weights(
                        y_train_local
                    )
                )


            else:

                model = model_creator(
                    features
                )


                y_train_local = (
                    y_train_global
                )


                local_to_global = None


                sample_weights = (
                    calculate_sample_weights(
                        y_train_local
                    )
                )


            # ================================================
            # TRAIN
            # ================================================

            print(
                "Training..."
            )

            train_start = (
                time.perf_counter()
            )


            try:

                if (
                    model_name == "XGBoost"
                ):

                    model.fit(
                        X_train,
                        y_train_local,
                        model__sample_weight=
                        sample_weights
                    )


                elif (
                    model_name == "MLP"
                    and
                    mlp_supports_sample_weight
                ):

                    model.fit(
                        X_train,
                        y_train_local,
                        model__sample_weight=
                        sample_weights
                    )


                else:

                    model.fit(
                        X_train,
                        y_train_local
                    )


            except Exception as error:

                print("\n")
                print(
                    "ERROR DURING TRAINING"
                )

                print(
                    f"Model: {model_name}"
                )

                print(
                    f"Feature set: "
                    f"{feature_set_name}"
                )

                print(
                    f"Fold: {fold}"
                )

                print(
                    f"Error: {error}"
                )

                raise


            train_time = (
                time.perf_counter()
                -
                train_start
            )


            print(
                f"Training time: "
                f"{train_time:.2f} seconds"
            )


            # ================================================
            # PREDICT
            # ================================================

            print(
                "Predicting..."
            )

            prediction_start = (
                time.perf_counter()
            )


            y_pred_local = model.predict(
                X_val
            )


            prediction_time = (
                time.perf_counter()
                -
                prediction_start
            )


            # ================================================
            # CONVERT XGBOOST LOCAL LABELS
            # BACK TO GLOBAL LABELS
            # ================================================

            if (
                model_name == "XGBoost"
            ):

                y_pred_global = np.array(
                    [
                        local_to_global[
                            int(value)
                        ]
                        for value
                        in y_pred_local
                    ],
                    dtype=int
                )

            else:

                y_pred_global = np.asarray(
                    y_pred_local,
                    dtype=int
                )


            print(
                f"Prediction time: "
                f"{prediction_time:.2f} seconds"
            )


            # ================================================
            # METRICS
            # ================================================

            metrics = calculate_metrics(
                y_val_global,
                y_pred_global
            )


            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Balanced Accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"Macro F1: "
                f"{metrics['macro_f1']:.4f}"
            )


            # ================================================
            # STORE FOLD RESULTS
            # ================================================

            fold_results.append(
                {
                    "feature_set":
                        feature_set_name,

                    "model":
                        model_name,

                    "fold":
                        fold,

                    "train_rows":
                        len(train_data),

                    "validation_rows":
                        len(validation_data),

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "balanced_accuracy":
                        metrics[
                            "balanced_accuracy"
                        ],

                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],

                    "training_time_seconds":
                        train_time,

                    "prediction_time_seconds":
                        prediction_time,
                }
            )


            # ================================================
            # PER-CLASS RECALL
            # ================================================

            recalls = recall_score(
                y_val_global,
                y_pred_global,
                labels=np.arange(
                    n_classes
                ),
                average=None,
                zero_division=0
            )


            class_support = np.bincount(
                y_val_global,
                minlength=n_classes
            )


            for class_index, class_label in enumerate(labels):

                per_class_recall_results.append(
                    {
                        "feature_set":
                            feature_set_name,

                        "model":
                            model_name,

                        "fold":
                            fold,

                        "class":
                            class_label,

                        "support":
                            int(
                                class_support[
                                    class_index
                                ]
                            ),

                        "recall":
                            recalls[
                                class_index
                            ],
                    }
                )


            # ================================================
            # OOF STORAGE
            # ================================================

            oof_true.extend(
                y_val_global.tolist()
            )

            oof_pred.extend(
                y_pred_global.tolist()
            )


            # ================================================
            # CLEAN
            # ================================================

            del model
            del X_train
            del X_val
            del y_train_global
            del y_val_global


        # ====================================================
        # OVERALL OOF
        # ====================================================

        oof_true = np.asarray(
            oof_true,
            dtype=int
        )

        oof_pred = np.asarray(
            oof_pred,
            dtype=int
        )


        oof_metrics = calculate_metrics(
            oof_true,
            oof_pred
        )


        print("\n")
        print(
            "=" * 80
        )

        print(
            "OVERALL OUT-OF-FOLD RESULTS"
        )

        print(
            f"Feature set: "
            f"{feature_set_name}"
        )

        print(
            f"Model: "
            f"{model_name}"
        )

        print(
            f"Accuracy: "
            f"{oof_metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{oof_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{oof_metrics['macro_f1']:.4f}"
        )

        print(
            "=" * 80
        )


        all_oof_predictions[
            (
                feature_set_name,
                model_name
            )
        ] = {
            "y_true":
                oof_true,

            "y_pred":
                oof_pred,
        }


# ============================================================
# 23. RESULT DATAFRAMES
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

per_class_recall_df = pd.DataFrame(
    per_class_recall_results
)


# ============================================================
# 24. SAVE FOLD RESULTS
# ============================================================

fold_results_file = os.path.join(
    OUTPUT_DIR,
    "fold_results.csv"
)

fold_results_df.to_csv(
    fold_results_file,
    index=False
)


# ============================================================
# 25. MODEL SUMMARY
# ============================================================

summary_rows = []


for (
    feature_set_name,
    model_name
), predictions in all_oof_predictions.items():

    y_true = predictions[
        "y_true"
    ]

    y_pred = predictions[
        "y_pred"
    ]


    oof_metrics = calculate_metrics(
        y_true,
        y_pred
    )


    model_fold_results = (
        fold_results_df[
            (
                fold_results_df[
                    "feature_set"
                ]
                ==
                feature_set_name
            )
            &
            (
                fold_results_df[
                    "model"
                ]
                ==
                model_name
            )
        ]
    )


    summary_rows.append(
        {
            "feature_set":
                feature_set_name,

            "model":
                model_name,

            "oof_accuracy":
                oof_metrics[
                    "accuracy"
                ],

            "oof_balanced_accuracy":
                oof_metrics[
                    "balanced_accuracy"
                ],

            "oof_macro_f1":
                oof_metrics[
                    "macro_f1"
                ],

            "mean_accuracy":
                model_fold_results[
                    "accuracy"
                ].mean(),

            "std_accuracy":
                model_fold_results[
                    "accuracy"
                ].std(),

            "mean_balanced_accuracy":
                model_fold_results[
                    "balanced_accuracy"
                ].mean(),

            "std_balanced_accuracy":
                model_fold_results[
                    "balanced_accuracy"
                ].std(),

            "mean_macro_f1":
                model_fold_results[
                    "macro_f1"
                ].mean(),

            "std_macro_f1":
                model_fold_results[
                    "macro_f1"
                ].std(),

            "mean_training_time_seconds":
                model_fold_results[
                    "training_time_seconds"
                ].mean(),

            "total_training_time_seconds":
                model_fold_results[
                    "training_time_seconds"
                ].sum(),

            "mean_prediction_time_seconds":
                model_fold_results[
                    "prediction_time_seconds"
                ].mean(),

            "total_prediction_time_seconds":
                model_fold_results[
                    "prediction_time_seconds"
                ].sum(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 26. SORT BY OOF MACRO F1
# ============================================================

summary_df = summary_df.sort_values(
    by="oof_macro_f1",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 27. SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "summary_results.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 28. SAVE PER-CLASS RECALL
# ============================================================

per_class_recall_file = os.path.join(
    OUTPUT_DIR,
    "per_class_recall.csv"
)

per_class_recall_df.to_csv(
    per_class_recall_file,
    index=False
)


# ============================================================
# 29. CONFUSION MATRIX DIRECTORY
# ============================================================

CM_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices"
)

os.makedirs(
    CM_DIR,
    exist_ok=True
)


# ============================================================
# 30. CONFUSION MATRICES
# ============================================================

print("\n")
print(
    "=" * 80
)

print(
    "CREATING CONFUSION MATRICES"
)

print(
    "=" * 80
)


for (
    feature_set_name,
    model_name
), predictions in all_oof_predictions.items():

    y_true = predictions[
        "y_true"
    ]

    y_pred = predictions[
        "y_pred"
    ]


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            n_classes
        )
    )


    safe_feature_name = (
        feature_set_name
        .replace(" ", "_")
    )

    safe_model_name = (
        model_name
        .replace(" ", "_")
    )


    # --------------------------------------------------------
    # RAW CSV
    # --------------------------------------------------------

    cm_csv_file = os.path.join(
        CM_DIR,
        f"{safe_feature_name}__"
        f"{safe_model_name}__raw.csv"
    )


    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    cm_df.to_csv(
        cm_csv_file
    )


    # --------------------------------------------------------
    # NORMALIZED
    # --------------------------------------------------------

    row_sums = cm.sum(
        axis=1,
        keepdims=True
    )


    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_sums != 0
    )


    cm_norm_csv_file = os.path.join(
        CM_DIR,
        f"{safe_feature_name}__"
        f"{safe_model_name}__normalized.csv"
    )


    cm_norm_df = pd.DataFrame(
        cm_normalized,
        index=labels,
        columns=labels
    )

    cm_norm_df.to_csv(
        cm_norm_csv_file
    )


    # --------------------------------------------------------
    # RAW PNG
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{feature_set_name} - "
        f"{model_name}\n"
        f"Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        np.arange(n_classes),
        labels,
        rotation=90
    )

    plt.yticks(
        np.arange(n_classes),
        labels
    )

    plt.colorbar()

    plt.tight_layout()


    raw_png_file = os.path.join(
        CM_DIR,
        f"{safe_feature_name}__"
        f"{safe_model_name}__raw.png"
    )


    plt.savefig(
        raw_png_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # NORMALIZED PNG
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        cm_normalized,
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"{feature_set_name} - "
        f"{model_name}\n"
        f"Normalized Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        np.arange(n_classes),
        labels,
        rotation=90
    )

    plt.yticks(
        np.arange(n_classes),
        labels
    )

    plt.colorbar()

    plt.tight_layout()


    norm_png_file = os.path.join(
        CM_DIR,
        f"{safe_feature_name}__"
        f"{safe_model_name}__normalized.png"
    )


    plt.savefig(
        norm_png_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


print(
    "Confusion matrices created."
)


# ============================================================
# 31. LABEL MAPPING
# ============================================================

label_mapping_df = pd.DataFrame(
    [
        {
            "class_index":
                index,

            "beat_label":
                label
        }

        for index, label
        in int_to_label.items()
    ]
)


label_mapping_file = os.path.join(
    OUTPUT_DIR,
    "label_mapping.csv"
)


label_mapping_df.to_csv(
    label_mapping_file,
    index=False
)


# ============================================================
# 32. RUN INFORMATION
# ============================================================

run_info_file = os.path.join(
    OUTPUT_DIR,
    "run_info.txt"
)


with open(
    run_info_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 6 — MODEL COMPARISON\n"
    )

    f.write(
        "=" * 70 +
        "\n\n"
    )

    f.write(
        "Development dataset:\n"
    )

    f.write(
        f"{DEVELOPMENT_FILE}\n\n"
    )

    f.write(
        "FINAL TEST SET USED: NO\n"
    )

    f.write(
        "Existing CV folds used: YES\n\n"
    )

    f.write(
        f"Rows: {len(df):,}\n"
    )

    f.write(
        f"Records: "
        f"{df['record_id'].nunique()}\n"
    )

    f.write(
        f"Groups: "
        f"{df['group_id'].nunique()}\n"
    )

    f.write(
        f"Classes: {n_classes}\n\n"
    )

    f.write(
        "Feature sets:\n"
    )

    for name, features in FEATURE_SETS.items():

        f.write(
            f"\n{name}:\n"
        )

        for feature in features:

            f.write(
                f"  - {feature}\n"
            )


    f.write(
        "\nModels:\n"
    )

    for model_name in MODEL_CREATORS:

        f.write(
            f"  - {model_name}\n"
        )


    f.write(
        "\nSVM implementation:\n"
    )

    f.write(
        "LinearSVC\n"
    )

    f.write(
        "Used because the development dataset "
        "contains approximately 175,000 beats.\n"
    )

    f.write(
        "\nMissing-value handling:\n"
    )

    f.write(
        "Median imputation + missingness indicators.\n"
    )

    f.write(
        "\nXGBoost rare-class handling:\n"
    )

    f.write(
        "Training labels are locally remapped "
        "per CV fold so that XGBoost receives "
        "consecutive class IDs.\n"
    )


# ============================================================
# 33. FINAL DISPLAY
# ============================================================

print("\n")
print(
    "=" * 80
)

print(
    "MODEL COMPARISON COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nOverall OOF model comparison:"
)


display_columns = [
    "feature_set",
    "model",
    "oof_accuracy",
    "oof_balanced_accuracy",
    "oof_macro_f1",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_training_time_seconds",
]


print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n")
print(
    "=" * 80
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 80
)


print(
    f"\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print(
    f"\n1. {fold_results_file}"
)

print(
    f"2. {per_class_recall_file}"
)

print(
    f"3. {summary_file}"
)

print(
    f"4. {label_mapping_file}"
)

print(
    f"5. {run_info_file}"
)

print(
    f"6. Confusion matrices:"
)

print(
    CM_DIR
)


print("\n")
print(
    "=" * 80
)

print(
    "IMPORTANT NEXT STEP"
)

print(
    "=" * 80
)

print(
    "\nDo NOT evaluate final_test.csv yet."
)

print(
    "First select the best model + feature set "
    "using the development CV results."
)

print(
    "Then the selected model will be retrained "
    "on the complete development dataset."
)

print(
    "Only after that will final_test.csv be used "
    "once for the final unbiased evaluation."
)

print(
    "\nStep 6 finished successfully."
)

print(
    "=" * 80
)