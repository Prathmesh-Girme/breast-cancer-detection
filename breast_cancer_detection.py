########################################################
# Breast Cancer Detection using Machine Learning
#
# Pipeline       : SimpleImputer -> StandardScaler -> classifier
# Models         : Logistic Regression, SVM (RBF), Random Forest
# Positive class : MALIGNANT tumour (1). In the CSV the column "target" uses
#                  0 = malignant and 1 = benign, so it is flipped here and
#                  recall / precision / F1 all describe malignant cases.
#
# For cancer screening a missed malignant tumour (false negative) is the
# costly mistake, so recall on malignant cases is tracked next to accuracy.
########################################################

########################################################
# Required Python Packages
########################################################
import json
import os
import platform
import warnings
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

########################################################
# Settings
########################################################

warnings.filterwarnings("ignore", category=FutureWarning)

SHOW_PLOTS = False  # False: only save plots to files (works without a display)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must come after backend selection

########################################################
# File Paths
########################################################

DATA_PATH = "breast_cancer.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_PATH = os.path.join(OUTPUT_DIR, "breast_cancer_model.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
HELD_OUT_PATH = os.path.join(OUTPUT_DIR, "held_out_samples.csv")

########################################################
# Columns and Labels
########################################################

TARGET = "target"
MALIGNANT_CODE = 0                       # value of "target" that means malignant in the CSV
CLASS_LABELS = ["Benign", "Malignant"]   # after flipping: 0 = Benign, 1 = Malignant

########################################################
# Model Settings
########################################################

RANDOM_STATE = 42
TRAIN_PERCENTAGE = 0.7
CV_FOLDS = 5
SCREENING_RECALL = 0.98   # target recall on malignant cases for the screening threshold
MODEL_STEP = "model"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

CANDIDATE_MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {f"{MODEL_STEP}__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "SVM (RBF)": (
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        {f"{MODEL_STEP}__C": [0.1, 1, 10], f"{MODEL_STEP}__gamma": ["scale", 0.01]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        {f"{MODEL_STEP}__max_depth": [None, 5, 10], f"{MODEL_STEP}__min_samples_leaf": [1, 3]},
    ),
}


########################################################
# Function name : read_data
# Description   : Read the dataset into a pandas dataframe
# Input         : Path of CSV file
# Output        : Dataframe
# Author        : Prathmesh Sunil Girme
########################################################
def read_data(path):
    """Read the data into pandas dataframe"""
    return pd.read_csv(path)


########################################################
# Function name : validate_dataset
# Description   : Check the target column and that all features are numbers
# Input         : Dataframe
# Output        : Nothing (raises ValueError if the data is not usable)
# Author        : Prathmesh Sunil Girme
########################################################
def validate_dataset(dataset):
    """Stop early with a clear message if the dataset is not as expected"""
    if TARGET not in dataset.columns:
        raise ValueError(f"Dataset has no '{TARGET}' column")

    if set(dataset[TARGET].unique()) - {0, 1}:
        raise ValueError(f"'{TARGET}' must contain only 0 and 1")

    features = dataset.drop(columns=TARGET)
    non_numeric = features.columns[~features.dtypes.apply(pd.api.types.is_numeric_dtype)].tolist()
    if non_numeric:
        raise ValueError(f"These feature columns are not numeric: {non_numeric}")


########################################################
# Function name : prepare_target
# Description   : Split features and target, flipping the target so that
#                 1 = Malignant (the class we most want to catch)
# Input         : Dataframe
# Output        : Features dataframe, target series
# Author        : Prathmesh Sunil Girme
########################################################
def prepare_target(dataset):
    features = dataset.drop(columns=TARGET)
    target = (dataset[TARGET] == MALIGNANT_CODE).astype(int)
    return features, target


########################################################
# Function name : dataset_statistics
# Description   : Display shape, class balance and highly correlated features
# Input         : Features, target
# Output        : Nothing (prints)
# Author        : Prathmesh Sunil Girme
########################################################
def dataset_statistics(features, target):
    print("Dataset Shape ::", features.shape)
    counts = target.value_counts().sort_index()
    for label in counts.index:
        print(f"Class {label} ({CLASS_LABELS[label]}) :: {counts[label]} rows ({counts[label] / len(target):.1%})")
    print(f"Majority-class baseline accuracy (always predict Benign) :: {1 - target.mean():.3f}")

    correlation = features.corr().abs()
    upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
    pairs = upper.stack()
    print(f"\nFeature pairs with correlation above 0.9 :: {int((pairs > 0.9).sum())} of {len(pairs)}")
    print("Strongest pairs:\n", pairs.sort_values(ascending=False).head(5).round(3))

    with_target = features.corrwith(target).abs().sort_values(ascending=False)
    print("\nFeatures most correlated with malignancy:\n", with_target.head(6).round(3))


########################################################
# Function name : check_missing_values
# Description   : Count missing values and duplicate rows
# Input         : Dataframe
# Output        : Total missing values
# Author        : Prathmesh Sunil Girme
########################################################
def check_missing_values(dataset):
    missing = int(dataset.isnull().sum().sum())
    print("Total missing values ::", missing)
    print("Duplicate rows       ::", int(dataset.duplicated().sum()))
    return missing


########################################################
# Function name : finish_plot
# Description   : Save the current figure (and show it if SHOW_PLOTS is True)
# Input         : File name
# Output        : Nothing (writes a PNG)
# Author        : Prathmesh Sunil Girme
########################################################
def finish_plot(file_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Plot saved to {path}")


########################################################
# Function name : plot_eda
# Description   : Exploratory plots: class balance, a key feature, strongest
#                 features, and the full correlation heatmap
# Input         : Features, target
# Output        : eda_overview.png, correlation_heatmap.png
# Author        : Prathmesh Sunil Girme
########################################################
def plot_eda(features, target):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    counts = target.value_counts().sort_index()
    axes[0].bar(CLASS_LABELS, counts.values, color=["#4c78a8", "#e45756"])
    for i, value in enumerate(counts.values):
        axes[0].text(i, value, str(value), ha="center", va="bottom")
    axes[0].set_title("Class Balance")
    axes[0].set_ylabel("Patients")

    for label in (0, 1):
        axes[1].hist(features.loc[target == label, "worst radius"], bins=20, alpha=0.6,
                     label=CLASS_LABELS[label])
    axes[1].set_title("'worst radius' by Diagnosis")
    axes[1].set_xlabel("worst radius")
    axes[1].legend()

    strongest = features.corrwith(target).abs().sort_values().tail(8)
    axes[2].barh(strongest.index, strongest.values, color="#72b7b2")
    axes[2].set_title("Correlation with Malignancy (absolute)")
    finish_plot("eda_overview.png")

    plt.figure(figsize=(9, 8))
    image = plt.imshow(features.corr(), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(features.shape[1]), features.columns, rotation=90, fontsize=6)
    plt.yticks(range(features.shape[1]), features.columns, fontsize=6)
    plt.title("Feature Correlation Heatmap")
    finish_plot("correlation_heatmap.png")


########################################################
# Function name : split_dataset
# Description   : Split into train/test using stratified sampling
# Input         : Features, target, train percentage
# Output        : train_x, test_x, train_y, test_y
# Author        : Prathmesh Sunil Girme
########################################################
def split_dataset(features, target, train_percentage, random_state=RANDOM_STATE):
    """Stratified split so both sets keep the same malignant share"""
    return train_test_split(features, target, train_size=train_percentage,
                            random_state=random_state, stratify=target)


########################################################
# Function name : build_pipeline
# Description   : Build a Pipeline:
#                 SimpleImputer  : fills any missing value with the median
#                 StandardScaler : puts all 30 measurements on one scale
#                 model          : the classifier
# Input         : Classifier
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
########################################################
def build_pipeline(estimator):
    """The imputer is not needed for this clean file, but it makes the saved
    model safe when a future measurement arrives with a gap."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (MODEL_STEP, estimator),
    ])


########################################################
# Function name : make_cv
# Description   : Cross-validation splitter shared by every comparison
# Input         : Nothing
# Output        : StratifiedKFold object
# Author        : Prathmesh Sunil Girme
########################################################
def make_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


########################################################
# Function name : tune_candidate
# Description   : Grid search one algorithm with 5-fold CV on TRAINING data
# Input         : Model name, classifier, parameter grid, training data
# Output        : Fitted best pipeline, dictionary of CV scores
# Author        : Prathmesh Sunil Girme
########################################################
def tune_candidate(name, estimator, grid, train_x, train_y):
    search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING,
                          refit="roc_auc", cv=make_cv(), n_jobs=-1)
    search.fit(train_x, train_y)

    best = search.best_index_
    results = search.cv_results_
    settings = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in search.best_params_.items())
    row = {
        "Model": name,
        "Settings": settings,
        "CV_Accuracy": results["mean_test_accuracy"][best],
        "CV_Precision": results["mean_test_precision"][best],
        "CV_Recall": results["mean_test_recall"][best],
        "CV_F1": results["mean_test_f1"][best],
        "CV_ROC_AUC": results["mean_test_roc_auc"][best],
        "CV_ROC_AUC_Std": results["std_test_roc_auc"][best],
    }
    print(f"  {name:<20} {settings:<32} CV AUC {row['CV_ROC_AUC']:.4f} (+/- {row['CV_ROC_AUC_Std']:.4f}) "
          f"| recall {row['CV_Recall']:.3f}")
    return search.best_estimator_, row


########################################################
# Function name : evaluate_model
# Description   : Accuracy, precision, recall, F1, ROC-AUC and error counts
# Input         : Model, features, labels, decision threshold
# Output        : Dictionary of metrics, predicted labels
# Author        : Prathmesh Sunil Girme
########################################################
def evaluate_model(model, x, y, threshold=0.5):
    """Positive class = Malignant"""
    probability = model.predict_proba(x)[:, 1]
    predictions = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "missed_malignant": int(fn),
        "false_alarms": int(fp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, predictions


########################################################
# Function name : compare_models
# Description   : Evaluate every tuned model on the test set ONCE
# Input         : Dictionary {name: model}, CV rows, train and test data
# Output        : Comparison dataframe
# Author        : Prathmesh Sunil Girme
########################################################
def compare_models(models, cv_rows, train_x, train_y, test_x, test_y):
    rows = []
    for row in cv_rows:
        model = models[row["Model"]]
        train_metrics, _ = evaluate_model(model, train_x, train_y)
        test_metrics, _ = evaluate_model(model, test_x, test_y)
        rows.append({
            **row,
            "Train_Accuracy": train_metrics["accuracy"],
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Precision": test_metrics["precision"],
            "Test_Recall": test_metrics["recall"],
            "Test_F1": test_metrics["f1"],
            "Test_ROC_AUC": test_metrics["roc_auc"],
            "Missed_Malignant": test_metrics["missed_malignant"],
            "False_Alarms": test_metrics["false_alarms"],
        })
    return pd.DataFrame(rows)


########################################################
# Function name : select_best_model
# Description   : Choose the final model by CV ROC-AUC rounded to 3 decimals
#                 (differences smaller than that are treated as ties), then by
#                 CV recall on malignant cases. The test set is not used.
# Input         : Comparison dataframe
# Output        : Name of the best model
# Author        : Prathmesh Sunil Girme
########################################################
def select_best_model(comparison):
    ranked = comparison.assign(_auc=comparison["CV_ROC_AUC"].round(3)).sort_values(
        ["_auc", "CV_Recall"], ascending=False)
    return ranked.iloc[0]["Model"]


########################################################
# Function name : tune_screening_threshold
# Description   : Find a lower threshold that reaches the target recall on
#                 malignant cases using cross-validated TRAINING predictions.
#                 Among thresholds that reach it, the highest one is used so
#                 as few false alarms as possible are created.
# Input         : Pipeline (cloned, not modified), training data
# Output        : Threshold between 0.05 and 0.95
# Author        : Prathmesh Sunil Girme
########################################################
def tune_screening_threshold(pipeline, train_x, train_y):
    from sklearn.base import clone

    probability = cross_val_predict(clone(pipeline), train_x, train_y, cv=make_cv(),
                                    method="predict_proba", n_jobs=-1)[:, 1]
    best = 0.05
    for threshold in np.arange(0.05, 0.96, 0.01):
        if recall_score(train_y, probability >= threshold, zero_division=0) >= SCREENING_RECALL:
            best = threshold
    return round(float(best), 2)


########################################################
# Function name : plot_confusion_matrix
# Description   : Display Confusion Matrix
# Input         : True labels, predicted labels, title, file name
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_confusion_matrix(y_true, y_pred, title, file_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.matshow(cm, cmap="Blues")
    fig.colorbar(image)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, pad=40)
    finish_plot(file_name)


########################################################
# Function name : plot_roc_curves
# Description   : ROC curves of all models on the test set
# Input         : Dictionary {name: model}, test data
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_roc_curves(models, test_x, test_y):
    plt.figure(figsize=(6.5, 5.5))
    for name, model in models.items():
        probability = model.predict_proba(test_x)[:, 1]
        fpr, tpr, _ = roc_curve(test_y, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(test_y, probability):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall on malignant)")
    plt.title("ROC Curves (test set)")
    plt.legend(loc="lower right", fontsize=8)
    finish_plot("roc_curves.png")


########################################################
# Function name : plot_feature_importance
# Description   : Permutation importance on the test set (top 10 features).
#                 Many features are highly correlated, so importance is
#                 shared between them and each single value looks small.
# Input         : Fitted pipeline, test data, feature names, title
# Output        : PNG file, importance series
# Author        : Prathmesh Sunil Girme
########################################################
def plot_feature_importance(model, test_x, test_y, title):
    result = permutation_importance(model, test_x, test_y, scoring="roc_auc", n_repeats=20,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=test_x.columns).sort_values(ascending=False)
    top = importance.head(10).iloc[::-1]
    errors = pd.Series(result.importances_std, index=test_x.columns)[top.index]

    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top.values, xerr=errors.values)
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title(title)
    finish_plot("feature_importance.png")
    return importance


########################################################
# Function name : predict_new_samples
# Description   : Score tumour measurements and give the malignancy probability
# Input         : Trained model, samples, threshold
# Output        : Dataframe with probability and diagnosis
# Author        : Prathmesh Sunil Girme
########################################################
def predict_new_samples(model, samples, threshold=0.5):
    probability = model.predict_proba(samples)[:, 1]
    return pd.DataFrame({
        "MalignancyProbability": probability.round(3),
        "Prediction": np.where(probability >= threshold, "Malignant", "Benign"),
    }, index=samples.index)


########################################################
# Function name : save_model / load_model
# Description   : Save or load the trained Pipeline with joblib.
#                 Only load model files you created yourself.
# Input         : Model and path / path
# Output        : Nothing / model
# Author        : Prathmesh Sunil Girme
########################################################
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


########################################################
# Function name : json_default
# Description   : Let json.dump handle numpy numbers
# Author        : Prathmesh Sunil Girme
########################################################
def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


########################################################
# Function name : save_model_metadata
# Description   : Save a "model card" next to the .joblib file: expected
#                 columns, training ranges, label meaning, thresholds,
#                 library versions and test metrics.
# Output        : outputs/model_metadata.json
# Author        : Prathmesh Sunil Girme
########################################################
def save_model_metadata(model_name, model, features, train_rows, test_metrics,
                        screening_threshold, cv_roc_auc, path=METADATA_PATH):
    estimator = model.named_steps[MODEL_STEP]
    simple_types = (int, float, str, bool, type(None))
    metadata = {
        "model_name": model_name,
        "model_class": type(estimator).__name__,
        "model_parameters": {k: v for k, v in estimator.get_params().items()
                             if isinstance(v, simple_types)},
        "selection": {"rule": "best cross-validated ROC-AUC on training data",
                      "cv_roc_auc": cv_roc_auc},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_rows": int(train_rows),
        "features": list(features.columns),
        "training_ranges": {c: [float(features[c].min()), float(features[c].max())] for c in features.columns},
        "positive_class": "Malignant (1)",
        "class_labels": {"0": CLASS_LABELS[0], "1": CLASS_LABELS[1]},
        "decision_threshold": 0.5,
        "screening_threshold": screening_threshold,
        "screening_target_recall": SCREENING_RECALL,
        "test_metrics": test_metrics,
        "disclaimer": "Educational project. Not for clinical use.",
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=json_default)
    print(f"Model metadata saved to {path}")


########################################################
# Function name : main
# Description   : Main function from where execution starts
# Author        : Prathmesh Sunil Girme
########################################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and understand the dataset
    dataset = read_data(DATA_PATH)
    validate_dataset(dataset)
    features, target = prepare_target(dataset)
    print("\n--- 1) Dataset ---")
    dataset_statistics(features, target)

    # 2) Missing values
    print("\n--- 2) Missing values ---")
    check_missing_values(dataset)

    # 3) Exploratory plots
    print("\n--- 3) Exploratory analysis ---")
    plot_eda(features, target)

    # 4) Split (stratified)
    print("\n--- 4) Train/test split ---")
    train_x, test_x, train_y, test_y = split_dataset(features, target, TRAIN_PERCENTAGE)
    print("Train_x Shape :: ", train_x.shape)
    print("Test_x Shape  :: ", test_x.shape)
    print(f"Malignant share -> train: {train_y.mean():.3f} | test: {test_y.mean():.3f}")
    test_x.assign(Actual=np.where(test_y == 1, "Malignant", "Benign")).to_csv(HELD_OUT_PATH, index=False)

    # 5) Tune every model with the same CV folds
    print("\n--- 5) Tuning models (5-fold CV on training data) ---")
    models, cv_rows = {}, []
    for name, (estimator, grid) in CANDIDATE_MODELS.items():
        models[name], row = tune_candidate(name, estimator, grid, train_x, train_y)
        cv_rows.append(row)

    # 6) Compare on the test set (once) and choose the final model
    print("\n--- 6) Model comparison ---")
    comparison = compare_models(models, cv_rows, train_x, train_y, test_x, test_y)
    comparison.round(4).to_csv(COMPARISON_PATH, index=False)
    print(comparison[["Model", "CV_ROC_AUC", "Train_Accuracy", "Test_Accuracy", "Test_Precision",
                      "Test_Recall", "Test_F1", "Test_ROC_AUC", "Missed_Malignant",
                      "False_Alarms"]].round(3).to_string(index=False))
    plot_roc_curves(models, test_x, test_y)

    final_name = select_best_model(comparison)
    final_model = models[final_name]
    final_row = comparison[comparison["Model"] == final_name].iloc[0]
    print(f"\nFinal model (best CV ROC-AUC) :: {final_name} (CV AUC {final_row['CV_ROC_AUC']:.4f})")

    # 7) Detailed report for the final model
    final_metrics, final_predictions = evaluate_model(final_model, test_x, test_y)
    final_metrics["train_accuracy"] = float(final_row["Train_Accuracy"])
    print(f"\n===== FINAL: {final_name} (threshold = 0.50) =====")
    print("Confusion Matrix:\n", confusion_matrix(test_y, final_predictions))
    print("Classification Report:\n",
          classification_report(test_y, final_predictions, target_names=CLASS_LABELS))
    plot_confusion_matrix(test_y, final_predictions, f"{final_name} - Confusion Matrix",
                          "confusion_matrix_final.png")

    # 8) Screening threshold (favour recall on malignant cases)
    print("\n--- 8) Screening threshold ---")
    screening_threshold = tune_screening_threshold(final_model, train_x, train_y)
    screening_metrics, screening_predictions = evaluate_model(final_model, test_x, test_y, screening_threshold)
    print(f"Threshold chosen on training data for recall >= {SCREENING_RECALL:.0%} :: {screening_threshold}")
    print(f"Test result at that threshold: recall {screening_metrics['recall']:.3f}, "
          f"precision {screening_metrics['precision']:.3f}, missed malignant "
          f"{screening_metrics['missed_malignant']}, false alarms {screening_metrics['false_alarms']}")
    plot_confusion_matrix(test_y, screening_predictions,
                          f"Screening threshold {screening_threshold} - Confusion Matrix",
                          "confusion_matrix_screening.png")

    # 9) Which measurements matter?
    print("\n--- 9) Feature importance (permutation, test set) ---")
    importance = plot_feature_importance(final_model, test_x, test_y, f"Top Features - {final_name}")
    print(importance.head(8).round(4).to_string())

    # 10) Predict for new samples (rows from the held-out test set)
    print("\n--- 10) Predictions for 4 held-out samples ---")
    sample = pd.concat([test_x[test_y == 1].head(2), test_x[test_y == 0].head(2)])
    result = predict_new_samples(final_model, sample)
    result["Actual"] = np.where(test_y[sample.index] == 1, "Malignant", "Benign")
    print(result.to_string())

    # 11) Save the model and metadata, load the model, and test a sample
    print("\n--- 11) Save and load ---")
    save_model(final_model, MODEL_PATH)
    save_model_metadata(final_name, final_model, train_x, len(train_x), final_metrics,
                        screening_threshold, float(final_row["CV_ROC_AUC"]))
    loaded = load_model(MODEL_PATH)
    print(f"Loaded model prediction for first test sample: "
          f"{CLASS_LABELS[int(loaded.predict(test_x.iloc[[0]])[0])]}")

    summary = {
        "final_model": final_name,
        "selection_rule": "best cross-validated ROC-AUC on training data (test set not used)",
        "majority_class_accuracy": float(1 - test_y.mean()),
        "final_model_test": final_metrics,
        "final_model_test_screening_threshold": {**screening_metrics, "threshold": screening_threshold},
        "feature_importance_roc_auc_drop": importance.head(10).to_dict(),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)
    print(f"Metrics saved to {METRICS_PATH}")


########################################################
# Application starter
########################################################

if __name__ == "__main__":
    main()
