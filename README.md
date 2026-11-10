# Breast Cancer Detection using Machine Learning & Pipeline

This project demonstrates an **end-to-end machine learning workflow** for classifying breast tumours as **Benign or Malignant** from 30 measurements of cell nuclei. It compares **Logistic Regression, a Support Vector Machine (SVM) and Random Forest**, picks the final model with cross-validation, and treats **missed malignant tumours** as the mistake that matters most.

It follows **industrial best practices** by:

- Putting **imputation and scaling inside a Pipeline**, so they are fitted on training data only (no data leakage)
- Using **stratified splitting** to keep the same malignant share in both sets
- Giving every algorithm the **same cross-validation folds and a similar tuning effort**
- Choosing the final model with **cross-validation on training data only**, so the test set is never used to pick a winner
- Tracking **recall on malignant cases** and counting missed cancers, not only accuracy
- Adding a **screening threshold** that trades a few false alarms for fewer missed cancers
- Saving the trained pipeline with **Joblib** and reusing it from a separate **prediction script**
- Saving plots, tables and metrics to files for **reproducibility**

---

## Dependencies

Requires **Python 3.9+** (tested on Python 3.12). Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Dataset Information

**Source:** `breast_cancer.csv`, the Breast Cancer Wisconsin (Diagnostic) dataset as shipped with scikit-learn (569 patients, 30 features, no missing values, no duplicate rows)

**Features (30):** for each tumour, ten measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given three ways:

| Group | Meaning | Example column |
|-------|---------|----------------|
| mean | Average over the cell nuclei | `mean radius` |
| error | Standard error of the measurement | `radius error` |
| worst | Mean of the three largest values | `worst radius` |

**Target:** the column `target` uses **0 = malignant** and **1 = benign**. The script flips it so that **Malignant = 1 is the positive class**. That way precision, recall and F1 all describe malignant cases, which are the ones we must not miss.

- **Benign:** 357 patients (62.7%)
- **Malignant:** 212 patients (37.3%)

An "always predict Benign" model is already **62.7% accurate**.

**Things to know about this data:**

- Many features are nearly copies of each other (for example `mean radius` and `mean perimeter` correlate at 0.998). 21 of the 900 feature pairs have a correlation above 0.9.
- This is a small, clean, well-known benchmark. Results on it are very high for almost any sensible model, so the model comparison is more about careful method than about big differences.

---

## Workflow

### Data Preparation

- Load the CSV and check the target (0 or 1 only) and that every feature is a number
- Flip the target so Malignant = 1
- Check missing values and duplicates (none found)
- Explore class balance, feature correlations and the strongest features

### Train-Test Split

- Split into **70% train (398 patients)** and **30% test (171 patients, 64 of them malignant)**
- Use **stratified sampling** so both sets keep the same malignant share

### Pipeline Construction

- **Step 1:** `SimpleImputer(strategy="median")` (not needed for this clean file, but it makes the saved model safe if a future measurement has a gap)
- **Step 2:** `StandardScaler`
- **Step 3:** the classifier

### Models Compared

| Model | Tuning (5-fold CV on training data) |
|-------|-------------------------------------|
| Logistic Regression | `C` in 0.01, 0.1, 1, 10, 100 |
| SVM (RBF kernel) | `C` in 0.1, 1, 10 and `gamma` in scale, 0.01 |
| Random Forest (300 trees) | `max_depth` in None, 5, 10 and `min_samples_leaf` in 1, 3 |

### Model Selection

- The final model has the best **cross-validated ROC-AUC**, rounded to 3 decimals so that tiny differences count as ties
- Ties are broken by cross-validated **recall on malignant cases**
- The test set is evaluated **once**, after the choice is made

### Screening Threshold

- A prediction is "Malignant" when its probability is at or above a threshold. The default is 0.5.
- For screening, a lower threshold is chosen **from training-data cross-validation** so that at least **98% of malignant cases** are caught. Among thresholds that reach this, the highest one is used, to keep false alarms low.

### Model Saving & Loading

- Save the full pipeline with **Joblib** (`joblib.dump`)
- Save a **metadata file** (expected columns, training ranges, label meaning, both thresholds, library versions, metrics)
- Load both in a separate script (`joblib.load`) and score new samples without retraining

---

## Running the Project

```bash
python breast_cancer_detection.py
```

The run takes about 30 seconds. Expected output (shortened):

```text
Dataset Shape :: (569, 30)
Class 0 (Benign) :: 357 rows (62.7%)
Class 1 (Malignant) :: 212 rows (37.3%)
Train_x Shape ::  (398, 30)
Test_x Shape  ::  (171, 30)

  Logistic Regression  C=0.1                  CV AUC 0.9935 (+/- 0.0052) | recall 0.925
  SVM (RBF)            C=10, gamma=0.01       CV AUC 0.9942 (+/- 0.0052) | recall 0.946
  Random Forest        max_depth=5, ...       CV AUC 0.9864 (+/- 0.0072) | recall 0.918

Final model (best CV ROC-AUC) :: SVM (RBF) (CV AUC 0.9942)
...
Model saved to outputs/breast_cancer_model.joblib
```

**Generated files** (in `outputs/`):

```text
outputs/
├── figures/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── confusion_matrix_final.png
│   ├── confusion_matrix_screening.png
│   └── feature_importance.png
├── model_comparison.csv
├── metrics.json
├── model_metadata.json
├── held_out_samples.csv          # the 171 test patients, with their true diagnosis
└── breast_cancer_model.joblib
```

---

## Results

All numbers below are from `random_state=42`. The test set has 171 patients: 107 benign and 64 malignant. **One missed patient changes recall by 1.6 percentage points and accuracy by 0.6**, so small gaps between models are not meaningful.

**Model comparison (test set, threshold 0.5):**

| Model | CV ROC-AUC | Train Acc | Test Acc | Precision | Recall | F1 | Test ROC-AUC | Missed malignant | False alarms |
|-------|-----------|-----------|----------|-----------|--------|----|--------------|------------------|--------------|
| Always predict "Benign" | - | - | 0.626 | - | 0.000 | - | 0.500 | 64 | 0 |
| Logistic Regression | 0.994 | 0.982 | 0.982 | 1.000 | 0.953 | 0.976 | 0.999 | 3 | 0 |
| **SVM (RBF) (selected)** | **0.994** | 0.987 | 0.977 | 1.000 | 0.938 | 0.968 | 0.998 | 4 | 0 |
| Random Forest | 0.986 | 0.987 | 0.965 | 1.000 | 0.906 | 0.951 | 0.996 | 6 | 0 |

Selected settings: SVM with `C=10`, `gamma=0.01`.

**Selected model at the screening threshold (0.14):**

| Setting | Threshold | Recall | Precision | Missed malignant | False alarms |
|---------|-----------|--------|-----------|------------------|--------------|
| Default | 0.50 | 0.938 | 1.000 | 4 | 0 |
| Screening | 0.14 | **0.984** | 0.955 | **1** | 3 |

```text
Default threshold 0.50                Screening threshold 0.14
                 Pred B   Pred M                       Pred B   Pred M
Actual Benign      107        0       Actual Benign      104        3
Actual Malignant     4       60       Actual Malignant     1       63
```

**Feature importance:** shuffling any single feature barely changes the model (the largest drop in ROC-AUC is 0.007, for `worst texture`). This is because many features carry the same information, so the model can fall back on the others. It does **not** mean the features are useless. Use `outputs/figures/eda_overview.png` for the features most correlated with malignancy: `worst concave points` (0.79), `worst perimeter` (0.78) and `mean concave points` (0.78).

**Key findings:**

1. **All three models perform almost the same.** Cross-validated ROC-AUC is 0.994, 0.994 and 0.986, and the test results differ by one or two patients. The SVM is selected because its cross-validated recall on malignant cases (0.946) is the highest of the three, which is the tie-break rule. On the test set Logistic Regression happens to miss one fewer patient (3 against 4), and that gap is within noise.
2. **The simple model is just as good.** Logistic Regression is far easier to explain than an SVM, which is worth weighing in any medical setting.
3. **The default threshold misses malignant tumours.** At 0.5 the selected model misses 4 of 64. At the screening threshold it misses 1 of 64, at the cost of 3 false alarms among 107 benign patients.
4. **Few false alarms, so false negatives are the weak point.** No model produced a false alarm at the default threshold. All their errors were missed malignant cases, which is the more serious kind.
5. **Test results are close to the cross-validation results.** Train accuracy (0.987) and test accuracy (0.977) are near each other, so there is no sign of serious overfitting.

**Limits:** 569 patients from one source is a small sample, and the dataset is a standard benchmark where almost every model scores highly. This project is **not for clinical use**.

---

## Visualizations

- **Exploratory overview:** `outputs/figures/eda_overview.png`
- **Correlation heatmap:** `outputs/figures/correlation_heatmap.png`
- **ROC curves:** `outputs/figures/roc_curves.png`
- **Confusion matrices:** `outputs/figures/confusion_matrix_final.png`, `outputs/figures/confusion_matrix_screening.png`
- **Feature importance:** `outputs/figures/feature_importance.png`

Set `SHOW_PLOTS = True` at the top of the script to also open the plots in a window.

---

## Model Persistence with Joblib (Train Once, Use Many Times)

**The idea:** training needs the full dataset and takes time. Scoring a new sample should take milliseconds and should not need either. So the trained model is written to a file with `joblib.dump`, and any other program can bring it back with `joblib.load` and call `predict`.

```text
 TRAINING (run once)                          PREDICTION (run any time)
 breast_cancer_detection.py                   predict_diagnosis.py
 ┌──────────────────────┐                     ┌──────────────────────────┐
 │ breast_cancer.csv    │                     │ new_samples.csv          │
 │ tune + compare models│                     │ load model + metadata    │
 │ joblib.dump(model)   │ ──► .joblib  ──►    │ check columns and ranges │
 │ write metadata.json  │ ──► .json    ──►    │ predict_proba            │
 └──────────────────────┘                     └──────────────────────────┘
```

**What is stored:**

- `outputs/breast_cancer_model.joblib`: the **whole pipeline** (fitted imputer, fitted scaler and the classifier), so new samples get exactly the same preparation as the training data
- `outputs/model_metadata.json`: the 30 expected column names, the value ranges seen in training, the label meaning, both thresholds (0.5 and 0.14), library versions and the test metrics

**Score new samples (no training, no dataset needed):**

```bash
python predict_diagnosis.py
```

The file name is optional: with no argument it scores `new_samples.csv` (4 patients taken from the held-out test set). Train first (`python breast_cancer_detection.py`) so the `outputs/` folder with the model exists.

```text
   MalignancyProbability Prediction     Actual
0                  1.000  Malignant  Malignant
1                  0.946  Malignant  Malignant
2                  0.076     Benign     Benign
3                  0.005     Benign     Benign
```

Use the lower screening threshold, or a threshold of your own, or score all 171 held-out patients at once:

```bash
python predict_diagnosis.py --screening
python predict_diagnosis.py --threshold 0.3
python predict_diagnosis.py outputs/held_out_samples.csv --output scored_samples.csv
```

**What the prediction script checks:** missing columns, non-numeric values, empty values, values outside the training range (warning), and a scikit-learn version different from the one that trained the model (warning).

**Rules for using saved models:**

1. **Same library version:** load the model with the scikit-learn version that trained it.
2. **Trusted files only:** Joblib files can run code when opened, so load only files you created yourself.
3. **Retrain when data changes:** a saved model does not learn. With new patient data, rerun `breast_cancer_detection.py` and check the results again.
4. **Git:** `.joblib` files are in `.gitignore`. After cloning, run `python breast_cancer_detection.py` once to create the model.

---

## Sample Prediction (from Python code)

```python
import pandas as pd
from breast_cancer_detection import load_model

model = load_model("outputs/breast_cancer_model.joblib")

samples = pd.read_csv("new_samples.csv").drop(columns="Actual")
probability = model.predict_proba(samples)[:, 1]   # probability of Malignant
print(probability.round(3))
print((probability >= 0.5).astype(int))             # 1 = Malignant, 0 = Benign
```

> **Disclaimer:** This is an educational project on a public benchmark dataset. It must **not** be used for diagnosis or any medical decision.

---

## Author

**Prathmesh Girme**  
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
