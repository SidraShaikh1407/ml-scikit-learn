# diabetes_pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from scipy.stats import randint

# ---------- 1) Load dataset (auto-download if not present) ----------
csv_local = "diabetes.csv"
csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

if os.path.exists(csv_local):
    df = pd.read_csv(csv_local)
    print(f"Loaded local file: {csv_local}")
else:
    print("Downloading dataset from URL...")
    df = pd.read_csv(csv_url)
    df.to_csv(csv_local, index=False)
    print(f"Downloaded and saved as: {csv_local}")

print("\nColumns:", df.columns.tolist())
print(df.head())

# ---------- 2) Quick EDA ----------
print("\nDataset shape:", df.shape)
print("\nOutcome value counts:\n", df['Outcome'].value_counts())
print("\nMissing values per column:\n", df.isna().sum())

# ---------- 3) Clean / Feature adjustments ----------
# In this dataset, zeros in certain medical features represent missing values.
zeros_as_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in zeros_as_missing:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

print("\nMissing counts after zero->NaN replacement:\n", df.isna().sum())

# ---------- 4) Split X/y ----------
target = 'Outcome'
X = df.drop(columns=[target])
y = df[target].astype(int)

# train-test split (stratify by outcome for balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- 5) Preprocessing pipeline ----------
# Numeric features (all columns here are numeric)
numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # fill missing with median
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# ---------- 6) Build two pipelines: Logistic Regression & RandomForest ----------
pipe_logreg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))
])

pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# ---------- 7) Train baseline models ----------
print("\nTraining Logistic Regression...")
pipe_logreg.fit(X_train, y_train)
print("Training Random Forest...")
pipe_rf.fit(X_train, y_train)

# ---------- 8) Evaluate function ----------
def evaluate_model(pipeline, X_test, y_test, plot_roc=False, name="model"):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n--- Evaluation: {name} ---")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)
    if roc_auc is not None:
        print("ROC AUC  :", roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    if plot_roc and y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.show()

# Evaluate baseline
evaluate_model(pipe_logreg, X_test, y_test, plot_roc=True, name="LogisticRegression")
evaluate_model(pipe_rf, X_test, y_test, plot_roc=True, name="RandomForest")

# ---------- 9) Cross-validation (5-fold stratified) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logreg_cv_scores = cross_val_score(pipe_logreg, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
rf_cv_scores = cross_val_score(pipe_rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print("\nLogReg CV accuracies:", logreg_cv_scores, "mean:", logreg_cv_scores.mean())
print("RandomForest CV accuracies:", rf_cv_scores, "mean:", rf_cv_scores.mean())

# ---------- 10) Hyperparameter tuning for RandomForest (RandomizedSearch) ----------
param_dist = {
    'clf__n_estimators': randint(50, 500),
    'clf__max_depth': [None] + list(range(3, 20)),
    'clf__min_samples_split': randint(2, 11),
    'clf__min_samples_leaf': randint(1, 6),
    'clf__max_features': ['sqrt', 'log2', None]
}

print("\nRunning RandomizedSearchCV for RandomForest (30 iterations)...")
rand_search = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rand_search.fit(X, y)  # we fit on full data with CV for robust hyperparameters
print("\nBest params (RandomForest):", rand_search.best_params_)
print("Best CV score:", rand_search.best_score_)

best_rf = rand_search.best_estimator_

# ---------- 11) Final evaluate best model on hold-out test set ----------
evaluate_model(best_rf, X_test, y_test, plot_roc=True, name="BestRandomForest")

# ---------- 12) Feature importance (from RandomForest) ----------
try:
    rf_clf = best_rf.named_steps['clf']
    # build feature names after preprocessing
    # since we used only numeric features, names are the same
    feat_names = X.columns.tolist()
    importances = rf_clf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("\nFeature importances:\n", fi)
    fi.plot(kind='bar', figsize=(8,4))
    plt.title("Feature importances (RandomForest)")
    plt.show()
except Exception as e:
    print("Could not compute feature importances:", e)

# ---------- 13) Save best model pipeline ----------
model_file = "diabetes_model.joblib"
joblib.dump(best_rf, model_file)
print(f"\nSaved best pipeline to: {model_file}")

# ---------- 14) Quick demo prediction ----------
example = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 20,
    'Insulin': 79,
    'BMI': 28.0,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 32
}
example_df = pd.DataFrame([example])
pred = best_rf.predict(example_df)[0]
prob = best_rf.predict_proba(example_df)[0][1] if hasattr(best_rf.named_steps['clf'], "predict_proba") else None
print("\nExample prediction -> 1 = diabetic, 0 = non-diabetic:", pred)
if prob is not None:
    print(f"Predicted probability of diabetes: {prob:.3f}")

# End
