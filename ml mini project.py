# titanic_full_pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# --------- 1) Load dataset ----------
def load_titanic():
    try:
        # try seaborn builtin dataset (convenient)
        df = sns.load_dataset('titanic')
        print("Loaded Titanic dataset from seaborn.")
    except Exception:
        # fallback: look for 'titanic.csv' in current folder
        if os.path.exists('titanic.csv'):
            df = pd.read_csv('titanic.csv')
            print("Loaded Titanic dataset from 'titanic.csv'.")
        else:
            raise FileNotFoundError(
                "Could not load the Titanic dataset. Install seaborn or place 'titanic.csv' in the script folder."
            )
    return df

df = load_titanic()
print("Initial shape:", df.shape)
print(df.head())

# --------- 2) Quick feature selection & cleaning ----------
# We'll use a practical subset of features commonly used:
# 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone', 'who'
# Target: 'survived'

# If dataset uses different column names, adapt accordingly
expected_cols = ['survived','pclass','sex','age','sibsp','parch','fare','embarked','alone','who','deck','embark_town']
# Keep only columns that exist
use_cols = [c for c in expected_cols if c in df.columns]
data = df[use_cols].copy()

# Some datasets (seaborn) use 'alone' and 'who' — keep them, but we'll mostly use core features.
# Show missing value counts
print("\nMissing values per column:\n", data.isna().sum())

# Create/derive features:
# - FamilySize = sibsp + parch
data['family_size'] = data.get('sibsp', 0).fillna(0) + data.get('parch', 0).fillna(0)

# - Fill/transform 'alone' if present (True/False -> 1/0)
if 'alone' in data.columns:
    data['alone'] = data['alone'].map({True:1, False:0}).fillna(0)

# - If 'embarked' has NaN, keep and impute later
# - If 'fare' has zeros or NaN, impute later

# Keep a clean subset
features = ['pclass','sex','age','fare','embarked','family_size','alone']
features = [f for f in features if f in data.columns]
target = 'survived'
data = data[features + [target]].copy()

# Drop rows where target is missing
data = data.dropna(subset=[target])
print("\nData after selecting features:", data.shape)

# --------- 3) Train-test split ----------
X = data[features]
y = data[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# --------- 4) Preprocessing pipeline ----------
# Numeric and categorical columns
numeric_features = [c for c in ['age','fare','family_size'] if c in X_train.columns]
categorical_features = [c for c in ['pclass','sex','embarked','alone'] if c in X_train.columns]

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline with RandomForest
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# --------- 5) Train model ----------
print("\nTraining model ...")
clf.fit(X_train, y_train)
print("Training completed.")

# --------- 6) Evaluate on test set ----------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf.named_steps['classifier'], "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

print("\n--- Test set performance ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
if roc_auc is not None:
    print(f"ROC AUC  : {roc_auc:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve (if probabilities present)
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# --------- 7) Cross-validation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"\n5-fold CV Accuracy scores: {cv_scores}")
print(f"CV Accuracy mean: {cv_scores.mean():.4f}  std: {cv_scores.std():.4f}")

# --------- 8) Feature importance (approx; requires mapping back after one-hot)
print("\nFeature importance (approx from RandomForest):")
# We extract feature names after preprocessing
ohe = None
feature_names = []
# numeric names first
feature_names += numeric_features
# get categorical names from OneHotEncoder
cat_idx = None
try:
    # access fitted ColumnTransformer
    cat_transformer = clf.named_steps['preprocessor'].named_transformers_['cat']
    ohe = cat_transformer.named_steps['onehot']
    # build names
    ohe_cols = ohe.get_feature_names_out(categorical_features)
    feature_names += list(ohe_cols)
except Exception:
    # fallback
    feature_names = None

importances = clf.named_steps['classifier'].feature_importances_
if feature_names is not None and len(importances) == len(feature_names):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(fi.head(12))
    fi.plot(kind='bar', figsize=(8,4))
    plt.title("Feature importances")
    plt.show()
else:
    print("Could not map feature importances to feature names, but raw importances:")
    print(importances)

# --------- 9) Save the trained pipeline ----------
model_filename = "titanic_model.joblib"
joblib.dump(clf, model_filename)
print(f"\nSaved full pipeline (preprocessor + model) to {model_filename}")

# --------- 10) Quick demo: predict for a single example ----------
example = {
    'pclass': 3,
    'sex': 'male',
    'age': 30,
    'fare': 7.25,
    'embarked': 'S',
    'family_size': 0,
    'alone': 1
}
# keep only features that exist in this run
example = {k:v for k,v in example.items() if k in X.columns}
example_df = pd.DataFrame([example])
pred_example = clf.predict(example_df)[0]
prob_example = clf.predict_proba(example_df)[0][1] if y_proba is not None else None
print("\nExample passenger prediction -> Survived (1) / Died (0):", pred_example)
if prob_example is not None:
    print(f"Survival probability: {prob_example:.3f}")

# End of script
