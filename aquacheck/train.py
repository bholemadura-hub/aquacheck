"""AquaCheck - train and compare five ML models on the Water Potability dataset.

Run:  python train.py
Creates: water_quality_model.joblib, results.csv, meta.json, charts in /charts
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
os.makedirs("charts", exist_ok=True)

# ---- Step 1: load data -------------------------------------------------
df = pd.read_csv("water_potability.csv")
FEATURES = [c for c in df.columns if c != "Potability"]
print("Rows:", len(df), "| Missing values:", int(df.isna().sum().sum()))

X, y = df[FEATURES], df["Potability"]

# ---- Step 3: 80/20 split (stratified keeps the class ratio equal) --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print("Train:", len(X_train), "| Test:", len(X_test))

# ---- Step 2: preprocessing lives INSIDE the pipeline ----------------------
# Median is learned from the training set only, so the test set never leaks.
def make(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])

models = {
    "Logistic Regression": make(LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": make(DecisionTreeClassifier(max_depth=6, random_state=SEED)),
    "Random Forest": make(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    "SVM": make(SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
    "KNN": make(KNeighborsClassifier(n_neighbors=7)),
}

# ---- Steps 4 & 5: train and evaluate -------------------------------------
rows, fitted = [], {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
    })
    fitted[name] = pipe

results = pd.DataFrame(rows).set_index("Model").round(4)
print(results)
results.to_csv("results.csv")

# ---- Step 6: save the best model (by F1, since classes are imbalanced) ----
best_name = results["F1-score"].idxmax()
best = fitted[best_name]
joblib.dump(best, "water_quality_model.joblib")
print("Best model:", best_name)

# Dataset statistics used by the website for slider ranges / defaults
train_imputed = pd.DataFrame(
    best.named_steps["impute"].transform(X_train), columns=FEATURES)
meta = {
    "best_model": best_name,
    "features": FEATURES,
    "n_total": len(df), "n_train": len(X_train), "n_test": len(X_test),
    "n_missing": int(df.isna().sum().sum()),
    "ranges": {f: [float(df[f].min()), float(df[f].max()), float(train_imputed[f].median())]
               for f in FEATURES},
}
json.dump(meta, open("meta.json", "w"), indent=2)

# ---- Charts ---------------------------------------------------------------
# 1. model comparison
ax = results.plot(kind="bar", figsize=(9, 4.5), ylim=(0, 1), rot=15)
ax.set_title("Model comparison on the 20% test set")
plt.tight_layout(); plt.savefig("charts/model_comparison.png", dpi=150); plt.close()

# 2. confusion matrix of the best model
cm = confusion_matrix(y_test, best.predict(X_test))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1], ["Not Potable", "Potable"]); ax.set_yticks([0, 1], ["Not Potable", "Potable"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{best_name}")
plt.tight_layout(); plt.savefig("charts/confusion_matrix.png", dpi=150); plt.close()

# 3. feature importance (Random Forest always trained, so we can show it)
rf = fitted["Random Forest"].named_steps["model"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(6, 4))
imp.plot(kind="barh", ax=ax, color="#1f77b4")
ax.set_title("Which parameters matter most? (Random Forest)")
plt.tight_layout(); plt.savefig("charts/feature_importance.png", dpi=150); plt.close()

# 4. class balance
fig, ax = plt.subplots(figsize=(4, 3.5))
y.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], rot=0)
ax.set_xticklabels(["Not Potable", "Potable"]); ax.set_title("Samples in dataset")
plt.tight_layout(); plt.savefig("charts/class_balance.png", dpi=150); plt.close()

# 5. correlation heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
fig.colorbar(im); ax.set_title("Correlation between columns")
plt.tight_layout(); plt.savefig("charts/correlation.png", dpi=150); plt.close()
print("Done.")
