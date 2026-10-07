"""
train.py
────────
HOW THE MODEL IS TRAINED:

Input:  processed_nslkdd.csv (3 classes: low=normal, medium=probe, high=attack)
Output: models/ids_model.pkl, models/feature_names.pkl, models/metrics.json

The model: Random Forest with 200 decision trees + probability calibration

HOW RANDOM FOREST WORKS:
  - 200 decision trees are built, each from a random subset of training data
  - Each tree learns its own set of rules (e.g. "if serror_rate > 1.5 AND
    diff_srv_rate > 1.0 → probably probe")
  - For a new record, all 200 trees vote. Majority vote = prediction.
  - Confidence = fraction of trees that voted for the winning class

WHY CALIBRATION (CalibratedClassifierCV):
  - Raw Random Forest vote fractions cluster at fixed values (e.g. always 76%)
  - Isotonic calibration remaps these to proper 0-100% probabilities
  - This makes the 50%/80% confidence thresholds meaningful

WHY class_weight="balanced":
  - low: 6009 rows, medium: 3000 rows, high: 3000 rows
  - Without balancing, the model ignores minority classes
  - "balanced" makes each class equally important during training
"""

import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from config import MODEL_FILE, FEATURES_FILE, METRICS_FILE, MODELS_DIR, RANDOM_STATE

os.makedirs(MODELS_DIR, exist_ok=True)
PROJECT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(PROJECT_DIR, "processed_nslkdd.csv")

print("=" * 60)
print("  Hybrid IDS — Model Training  (3-class + calibrated)")
print("=" * 60)

# Step 1: Load
print(f"\n[1/6] Loading: {PROCESSED_CSV}")
if not os.path.exists(PROCESSED_CSV):
    raise FileNotFoundError("Run option 1 first.")
df = pd.read_csv(PROCESSED_CSV)
print(f"      Rows: {len(df):,}  Columns: {df.shape[1]}")

# Step 2: Prepare X (features) and y (labels)
print("\n[2/6] Preparing features and target ...")
drop_meta = ["label","attack_group","dataset_split","difficulty"]
df_m = df.drop([c for c in drop_meta if c in df.columns], axis=1)
X = df_m.drop("severity", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
y = df_m["severity"]
FEATURE_NAMES = list(X.columns)
unique = sorted(y.unique())
print(f"      Features: {len(FEATURE_NAMES)}  Classes: {unique}")
for cls, cnt in y.value_counts().items():
    print(f"        {cls:8s}: {cnt:,}  ({cnt/len(y)*100:.1f}%)")
if "medium" not in unique:
    print("\n  ERROR: medium class missing. Re-run option 1.")
    raise SystemExit(1)
print("  ✅ 3 classes confirmed: low (normal) / medium (probe) / high (attack)")

# Step 3: 80/20 train/test split — stratified to keep class ratios equal
print("\n[3/6] Splitting 80% train / 20% test ...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"      Train: {len(X_tr):,}  Test: {len(X_te):,}")

# Step 4: Train Random Forest + probability calibration
print("\n[4/6] Training Random Forest (200 trees) + probability calibration ...")
print("      Takes 2-5 minutes ...")
base = RandomForestClassifier(
    n_estimators=200,       # 200 independent decision trees
    class_weight="balanced", # equal importance for all 3 classes
    random_state=RANDOM_STATE,
    n_jobs=-1               # use all CPU cores
)
# Isotonic calibration: remaps vote fractions to proper probability values
model = CalibratedClassifierCV(base, method="isotonic", cv=3)
model.fit(X_tr, y_tr)
print(f"      Classes in model: {list(model.classes_)}")

# Step 5: Evaluate on the held-out 20% test set
print("\n[5/6] Evaluating on test set ...")
y_pred  = model.predict(X_te)
y_proba = model.predict_proba(X_te)
acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
cm   = confusion_matrix(y_te, y_pred, labels=list(model.classes_)).tolist()
print(f"\n  Accuracy:{acc:.4f}  Precision:{prec:.4f}  Recall:{rec:.4f}  F1:{f1:.4f}")
print(f"\n{classification_report(y_te, y_pred, zero_division=0)}")
mp = y_proba.max(axis=1)
print(f"  Probability spread: Min:{mp.min():.2%} Max:{mp.max():.2%} "
      f"Mean:{mp.mean():.2%} Std:{mp.std():.2%}")

# Step 6: Save model artefacts
print("\n[6/6] Saving model artefacts ...")
joblib.dump(model, MODEL_FILE)
joblib.dump(FEATURE_NAMES, FEATURES_FILE)
with open(METRICS_FILE, "w") as f:
    json.dump({
        "train_size":    len(X_tr),
        "test_size":     len(X_te),
        "feature_count": len(FEATURE_NAMES),
        "classes":       list(model.classes_),
        "model_type":    "RandomForest(200) + CalibratedClassifierCV(isotonic, cv=3)",
        "models": {"RandomForest": {
            "accuracy":           round(acc, 6),
            "precision_weighted": round(prec, 6),
            "recall_weighted":    round(rec, 6),
            "f1_weighted":        round(f1, 6),
            "confusion_matrix":   cm,
        }},
        "best_model": "RandomForest",
    }, f, indent=2)
print(f"  Model   : {MODEL_FILE}")
print(f"  Features: {FEATURES_FILE}")
print(f"  Metrics : {METRICS_FILE}")
print("\n" + "=" * 60)
print(f"  Training complete!  Classes: {list(model.classes_)}")
print("=" * 60)
