"""
train_model.py
Trains an ensemble ML model on fraud_dataset_1500.csv and saves model_bundle.pkl + metrics.json
Columns: transaction_id, amount, time, location, device, is_new_user,
         transaction_type, prev_transactions, avg_amount, is_night, Class
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("  FraudGuard — ML Model Training")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────────
df = pd.read_csv("fraud_dataset_1500.csv")
print(f"\n✓ Loaded {len(df)} rows")
print(f"  Fraud (Class=1): {df['Class'].sum()}")
print(f"  Legit (Class=0): {(df['Class']==0).sum()}")

# ── 2. Feature Engineering ─────────────────────────────────────
# Encode categoricals
le_location = LabelEncoder()
le_device   = LabelEncoder()
le_txn_type = LabelEncoder()

df["location_enc"]      = le_location.fit_transform(df["location"])
df["device_enc"]        = le_device.fit_transform(df["device"])
df["transaction_type_enc"] = le_txn_type.fit_transform(df["transaction_type"])

# Derived features
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount"] + 1)
df["is_high_amount"]      = (df["amount"] > df["amount"].quantile(0.90)).astype(int)
df["is_new_with_high_amt"]= ((df["is_new_user"] == 1) & (df["amount"] > 50000)).astype(int)
df["low_history"]         = (df["prev_transactions"] < 5).astype(int)

FEATURES = [
    "amount",
    "time",
    "is_new_user",
    "prev_transactions",
    "avg_amount",
    "is_night",
    "location_enc",
    "device_enc",
    "transaction_type_enc",
    "amount_to_avg_ratio",
    "is_high_amount",
    "is_new_with_high_amt",
    "low_history",
]

X = df[FEATURES]
y = df["Class"]

# ── 3. Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")
print(f"  Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

# ── 4. Scale ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 5. Train Models ────────────────────────────────────────────
print("\nTraining models...")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train_s, y_train)
rf_prob = rf.predict_proba(X_test_s)[:, 1]
rf_auc  = roc_auc_score(y_test, rf_prob)
print(f"  Random Forest      AUC: {rf_auc:.4f}")

gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb.fit(X_train_s, y_train)
gb_prob = gb.predict_proba(X_test_s)[:, 1]
gb_auc  = roc_auc_score(y_test, gb_prob)
print(f"  Gradient Boosting  AUC: {gb_auc:.4f}")

lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
lr.fit(X_train_s, y_train)
lr_prob = lr.predict_proba(X_test_s)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)
print(f"  Logistic Regression AUC: {lr_auc:.4f}")

# ── 6. Ensemble ────────────────────────────────────────────────
ensemble_prob = 0.45 * rf_prob + 0.45 * gb_prob + 0.10 * lr_prob
ens_auc = roc_auc_score(y_test, ensemble_prob)
print(f"  Ensemble (best)    AUC: {ens_auc:.4f}")

# Best threshold by F1
thresholds = np.arange(0.05, 0.95, 0.01)
f1s = [f1_score(y_test, (ensemble_prob >= t).astype(int), zero_division=0) for t in thresholds]
best_threshold = float(thresholds[np.argmax(f1s)])
y_pred = (ensemble_prob >= best_threshold).astype(int)
print(f"\n✓ Best Threshold: {best_threshold:.2f}  (F1={max(f1s):.4f})")

# ── 7. Metrics ─────────────────────────────────────────────────
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm     = confusion_matrix(y_test, y_pred).tolist()
fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
prec, rec, _ = precision_recall_curve(y_test, ensemble_prob)

fi_dict = dict(zip(FEATURES, rf.feature_importances_.tolist()))

metrics = {
    "auc_roc"               : round(ens_auc, 4),
    "best_threshold"        : round(best_threshold, 3),
    "classification_report" : report,
    "confusion_matrix"      : cm,
    "roc_curve"             : {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    "pr_curve"              : {"precision": prec.tolist(), "recall": rec.tolist()},
    "feature_importance"    : fi_dict,
    "features"              : FEATURES,
    "train_size"            : int(len(X_train)),
    "test_size"             : int(len(X_test)),
    "fraud_count"           : int(y.sum()),
    "total_count"           : int(len(y)),
}

print("\n" + classification_report(y_test, y_pred, zero_division=0))

# ── 8. Save ────────────────────────────────────────────────────
bundle = {
    "rf": rf,
    "gb": gb,
    "lr": lr,
    "scaler": scaler,
    "features": FEATURES,
    "best_threshold": best_threshold,
    "label_encoders": {
        "location"        : le_location,
        "device"          : le_device,
        "transaction_type": le_txn_type,
    },
    "label_encoder_classes": {
        "location"        : le_location.classes_.tolist(),
        "device"          : le_device.classes_.tolist(),
        "transaction_type": le_txn_type.classes_.tolist(),
    }
}

with open("model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✓ Saved: model_bundle.pkl")
print("✓ Saved: metrics.json")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
