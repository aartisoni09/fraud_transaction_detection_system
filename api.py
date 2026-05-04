"""
api.py  —  FraudGuard FastAPI Backend
Columns from dataset: transaction_id, amount, time, location, device,
  is_new_user, transaction_type, prev_transactions, avg_amount, is_night, Class
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os, pickle, json, uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load artefacts ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model_bundle.pkl"), "rb") as f:
    BUNDLE = pickle.load(f)

with open(os.path.join(BASE, "metrics.json")) as f:
    MODEL_METRICS = json.load(f)

RF        = BUNDLE["rf"]
GB        = BUNDLE["gb"]
LR        = BUNDLE["lr"]
SCALER    = BUNDLE["scaler"]
FEATURES  = BUNDLE["features"]
THRESHOLD = BUNDLE["best_threshold"]
LE        = BUNDLE["label_encoders"]
LE_CLASSES= BUNDLE["label_encoder_classes"]

DATA_PATH = os.path.join(BASE, "fraud_dataset_1500.csv")

# ── In-memory stores ───────────────────────────────────────────
SCORED: List[dict] = []          # all scored transactions
CASES:  Dict[str, dict] = {}     # flagged cases

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudGuard API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Schemas ────────────────────────────────────────────────────
class TransactionIn(BaseModel):
    transaction_id:    Optional[str] = None
    amount:            float = Field(..., gt=0, description="Transaction amount (₹)")
    time:              int   = Field(..., ge=0, le=23, description="Hour of day 0-23")
    location:          str   = Field(..., description="Hyderabad|Bangalore|Kolkata|Delhi|Mumbai")
    device:            str   = Field(..., description="Mobile|Laptop|Tablet")
    is_new_user:       int   = Field(..., ge=0, le=1)
    transaction_type:  str   = Field(..., description="Card|UPI|NetBanking")
    prev_transactions: int   = Field(..., ge=0, description="Number of past transactions")
    avg_amount:        float = Field(..., gt=0, description="User's historical average amount")
    is_night:          int   = Field(..., ge=0, le=1, description="1 if night transaction")

class CaseDecision(BaseModel):
    analyst_id: str
    decision:   str = Field(..., description="approve|decline|hold|investigate")
    notes:      Optional[str] = None

class FeedbackIn(BaseModel):
    transaction_id: str
    actual_fraud:   int = Field(..., ge=0, le=1)
    analyst_id:     str
    notes:          Optional[str] = None

# ── Helpers ────────────────────────────────────────────────────
def safe_encode(encoder, value: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return 0

def build_feature_vector(t: TransactionIn) -> np.ndarray:
    loc_enc  = safe_encode(LE["location"],         t.location)
    dev_enc  = safe_encode(LE["device"],            t.device)
    typ_enc  = safe_encode(LE["transaction_type"], t.transaction_type)

    amount_to_avg_ratio = t.amount / (t.avg_amount + 1)
    is_high_amount      = int(t.amount > 70000)           # ~90th pctile of dataset
    is_new_with_high    = int(t.is_new_user == 1 and t.amount > 50000)
    low_history         = int(t.prev_transactions < 5)

    row = [
        t.amount, t.time, t.is_new_user, t.prev_transactions,
        t.avg_amount, t.is_night,
        loc_enc, dev_enc, typ_enc,
        amount_to_avg_ratio, is_high_amount, is_new_with_high, low_history,
    ]
    return np.array([row], dtype=float)

def predict(t: TransactionIn) -> dict:
    vec   = build_feature_vector(t)
    vec_s = SCALER.transform(vec)

    rf_p  = float(RF.predict_proba(vec_s)[0][1])
    gb_p  = float(GB.predict_proba(vec_s)[0][1])
    lr_p  = float(LR.predict_proba(vec_s)[0][1])
    prob  = round(0.45 * rf_p + 0.45 * gb_p + 0.10 * lr_p, 4)

    if prob >= 0.80:
        risk, recommendation = "CRITICAL", "auto_decline"
    elif prob >= THRESHOLD:
        risk, recommendation = "HIGH",     "hold_for_review"
    elif prob >= 0.30:
        risk, recommendation = "MEDIUM",   "monitor"
    else:
        risk, recommendation = "LOW",      "approve"

    amount_to_avg = t.amount / (t.avg_amount + 1)
    signals = {
        "amount_vs_avg_ratio"  : round(min(amount_to_avg / 10, 1.0), 3),
        "new_user_risk"        : float(t.is_new_user),
        "low_history_risk"     : float(t.prev_transactions < 5),
        "night_transaction"    : float(t.is_night),
        "high_amount_flag"     : float(t.amount > 70000),
    }

    return {
        "fraud_probability": prob,
        "risk_level":        risk,
        "recommendation":    recommendation,
        "model_scores": {
            "random_forest":       round(rf_p, 4),
            "gradient_boosting":   round(gb_p, 4),
            "logistic_regression": round(lr_p, 4),
            "ensemble":            prob,
        },
        "signals": signals,
    }

# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "FraudGuard API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RF+GB+LR ensemble",
        "threshold": THRESHOLD,
        "features": FEATURES,
    }

# Score a single transaction
@app.post("/score")
def score(txn: TransactionIn):
    t0    = datetime.now()
    res   = predict(txn)
    latency = round((datetime.now() - t0).total_seconds() * 1000, 2)

    txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "transaction_id":    txn_id,
        "amount":            txn.amount,
        "time":              txn.time,
        "location":          txn.location,
        "device":            txn.device,
        "transaction_type":  txn.transaction_type,
        "is_new_user":       txn.is_new_user,
        "prev_transactions": txn.prev_transactions,
        "avg_amount":        txn.avg_amount,
        "is_night":          txn.is_night,
        "scored_at":         datetime.now().isoformat(),
        "latency_ms":        latency,
        **res,
    }
    SCORED.append(record)

    # Auto-create case for HIGH/CRITICAL
    if res["risk_level"] in ("HIGH", "CRITICAL"):
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        CASES[case_id] = {
            "case_id":        case_id,
            "transaction_id": txn_id,
            "transaction":    txn.model_dump(),
            **res,
            "status":   "open",
            "priority": "urgent" if res["risk_level"] == "CRITICAL" else "high",
            "created_at":  datetime.now().isoformat(),
            "updated_at":  datetime.now().isoformat(),
            "analyst_id": None, "decision": None, "notes": None,
        }
        record["case_id"] = case_id

    return record

# Score batch
@app.post("/score/batch")
def score_batch(transactions: List[TransactionIn]):
    results = []
    for txn in transactions:
        txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
        res    = predict(txn)
        results.append({"transaction_id": txn_id, **res})
    return {"count": len(results), "results": results}

# Cases
@app.get("/cases")
def list_cases(status: Optional[str] = None, priority: Optional[str] = None):
    cases = list(CASES.values())
    if status:   cases = [c for c in cases if c["status"]   == status]
    if priority: cases = [c for c in cases if c["priority"] == priority]
    cases.sort(key=lambda c: c["created_at"], reverse=True)
    return {"total": len(cases), "cases": cases}

@app.get("/cases/{case_id}")
def get_case(case_id: str):
    if case_id not in CASES:
        raise HTTPException(404, f"Case {case_id} not found")
    return CASES[case_id]

@app.patch("/cases/{case_id}")
def update_case(case_id: str, req: CaseDecision):
    if case_id not in CASES:
        raise HTTPException(404, f"Case {case_id} not found")
    CASES[case_id].update({
        "status":     "resolved",
        "analyst_id": req.analyst_id,
        "decision":   req.decision,
        "notes":      req.notes,
        "updated_at": datetime.now().isoformat(),
    })
    return CASES[case_id]

# Feedback
FEEDBACK: List[dict] = []

@app.post("/feedback")
def feedback(req: FeedbackIn):
    entry = {**req.model_dump(), "id": str(uuid.uuid4()), "submitted_at": datetime.now().isoformat()}
    FEEDBACK.append(entry)
    return {"message": "Feedback recorded", "id": entry["id"]}

# Analytics
@app.get("/analytics/summary")
def summary():
    total = len(SCORED)
    if total == 0:
        return {"message": "No transactions scored yet. Use /transactions/load-dataset first."}
    fraud_count = sum(1 for t in SCORED if t["fraud_probability"] >= THRESHOLD)
    risk_dist   = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for t in SCORED:
        risk_dist[t["risk_level"]] = risk_dist.get(t["risk_level"], 0) + 1
    avg_prob = float(np.mean([t["fraud_probability"] for t in SCORED]))
    return {
        "total_transactions": total,
        "flagged_fraud":      fraud_count,
        "fraud_rate":         round(fraud_count / total, 4),
        "avg_fraud_probability": round(avg_prob, 4),
        "risk_distribution":  risk_dist,
        "open_cases":         sum(1 for c in CASES.values() if c["status"] == "open"),
        "resolved_cases":     sum(1 for c in CASES.values() if c["status"] == "resolved"),
        "feedback_count":     len(FEEDBACK),
    }

@app.get("/analytics/model-performance")
def model_perf():
    return {
        "auc_roc":               MODEL_METRICS["auc_roc"],
        "threshold":             MODEL_METRICS["best_threshold"],
        "classification_report": MODEL_METRICS["classification_report"],
        "confusion_matrix":      MODEL_METRICS["confusion_matrix"],
        "feature_importance":    MODEL_METRICS["feature_importance"],
        "training_samples":      MODEL_METRICS["train_size"],
        "test_samples":          MODEL_METRICS["test_size"],
        "fraud_in_dataset":      MODEL_METRICS["fraud_count"],
        "total_in_dataset":      MODEL_METRICS["total_count"],
    }

@app.get("/transactions/recent")
def recent(limit: int = 100):
    txns = list(reversed(SCORED[-limit:]))
    return {"count": len(txns), "transactions": txns}

# Load entire dataset into scored store (for demo)
@app.get("/transactions/load-dataset")
def load_dataset():
    global SCORED, CASES
    SCORED.clear()
    CASES.clear()

    df = pd.read_csv(DATA_PATH)
    for _, row in df.iterrows():
        txn = TransactionIn(
            transaction_id   = str(row["transaction_id"]),
            amount           = float(row["amount"]),
            time             = int(row["time"]),
            location         = str(row["location"]),
            device           = str(row["device"]),
            is_new_user      = int(row["is_new_user"]),
            transaction_type = str(row["transaction_type"]),
            prev_transactions= int(row["prev_transactions"]),
            avg_amount       = float(row["avg_amount"]),
            is_night         = int(row["is_night"]),
        )
        res    = predict(txn)
        record = {
            "transaction_id":    str(row["transaction_id"]),
            "amount":            float(row["amount"]),
            "time":              int(row["time"]),
            "location":          str(row["location"]),
            "device":            str(row["device"]),
            "transaction_type":  str(row["transaction_type"]),
            "is_new_user":       int(row["is_new_user"]),
            "prev_transactions": int(row["prev_transactions"]),
            "avg_amount":        float(row["avg_amount"]),
            "is_night":          int(row["is_night"]),
            "actual_class":      int(row["Class"]),
            "scored_at":         datetime.now().isoformat(),
            "latency_ms":        0.0,
            **res,
        }
        SCORED.append(record)

        if res["risk_level"] in ("HIGH", "CRITICAL"):
            case_id = f"CASE-{row['transaction_id']}"
            CASES[case_id] = {
                "case_id":        case_id,
                "transaction_id": str(row["transaction_id"]),
                **res,
                "amount":    float(row["amount"]),
                "location":  str(row["location"]),
                "device":    str(row["device"]),
                "status":    "open",
                "priority":  "urgent" if res["risk_level"] == "CRITICAL" else "high",
                "created_at":  datetime.now().isoformat(),
                "updated_at":  datetime.now().isoformat(),
                "analyst_id": None, "decision": None, "notes": None,
            }

    return {
        "message":       f"Loaded {len(df)} transactions from dataset",
        "scored":        len(SCORED),
        "cases_created": len(CASES),
    }

# Dropdown options (for Streamlit form)
@app.get("/options")
def options():
    return {
        "locations":         LE_CLASSES["location"],
        "devices":           LE_CLASSES["device"],
        "transaction_types": LE_CLASSES["transaction_type"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
