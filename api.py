"""
api.py  —  FraudGuard FastAPI Backend

Real-time fraud detection API using an ensemble ML model (RF + GB + LR).
Supports transaction scoring, case management, analyst feedback, and analytics.

Configuration via environment variables or .env file — see .env.example.
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fraudguard.config import get_settings
from fraudguard.logging_config import setup_logging, get_logger
from fraudguard.auth import verify_api_key
from fraudguard import database as db

# ── Initialize logging ────────────────────────────────────────
setup_logging()
logger = get_logger("fraudguard.api")

# ── Load settings ─────────────────────────────────────────────
settings = get_settings()

# ── Load model artefacts ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    """Resolve a path relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE, path)


model_path = _resolve_path(settings.model_bundle_path)
metrics_path = _resolve_path(settings.metrics_path)

logger.info("Loading model bundle from: %s", model_path)
with open(model_path, "rb") as f:
    BUNDLE = pickle.load(f)

with open(metrics_path) as f:
    MODEL_METRICS = json.load(f)

RF = BUNDLE["rf"]
GB = BUNDLE["gb"]
LR = BUNDLE["lr"]
SCALER = BUNDLE["scaler"]
FEATURES = BUNDLE["features"]
THRESHOLD = BUNDLE["best_threshold"]
LE = BUNDLE["label_encoders"]
LE_CLASSES = BUNDLE["label_encoder_classes"]

DATA_PATH = _resolve_path(settings.dataset_path)

logger.info("Model loaded successfully. Threshold: %.3f", THRESHOLD)

# ── Initialize database ───────────────────────────────────────
db.init_db()
logger.info("Database initialized")

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="FraudGuard API",
    description="Real-time fraud detection for financial transactions",
    version="2.0.0",
)

# CORS — restricted origins from config (Issue #2 fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS origins: %s", settings.cors_origins)
if settings.auth_enabled:
    logger.info("API key authentication: ENABLED")
else:
    logger.info("API key authentication: DISABLED (set API_KEY to enable)")


# ── Pydantic Schemas ──────────────────────────────────────────
class TransactionIn(BaseModel):
    """Input schema for a financial transaction to be scored.

    All fields correspond to the dataset columns used for model training.
    """

    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount (₹)")
    time: int = Field(..., ge=0, le=23, description="Hour of day 0-23")
    location: str = Field(
        ..., description="Hyderabad|Bangalore|Kolkata|Delhi|Mumbai"
    )
    device: str = Field(..., description="Mobile|Laptop|Tablet")
    is_new_user: int = Field(..., ge=0, le=1)
    transaction_type: str = Field(..., description="Card|UPI|NetBanking")
    prev_transactions: int = Field(
        ..., ge=0, description="Number of past transactions"
    )
    avg_amount: float = Field(
        ..., gt=0, description="User's historical average amount"
    )
    is_night: int = Field(
        ..., ge=0, le=1, description="1 if night transaction"
    )


class CaseDecision(BaseModel):
    """Input schema for an analyst's decision on a fraud case."""

    analyst_id: str
    decision: str = Field(
        ..., description="approve|decline|hold|investigate"
    )
    notes: Optional[str] = None


class FeedbackIn(BaseModel):
    """Input schema for analyst ground-truth feedback."""

    transaction_id: str
    actual_fraud: int = Field(..., ge=0, le=1)
    analyst_id: str
    notes: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────
def safe_encode(encoder, value: str) -> int:
    """Safely encode a categorical value using the fitted label encoder.

    Args:
        encoder: Fitted sklearn LabelEncoder instance.
        value: The categorical value to encode.

    Returns:
        Encoded integer value, or 0 if encoding fails.
    """
    try:
        return int(encoder.transform([value])[0])
    except Exception as e:
        # Issue #11 fix: log warning instead of silently returning 0
        logger.warning(
            "Encoding failed for value '%s': %s. Defaulting to 0.",
            value,
            str(e),
        )
        return 0


def build_feature_vector(t: TransactionIn) -> np.ndarray:
    """Build a feature vector from a transaction for model prediction.

    Args:
        t: The transaction input.

    Returns:
        Numpy array of shape (1, n_features) ready for model prediction.
    """
    loc_enc = safe_encode(LE["location"], t.location)
    dev_enc = safe_encode(LE["device"], t.device)
    typ_enc = safe_encode(LE["transaction_type"], t.transaction_type)

    amount_to_avg_ratio = t.amount / (t.avg_amount + 1)
    is_high_amount = int(t.amount > 70000)  # ~90th percentile of dataset
    is_new_with_high = int(t.is_new_user == 1 and t.amount > 50000)
    low_history = int(t.prev_transactions < 5)

    row = [
        t.amount,
        t.time,
        t.is_new_user,
        t.prev_transactions,
        t.avg_amount,
        t.is_night,
        loc_enc,
        dev_enc,
        typ_enc,
        amount_to_avg_ratio,
        is_high_amount,
        is_new_with_high,
        low_history,
    ]
    return np.array([row], dtype=float)


def predict(t: TransactionIn) -> dict:
    """Run the ensemble model on a transaction and return prediction results.

    The ensemble combines Random Forest (45%), Gradient Boosting (45%),
    and Logistic Regression (10%).

    Args:
        t: The transaction to score.

    Returns:
        Dictionary with fraud_probability, risk_level, recommendation,
        model_scores, and risk signals.
    """
    vec = build_feature_vector(t)
    vec_s = SCALER.transform(vec)

    rf_p = float(RF.predict_proba(vec_s)[0][1])
    gb_p = float(GB.predict_proba(vec_s)[0][1])
    lr_p = float(LR.predict_proba(vec_s)[0][1])
    prob = round(0.45 * rf_p + 0.45 * gb_p + 0.10 * lr_p, 4)

    # Issue #9 fix: configurable thresholds from a single source
    if prob >= settings.critical_threshold:
        risk, recommendation = "CRITICAL", "auto_decline"
    elif prob >= THRESHOLD:
        risk, recommendation = "HIGH", "hold_for_review"
    elif prob >= settings.medium_threshold:
        risk, recommendation = "MEDIUM", "monitor"
    else:
        risk, recommendation = "LOW", "approve"

    amount_to_avg = t.amount / (t.avg_amount + 1)
    signals = {
        "amount_vs_avg_ratio": round(min(amount_to_avg / 10, 1.0), 3),
        "new_user_risk": float(t.is_new_user),
        "low_history_risk": float(t.prev_transactions < 5),
        "night_transaction": float(t.is_night),
        "high_amount_flag": float(t.amount > 70000),
    }

    return {
        "fraud_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation,
        "model_scores": {
            "random_forest": round(rf_p, 4),
            "gradient_boosting": round(gb_p, 4),
            "logistic_regression": round(lr_p, 4),
            "ensemble": prob,
        },
        "signals": signals,
    }


# ── Public Routes (no auth required) ──────────────────────────


@app.get("/", tags=["Status"])
def root():
    """Root endpoint — returns service info and available endpoints."""
    return {
        "service": "FraudGuard API",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health", tags=["Status"])
def health():
    """Health check — returns model readiness info."""
    return {
        "status": "healthy",
        "model": "RF+GB+LR ensemble",
        "threshold": THRESHOLD,
        "features": FEATURES,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Protected Routes (auth required when API_KEY is set) ──────


@app.post("/score", tags=["Scoring"], dependencies=[Depends(verify_api_key)])
def score(txn: TransactionIn):
    """Score a single transaction for fraud probability.

    Returns the fraud probability, risk level, recommendation,
    individual model scores, and risk signal breakdown.
    Auto-creates a case for HIGH/CRITICAL risk transactions.
    """
    t0 = datetime.now()
    res = predict(txn)
    latency = round((datetime.now() - t0).total_seconds() * 1000, 2)

    txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "transaction_id": txn_id,
        "amount": txn.amount,
        "time": txn.time,
        "location": txn.location,
        "device": txn.device,
        "transaction_type": txn.transaction_type,
        "is_new_user": txn.is_new_user,
        "prev_transactions": txn.prev_transactions,
        "avg_amount": txn.avg_amount,
        "is_night": txn.is_night,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency,
        **res,
    }

    # Persist to database (Issue #4 fix)
    db.save_scored(record)

    # Auto-create case for HIGH/CRITICAL
    if res["risk_level"] in ("HIGH", "CRITICAL"):
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        case = {
            "case_id": case_id,
            "transaction_id": txn_id,
            "transaction": txn.model_dump(),
            **res,
            "status": "open",
            "priority": (
                "urgent" if res["risk_level"] == "CRITICAL" else "high"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "analyst_id": None,
            "decision": None,
            "notes": None,
        }
        db.save_case(case)
        record["case_id"] = case_id
        logger.info(
            "Case %s created for %s (risk: %s, prob: %.4f)",
            case_id,
            txn_id,
            res["risk_level"],
            res["fraud_probability"],
        )

    return record


@app.post(
    "/score/batch", tags=["Scoring"], dependencies=[Depends(verify_api_key)]
)
def score_batch(transactions: List[TransactionIn]):
    """Score multiple transactions in a single request.

    Returns the count and individual results for each transaction.
    """
    results = []
    for txn in transactions:
        txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
        res = predict(txn)
        results.append({"transaction_id": txn_id, **res})
    return {"count": len(results), "results": results}


@app.get("/cases", tags=["Cases"], dependencies=[Depends(verify_api_key)])
def list_cases(
    status: Optional[str] = None, priority: Optional[str] = None
):
    """List fraud cases with optional filters.

    Args:
        status: Filter by case status (open/resolved).
        priority: Filter by priority (urgent/high).
    """
    cases = db.list_cases(status=status, priority=priority)
    return {"total": len(cases), "cases": cases}


@app.get(
    "/cases/{case_id}", tags=["Cases"], dependencies=[Depends(verify_api_key)]
)
def get_case(case_id: str):
    """Get a specific fraud case by its ID.

    Raises:
        HTTPException: 404 if the case is not found.
    """
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    return case


@app.patch(
    "/cases/{case_id}", tags=["Cases"], dependencies=[Depends(verify_api_key)]
)
def update_case(case_id: str, req: CaseDecision):
    """Update a fraud case with an analyst's decision.

    Sets the case status to 'resolved' and records the analyst's
    decision, ID, and notes.

    Raises:
        HTTPException: 404 if the case is not found.
    """
    updated = db.update_case(
        case_id,
        {
            "status": "resolved",
            "analyst_id": req.analyst_id,
            "decision": req.decision,
            "notes": req.notes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if not updated:
        raise HTTPException(404, f"Case {case_id} not found")
    logger.info(
        "Case %s resolved by %s: %s", case_id, req.analyst_id, req.decision
    )
    return updated


@app.post("/feedback", tags=["Feedback"], dependencies=[Depends(verify_api_key)])
def feedback(req: FeedbackIn):
    """Submit analyst ground-truth feedback for a transaction.

    This feedback is used to track model accuracy and can be used
    for future model retraining.
    """
    entry = {
        **req.model_dump(),
        "id": str(uuid.uuid4()),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    db.save_feedback(entry)
    logger.info(
        "Feedback recorded for %s by %s (actual_fraud=%d)",
        req.transaction_id,
        req.analyst_id,
        req.actual_fraud,
    )
    return {"message": "Feedback recorded", "id": entry["id"]}


@app.get(
    "/analytics/summary",
    tags=["Analytics"],
    dependencies=[Depends(verify_api_key)],
)
def summary():
    """Get a summary of all scored transactions.

    Returns KPIs including total transactions, fraud count, fraud rate,
    risk distribution, and case/feedback counts.
    """
    all_scored = db.get_all_scored()
    total = len(all_scored)

    if total == 0:
        return {
            "message": "No transactions scored yet. Use POST /transactions/load-dataset first."
        }

    fraud_count = sum(
        1 for t in all_scored if t["fraud_probability"] >= THRESHOLD
    )
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for t in all_scored:
        risk_dist[t["risk_level"]] = risk_dist.get(t["risk_level"], 0) + 1

    avg_prob = float(np.mean([t["fraud_probability"] for t in all_scored]))

    return {
        "total_transactions": total,
        "flagged_fraud": fraud_count,
        "fraud_rate": round(fraud_count / total, 4),
        "avg_fraud_probability": round(avg_prob, 4),
        "risk_distribution": risk_dist,
        "open_cases": db.count_cases(status="open"),
        "resolved_cases": db.count_cases(status="resolved"),
        "feedback_count": db.count_feedback(),
    }


@app.get(
    "/analytics/model-performance",
    tags=["Analytics"],
    dependencies=[Depends(verify_api_key)],
)
def model_perf():
    """Get model performance metrics from training.

    Returns AUC-ROC, classification report, confusion matrix,
    feature importance, and training statistics.
    """
    return {
        "auc_roc": MODEL_METRICS["auc_roc"],
        "threshold": MODEL_METRICS["best_threshold"],
        "classification_report": MODEL_METRICS["classification_report"],
        "confusion_matrix": MODEL_METRICS["confusion_matrix"],
        "feature_importance": MODEL_METRICS["feature_importance"],
        "training_samples": MODEL_METRICS["train_size"],
        "test_samples": MODEL_METRICS["test_size"],
        "fraud_in_dataset": MODEL_METRICS["fraud_count"],
        "total_in_dataset": MODEL_METRICS["total_count"],
    }


@app.get(
    "/transactions/recent",
    tags=["Transactions"],
    dependencies=[Depends(verify_api_key)],
)
def recent(limit: int = 100):
    """Get the most recently scored transactions.

    Args:
        limit: Maximum number of transactions to return (default 100).
    """
    txns = db.get_recent_scored(limit=limit)
    return {"count": len(txns), "transactions": txns}


# Issue #8 fix: Changed from GET to POST for state-changing operation
@app.post(
    "/transactions/load-dataset",
    tags=["Transactions"],
    dependencies=[Depends(verify_api_key)],
)
def load_dataset():
    """Load and score the entire fraud dataset from CSV.

    Clears existing scored transactions and cases, then scores all
    transactions from the configured dataset CSV file.

    Raises:
        HTTPException: 404 if the dataset file is not found.
        HTTPException: 500 if there is an error processing the dataset.
    """
    # Issue #10 fix: proper error handling for missing CSV
    if not os.path.exists(DATA_PATH):
        logger.error("Dataset file not found: %s", DATA_PATH)
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {DATA_PATH}. "
            "Ensure fraud_dataset_1500.csv exists in the project root.",
        )

    try:
        # Clear existing data
        db.clear_scored()
        db.clear_cases()
        logger.info("Cleared existing data, loading dataset from: %s", DATA_PATH)

        df = pd.read_csv(DATA_PATH)
        scored_records = []
        cases_created = 0

        for _, row in df.iterrows():
            txn = TransactionIn(
                transaction_id=str(row["transaction_id"]),
                amount=float(row["amount"]),
                time=int(row["time"]),
                location=str(row["location"]),
                device=str(row["device"]),
                is_new_user=int(row["is_new_user"]),
                transaction_type=str(row["transaction_type"]),
                prev_transactions=int(row["prev_transactions"]),
                avg_amount=float(row["avg_amount"]),
                is_night=int(row["is_night"]),
            )
            res = predict(txn)
            record = {
                "transaction_id": str(row["transaction_id"]),
                "amount": float(row["amount"]),
                "time": int(row["time"]),
                "location": str(row["location"]),
                "device": str(row["device"]),
                "transaction_type": str(row["transaction_type"]),
                "is_new_user": int(row["is_new_user"]),
                "prev_transactions": int(row["prev_transactions"]),
                "avg_amount": float(row["avg_amount"]),
                "is_night": int(row["is_night"]),
                "actual_class": int(row["Class"]),
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": 0.0,
                **res,
            }
            scored_records.append(record)

            if res["risk_level"] in ("HIGH", "CRITICAL"):
                case_id = f"CASE-{row['transaction_id']}"
                case = {
                    "case_id": case_id,
                    "transaction_id": str(row["transaction_id"]),
                    **res,
                    "amount": float(row["amount"]),
                    "location": str(row["location"]),
                    "device": str(row["device"]),
                    "status": "open",
                    "priority": (
                        "urgent"
                        if res["risk_level"] == "CRITICAL"
                        else "high"
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "analyst_id": None,
                    "decision": None,
                    "notes": None,
                }
                db.save_case(case)
                cases_created += 1

        # Batch save for performance
        db.save_scored_batch(scored_records)

        logger.info(
            "Dataset loaded: %d transactions scored, %d cases created",
            len(scored_records),
            cases_created,
        )

        return {
            "message": f"Loaded {len(df)} transactions from dataset",
            "scored": len(scored_records),
            "cases_created": cases_created,
        }

    except FileNotFoundError:
        logger.error("Dataset file not found: %s", DATA_PATH)
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {DATA_PATH}",
        )
    except Exception as e:
        logger.error("Error loading dataset: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dataset: {str(e)}",
        )


@app.get("/options", tags=["Utilities"])
def options():
    """Get valid dropdown values for the Streamlit form.

    Returns lists of valid locations, devices, and transaction types
    that the model was trained on.
    """
    return {
        "locations": LE_CLASSES["location"],
        "devices": LE_CLASSES["device"],
        "transaction_types": LE_CLASSES["transaction_type"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )
