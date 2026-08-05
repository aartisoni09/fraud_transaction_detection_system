"""
tests/conftest.py — Shared test fixtures for FraudGuard.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables BEFORE importing the app
os.environ["DATABASE_URL"] = ":memory:"
os.environ["API_KEY"] = ""
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["DATASET_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fraud_dataset_1500.csv",
)
os.environ["MODEL_BUNDLE_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_bundle.pkl",
)
os.environ["METRICS_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics.json",
)


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    # Import here so env vars are set first
    from api import app
    from fraudguard.database import init_db
    
    init_db()
    
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_transaction():
    """Return a sample valid transaction payload."""
    return {
        "transaction_id": "TXN-TEST001",
        "amount": 25000.0,
        "time": 14,
        "location": "Mumbai",
        "device": "Mobile",
        "is_new_user": 0,
        "transaction_type": "Card",
        "prev_transactions": 30,
        "avg_amount": 5000.0,
        "is_night": 0,
    }


@pytest.fixture
def high_risk_transaction():
    """Return a transaction likely to be flagged as high risk."""
    return {
        "transaction_id": "TXN-RISKY001",
        "amount": 450000.0,
        "time": 2,
        "location": "Delhi",
        "device": "Tablet",
        "is_new_user": 1,
        "transaction_type": "NetBanking",
        "prev_transactions": 0,
        "avg_amount": 1000.0,
        "is_night": 1,
    }
