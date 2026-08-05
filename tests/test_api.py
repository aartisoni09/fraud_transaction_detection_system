"""
tests/test_api.py — API endpoint tests for FraudGuard.
"""

import pytest


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_root(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "FraudGuard API"
        assert data["status"] == "running"

    def test_health(self, client):
        """Test health endpoint returns model info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "threshold" in data
        assert "features" in data


class TestScoring:
    """Test transaction scoring endpoints."""

    def test_score_valid_transaction(self, client, sample_transaction):
        """Test scoring a valid transaction."""
        response = client.post("/score", json=sample_transaction)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= data["fraud_probability"] <= 1
        assert data["transaction_id"] == "TXN-TEST001"

    def test_score_auto_generates_id(self, client, sample_transaction):
        """Test that missing transaction_id is auto-generated."""
        payload = {**sample_transaction}
        del payload["transaction_id"]
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        assert response.json()["transaction_id"].startswith("TXN-")

    def test_score_batch(self, client, sample_transaction):
        """Test batch scoring."""
        response = client.post("/score/batch", json=[sample_transaction])
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    def test_score_invalid_amount(self, client, sample_transaction):
        """Test validation rejects invalid amount."""
        payload = {**sample_transaction, "amount": -100}
        response = client.post("/score", json=payload)
        assert response.status_code == 422


class TestCases:
    """Test case management endpoints."""

    def test_list_cases(self, client):
        """Test listing cases."""
        response = client.get("/cases")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "cases" in data

    def test_get_nonexistent_case(self, client):
        """Test 404 for non-existent case."""
        response = client.get("/cases/CASE-NONEXISTENT")
        assert response.status_code == 404


class TestFeedback:
    """Test feedback endpoints."""

    def test_submit_feedback(self, client):
        """Test submitting analyst feedback."""
        payload = {
            "transaction_id": "TXN-TEST001",
            "actual_fraud": 0,
            "analyst_id": "analyst_test",
            "notes": "Test feedback",
        }
        response = client.post("/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback recorded"
        assert "id" in data


class TestDataset:
    """Test dataset loading endpoint."""

    def test_load_dataset_is_post(self, client):
        """Test that load-dataset uses POST method."""
        # GET should return 405 Method Not Allowed
        response = client.get("/transactions/load-dataset")
        assert response.status_code == 405

    def test_load_dataset(self, client):
        """Test loading the dataset via POST."""
        response = client.post("/transactions/load-dataset")
        assert response.status_code == 200
        data = response.json()
        assert "scored" in data
        assert data["scored"] > 0


class TestAnalytics:
    """Test analytics endpoints."""

    def test_summary(self, client):
        """Test analytics summary (requires data to be loaded)."""
        response = client.get("/analytics/summary")
        assert response.status_code == 200

    def test_model_performance(self, client):
        """Test model performance metrics."""
        response = client.get("/analytics/model-performance")
        assert response.status_code == 200
        data = response.json()
        assert "auc_roc" in data
        assert "confusion_matrix" in data


class TestOptions:
    """Test dropdown options endpoint."""

    def test_options(self, client):
        """Test that options returns valid dropdown values."""
        response = client.get("/options")
        assert response.status_code == 200
        data = response.json()
        assert "locations" in data
        assert "devices" in data
        assert "transaction_types" in data
