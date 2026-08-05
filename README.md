# 🛡️ FraudGuard — Real-Time Fraud Detection System

[![CI](https://github.com/Raghuvaranlokati/fraud_transaction_detection_system/actions/workflows/ci.yml/badge.svg)](https://github.com/Raghuvaranlokati/fraud_transaction_detection_system/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

A production-grade fraud detection system built with **FastAPI**, **Streamlit**, and an **ensemble ML model** (Random Forest + Gradient Boosting + Logistic Regression).

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Docker Setup](#-docker-setup)
- [Running Tests](#-running-tests)
- [API Documentation](#-api-documentation)
- [Streamlit Pages](#-streamlit-pages)
- [ML Model](#-ml-model)
- [Security](#-security)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)

---

## ✨ Features

- **Real-time fraud scoring** — Score transactions via REST API with sub-50ms latency
- **Ensemble ML model** — RF (45%) + GBM (45%) + LR (10%) with configurable thresholds
- **Interactive dashboard** — 6-page Streamlit UI with KPIs, charts, and case management
- **SQLite persistence** — Data survives server restarts (no more in-memory loss)
- **API key authentication** — Optional `X-API-Key` header for protected endpoints
- **Configurable CORS** — Restricted origins instead of wildcard `*`
- **Environment-based config** — All settings via `.env` file or environment variables
- **Docker support** — One-command startup with `docker compose up`
- **CI/CD pipeline** — GitHub Actions for automated testing
- **Comprehensive tests** — pytest test suite for all API endpoints

---

## 🏗️ Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Streamlit UI    │  HTTP   │  FastAPI Backend  │
│  (port 8501)     │────────▶│  (port 8000)      │
│                  │         │                   │
│  📊 Dashboard    │         │  /score           │
│  🔍 Scoring      │         │  /cases           │
│  📋 Cases        │         │  /analytics       │
│  📈 Performance  │         │  /feedback        │
│  🔔 Alerts       │         │                   │
│  📁 Explorer     │         │  ┌─────────────┐  │
│                  │         │  │ SQLite DB   │  │
└──────────────────┘         │  └─────────────┘  │
                             │  ┌─────────────┐  │
                             │  │ ML Ensemble │  │
                             │  │ RF+GB+LR    │  │
                             │  └─────────────┘  │
                             └──────────────────┘
```

---

## 📦 Prerequisites

- **Python 3.11+** ([download](https://www.python.org/downloads/))
- **pip** (included with Python)
- **Git** ([download](https://git-scm.com/))
- *Optional:* **Docker** & **Docker Compose** for containerized setup

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Raghuvaranlokati/fraud_transaction_detection_system.git
cd fraud_transaction_detection_system

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env to customize (all defaults work out of the box)
```

### 3. Train the Model (first time only)

```bash
python train_model.py
```

This creates `model_bundle.pkl` and `metrics.json`.

### 4. Start the Application

**Terminal 1 — API Backend:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run app.py --server.port 8501
```

### 5. Open the Dashboard

1. Open **http://localhost:8501**
2. Click **"📥 Load Dataset"** in the sidebar
3. Explore the dashboard! 🎉

---

## ⚙️ Configuration

All configuration is managed via environment variables or a `.env` file. See [`.env.example`](.env.example) for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | Backend API URL (used by Streamlit) |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |
| `API_KEY` | *(empty — auth disabled)* | API key for authentication |
| `ALLOWED_ORIGINS` | `http://localhost:8501,http://localhost:3000` | CORS allowed origins (comma-separated) |
| `DATABASE_URL` | `fraudguard.db` | SQLite database file path |
| `DATASET_PATH` | `fraud_dataset_1500.csv` | Fraud dataset CSV path |
| `MODEL_BUNDLE_PATH` | `model_bundle.pkl` | Model bundle pickle path |
| `METRICS_PATH` | `metrics.json` | Model metrics JSON path |
| `CRITICAL_THRESHOLD` | `0.80` | Fraud probability for CRITICAL risk |
| `MEDIUM_THRESHOLD` | `0.30` | Fraud probability for MEDIUM risk |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## 🐳 Docker Setup

### Using Docker Compose (recommended)

```bash
# Build and start both API and UI
docker compose up --build

# With custom API key
API_KEY=my-secret-key docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501
- Swagger docs: http://localhost:8000/docs

### Using Docker directly

```bash
# Build API image
docker build --target api -t fraudguard-api .

# Build UI image
docker build --target ui -t fraudguard-ui .

# Run API
docker run -p 8000:8000 -e API_KEY=my-key fraudguard-api

# Run UI
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 fraudguard-ui
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with short traceback
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_api.py::TestScoring -v

# Run config tests
pytest tests/test_config.py -v
```

---

## 📡 API Documentation

Interactive Swagger UI available at: **http://localhost:8000/docs**

### Key Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | ❌ | Service info |
| `GET` | `/health` | ❌ | Health check + model info |
| `POST` | `/score` | ✅ | Score a single transaction |
| `POST` | `/score/batch` | ✅ | Score multiple transactions |
| `POST` | `/transactions/load-dataset` | ✅ | Load & score the CSV dataset |
| `GET` | `/transactions/recent` | ✅ | Get recent scored transactions |
| `GET` | `/cases` | ✅ | List fraud cases (filterable) |
| `GET` | `/cases/{id}` | ✅ | Get a specific case |
| `PATCH` | `/cases/{id}` | ✅ | Update case with analyst decision |
| `POST` | `/feedback` | ✅ | Submit ground-truth feedback |
| `GET` | `/analytics/summary` | ✅ | KPIs and statistics |
| `GET` | `/analytics/model-performance` | ✅ | Model metrics |
| `GET` | `/options` | ❌ | Dropdown values for forms |

### Authentication

When `API_KEY` is set, include the key in request headers:

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/score -X POST -H "Content-Type: application/json" -d '{
  "amount": 50000,
  "time": 2,
  "location": "Delhi",
  "device": "Mobile",
  "is_new_user": 1,
  "transaction_type": "UPI",
  "prev_transactions": 0,
  "avg_amount": 5000,
  "is_night": 1
}'
```

---

## 🖥️ Streamlit Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | KPI cards, risk distribution donut, probability histogram, location & device charts, live transaction table |
| 🔍 **Score Transaction** | Manual form → fraud gauge, model breakdown bar chart, signal bars |
| 📋 **Case Management** | Review HIGH/CRITICAL cases, assign analyst, record decision |
| 📈 **Model Performance** | AUC-ROC, confusion matrix, feature importance, training stats |
| 🔔 **Alert Feed** | Real-time HIGH/CRITICAL alerts with actual labels |
| 📁 **Data Explorer** | Filter by risk/location/device/amount, scatter plots, hourly trend, CSV download |

---

## 🤖 ML Model

**Ensemble = Random Forest (45%) + Gradient Boosting (45%) + Logistic Regression (10%)**

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `amount` | Raw | Transaction amount (₹) |
| `time` | Raw | Hour of day (0-23) |
| `is_new_user` | Raw | New user flag (0/1) |
| `prev_transactions` | Raw | Historical transaction count |
| `avg_amount` | Raw | User's historical average |
| `is_night` | Raw | Night transaction flag (0/1) |
| `location` | Encoded | City (label encoded) |
| `device` | Encoded | Device type (label encoded) |
| `transaction_type` | Encoded | Card/UPI/NetBanking (label encoded) |
| `amount_to_avg_ratio` | Derived | `amount / (avg_amount + 1)` |
| `is_high_amount` | Derived | Top 10% amount flag |
| `is_new_with_high_amt` | Derived | New user + high amount |
| `low_history` | Derived | < 5 past transactions |

### Risk Thresholds

| Probability | Level | Action |
|-------------|-------|--------|
| ≥ 80% | 🔴 **CRITICAL** | `auto_decline` |
| ≥ threshold (from training) | 🟠 **HIGH** | `hold_for_review` |
| ≥ 30% | 🔵 **MEDIUM** | `monitor` |
| < 30% | 🟢 **LOW** | `approve` |

All thresholds are configurable via environment variables.

---

## 🔒 Security

| Feature | Status |
|---------|--------|
| API Key Authentication | ✅ Optional via `API_KEY` env var |
| CORS Origin Restriction | ✅ Configurable via `ALLOWED_ORIGINS` |
| Input Validation | ✅ Pydantic models with constraints |
| SQL Injection Prevention | ✅ Parameterized queries |
| Environment-based Config | ✅ No hardcoded secrets |

---

## 🚀 Deployment

### Render

The project includes a `render.yaml` for [Render](https://render.com/) deployment:

1. Push to GitHub
2. Connect to Render
3. Set environment variables (`API_KEY`, `ALLOWED_ORIGINS`)
4. Deploy automatically

### Manual Deployment

```bash
# Production API
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Production UI
streamlit run app.py --server.port 8501 --server.headless true
```

---

## 📁 Project Structure

```
fraud_transaction_detection_system/
├── .github/
│   └── workflows/
│       └── ci.yml                 ← GitHub Actions CI pipeline
├── fraudguard/                    ← Core Python package
│   ├── __init__.py
│   ├── config.py                  ← Centralized configuration (pydantic-settings)
│   ├── database.py                ← SQLite persistence layer
│   ├── auth.py                    ← API key authentication
│   └── logging_config.py          ← Structured logging
├── tests/                         ← Test suite
│   ├── __init__.py
│   ├── conftest.py                ← Shared fixtures
│   ├── test_api.py                ← API endpoint tests
│   └── test_config.py             ← Configuration tests
├── api.py                         ← FastAPI backend
├── app.py                         ← Streamlit frontend
├── train_model.py                 ← ML model training script
├── fraud_dataset_1500.csv         ← Dataset (1500 transactions)
├── model_bundle.pkl               ← Trained model bundle
├── metrics.json                   ← Model metrics
├── requirements.txt               ← Pinned dependencies
├── Dockerfile                     ← Multi-target Docker image
├── docker-compose.yml             ← One-command startup
├── render.yaml                    ← Render deployment config
├── .env.example                   ← Environment variable template
├── .gitignore                     ← Git ignore rules
├── runtime.txt                    ← Python version for Render
└── README.md                      ← This file
```

---

## 📄 Dataset

**`fraud_dataset_1500.csv`** — 1,500 financial transactions with 11 columns:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier |
| `amount` | Transaction amount (₹) |
| `time` | Hour of day (0-23) |
| `location` | City name |
| `device` | Device type |
| `is_new_user` | New user flag (0/1) |
| `transaction_type` | Payment method |
| `prev_transactions` | Past transaction count |
| `avg_amount` | Historical average |
| `is_night` | Night flag (0/1) |
| `Class` | Ground truth (0=legit, 1=fraud) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📝 License

This project is for educational and demonstration purposes.
