# 🛡️ FraudGuard — Real-Time Fraud Detection System

Built on your `fraud_dataset_1500.csv`

## Dataset columns used
`transaction_id | amount | time | location | device | is_new_user | transaction_type | prev_transactions | avg_amount | is_night | Class`

---

## 📁 Files
```
fraudguard/
├── fraud_dataset_1500.csv   ← your dataset (put it here)
├── train_model.py           ← trains the ML ensemble → model_bundle.pkl + metrics.json
├── api.py                   ← FastAPI backend  (port 8000)
├── app.py                   ← Streamlit frontend (port 8501)
├── requirements.txt
└── README.md
```

---

## 🚀 Run (3 steps)

```bash
# Step 1 — install
pip install -r requirements.txt

# Step 2 — train model (only once)
python train_model.py

# Step 3a — start API (Terminal 1)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Step 3b — start Streamlit (Terminal 2)
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** then click **"Load Dataset"** in the sidebar.

---

## 🤖 ML Model

**Ensemble = RF (45%) + GBM (45%) + Logistic Regression (10%)**

Features trained on:
| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `time` | Hour of day (0-23) |
| `is_new_user` | New user flag |
| `prev_transactions` | Historical transaction count |
| `avg_amount` | User's historical average |
| `is_night` | Night transaction flag |
| `location` | City (encoded) |
| `device` | Device type (encoded) |
| `transaction_type` | Card / UPI / NetBanking (encoded) |
| `amount_to_avg_ratio` | Derived: amount / avg_amount |
| `is_high_amount` | Derived: top 10% amount flag |
| `is_new_with_high_amt` | Derived: new user + high amount |
| `low_history` | Derived: < 5 past transactions |

---

## 📡 Key API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `POST` | `/score` | Score one transaction → returns fraud_probability, risk_level, signals |
| `POST` | `/score/batch` | Score multiple transactions |
| `GET`  | `/transactions/load-dataset` | Load all 1500 rows from CSV and score them |
| `GET`  | `/transactions/recent` | Get recently scored transactions |
| `GET`  | `/cases` | List all fraud cases |
| `PATCH`| `/cases/{id}` | Analyst decision: approve / decline / hold |
| `POST` | `/feedback` | Submit ground-truth label back |
| `GET`  | `/analytics/summary` | KPIs: counts, fraud rate, risk distribution |
| `GET`  | `/analytics/model-performance` | AUC, confusion matrix, feature importance |
| `GET`  | `/options` | Valid dropdown values for the form |
| `GET`  | `/docs` | Swagger UI |

---

## 🖥️ Streamlit Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPI cards, risk donut, probability histogram, location & device charts, live table |
| 🔍 Score Transaction | Manual form → gauge, model breakdown, signal bars |
| 📋 Case Management | Review HIGH/CRITICAL cases, assign analyst, record decision |
| 📈 Model Performance | AUC, confusion matrix, feature importance, training stats |
| 🔔 Alert Feed | Real-time HIGH/CRITICAL alerts with actual label shown |
| 📁 Data Explorer | Filter by risk/location/device/amount, scatter, hourly trend, download CSV |

---

## ⚙️ Risk Thresholds

| Probability | Level | Action |
|-------------|-------|--------|
| ≥ 80% | 🔴 CRITICAL | auto_decline |
| ≥ threshold (from training) | 🟠 HIGH | hold_for_review |
| ≥ 30% | 🔵 MEDIUM | monitor |
| < 30% | 🟢 LOW | approve |
