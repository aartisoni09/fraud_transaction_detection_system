"""
app.py  —  FraudGuard Streamlit Frontend
Run: streamlit run app.py --server.port 8501
"""


import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import time

# Wait for API to be ready on cold start
def wait_for_api(retries=5):
    for i in range(retries):
        try:
            r = requests.get(f"{API}/health", timeout=3)
            if r.ok:
                return True
        except:
            time.sleep(2)
    return False

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "https://fraudguard-api.onrender.com"

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp { background-color: #0d1117; }

/* Cards */
.kpi-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.kpi-val  { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
.kpi-lbl  { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }

/* Risk badges */
.badge-CRITICAL { background:#3d0000; border:1.5px solid #ef4444; border-radius:8px; padding:14px; margin:4px 0; }
.badge-HIGH     { background:#2d1a00; border:1.5px solid #f59e0b; border-radius:8px; padding:14px; margin:4px 0; }
.badge-MEDIUM   { background:#00193d; border:1.5px solid #3b82f6; border-radius:8px; padding:14px; margin:4px 0; }
.badge-LOW      { background:#00200f; border:1.5px solid #22c55e; border-radius:8px; padding:14px; margin:4px 0; }

/* Sidebar brand */
.brand { font-size:1.4rem; font-weight:800; color:#f0f6fc; letter-spacing:.5px; }
.brand span { color:#ef4444; }

/* General text */
h1,h2,h3 { color:#f0f6fc !important; }
p, li { color:#c9d1d9; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,#1f6feb,#8957e5);
    color:#fff; border:none; border-radius:8px;
    font-weight:600; padding:.45rem 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────
def api_get(path, params=None):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=6)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_post(path, data):
    try:
        r = requests.post(f"{API}{path}", json=data, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_patch(path, data):
    try:
        r = requests.patch(f"{API}{path}", json=data, timeout=6)
        return r.json() if r.ok else None
    except Exception:
        return None

RISK_COLOR = {"CRITICAL":"#ef4444","HIGH":"#f59e0b","MEDIUM":"#3b82f6","LOW":"#22c55e"}
RISK_EMOJI = {"CRITICAL":"🚨","HIGH":"⚠️","MEDIUM":"🔵","LOW":"✅"}

def dark_chart(fig, height=320):
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#c9d1d9", height=height,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🛡️ Fraud<span>Guard</span></div>', unsafe_allow_html=True)
    st.markdown("Real-Time Detection System")
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🔍 Score Transaction",
        "📋 Case Management",
        "📈 Model Performance",
        "🔔 Alert Feed",
        "📁 Data Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")

    # API status
    health = api_get("/health")
    if health:
        st.success("✅ API Connected")
        st.caption(f"Threshold: {health.get('threshold','—')}")
    else:
        st.error("❌ API Offline")
        st.caption(f"Start: uvicorn api:app --port 8000")

    st.markdown("---")

    if st.button("📥 Load Dataset (1500 txns)"):
        with st.spinner("Scoring all 1500 transactions…"):
            res = api_get("/transactions/load-dataset")
        if res:
            st.success(res.get("message", "Loaded"))
            st.caption(f"Cases opened: {res.get('cases_created', 0)}")
        else:
            st.error("Failed to load — API not ready yet")

# ════════════════════════════════════════════════════════════
# PAGE 1 ── DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Live Fraud Detection Dashboard")

summary = api_get("/analytics/summary")
txns_raw = api_get("/transactions/recent", {"limit": 1500})

# ← ADD THIS CHECK
if not summary or "message" in summary or summary is None:
    st.info("👈 Click **Load Dataset** in the sidebar to populate the dashboard.")
    st.stop()

    # ── KPIs ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, summary["total_transactions"], "Total Transactions",       "#f0f6fc"),
        (c2, summary["flagged_fraud"],       "Flagged as Fraud",         "#ef4444"),
        (c3, f"{summary['fraud_rate']*100:.1f}%", "Fraud Rate",         "#f59e0b"),
        (c4, summary["open_cases"],          "Open Cases",               "#f59e0b"),
        (c5, summary["resolved_cases"],      "Resolved Cases",           "#22c55e"),
    ]
    for col, val, lbl, color in kpis:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        rd = summary["risk_distribution"]
        fig = go.Figure(go.Pie(
            labels=list(rd.keys()), values=list(rd.values()), hole=0.6,
            marker_colors=["#22c55e","#3b82f6","#f59e0b","#ef4444"],
        ))
        fig.update_layout(title="Risk Level Distribution", legend=dict(bgcolor="#161b22"))
        st.plotly_chart(dark_chart(fig), use_container_width=True)

    with col2:
        if txns_raw and txns_raw.get("transactions"):
            df_t = pd.DataFrame(txns_raw["transactions"])
            fig = px.histogram(df_t, x="fraud_probability", nbins=40,
                               color_discrete_sequence=["#8957e5"],
                               title="Fraud Probability Distribution")
            fig.add_vline(x=health.get("threshold", 0.05) if health else 0.05,
                          line_dash="dash", line_color="#ef4444",
                          annotation_text="Threshold", annotation_font_color="#ef4444")
            st.plotly_chart(dark_chart(fig), use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────
    if txns_raw and txns_raw.get("transactions"):
        df_t = pd.DataFrame(txns_raw["transactions"])
        col3, col4 = st.columns(2)

        with col3:
            if "location" in df_t.columns:
                loc_risk = (df_t.groupby("location")["fraud_probability"]
                              .mean().reset_index()
                              .sort_values("fraud_probability", ascending=False))
                fig = px.bar(loc_risk, x="location", y="fraud_probability",
                             color="fraud_probability", color_continuous_scale="RdYlGn_r",
                             title="Avg Fraud Probability by Location")
                st.plotly_chart(dark_chart(fig), use_container_width=True)

        with col4:
            if "transaction_type" in df_t.columns:
                tt_risk = (df_t.groupby("transaction_type")["fraud_probability"]
                             .mean().reset_index())
                fig = px.bar(tt_risk, x="transaction_type", y="fraud_probability",
                             color="fraud_probability", color_continuous_scale="RdYlGn_r",
                             title="Avg Fraud Probability by Transaction Type")
                st.plotly_chart(dark_chart(fig), use_container_width=True)

    # ── Transaction Feed ──────────────────────────────────────
    if txns_raw and txns_raw.get("transactions"):
        st.subheader("🔄 Recent Transaction Feed")
        df_t = pd.DataFrame(txns_raw["transactions"])
        cols = [c for c in ["transaction_id","amount","location","device",
                             "transaction_type","fraud_probability","risk_level",
                             "recommendation","actual_class"] if c in df_t.columns]
        df_show = df_t[cols].head(30).copy()
        df_show["amount"]            = df_show["amount"].apply(lambda x: f"₹{x:,.0f}")
        df_show["fraud_probability"] = df_show["fraud_probability"].apply(lambda x: f"{x:.2%}")

        def color_risk(val):
            colors = {"CRITICAL":"background-color:#3d0000",
                      "HIGH"    :"background-color:#2d1a00",
                      "MEDIUM"  :"background-color:#00193d",
                      "LOW"     :"background-color:#00200f"}
            return colors.get(val, "")

        styled = df_show.style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True, height=360)

# ════════════════════════════════════════════════════════════
# PAGE 2 ── SCORE TRANSACTION
# ════════════════════════════════════════════════════════════
elif page == "🔍 Score Transaction":
    st.title("🔍 Score a Transaction in Real-Time")

    # Fetch options from API
    opts = api_get("/options") or {
        "locations"        : ["Hyderabad","Bangalore","Kolkata","Delhi","Mumbai"],
        "devices"          : ["Laptop","Mobile","Tablet"],
        "transaction_types": ["Card","NetBanking","UPI"],
    }

    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Transaction")
            txn_id  = st.text_input("Transaction ID", value=f"TXN-{int(time.time())}")
            amount  = st.number_input("Amount (₹)", min_value=1.0, max_value=500000.0,
                                      value=25000.0, step=1000.0)
            txn_type= st.selectbox("Transaction Type", opts["transaction_types"])
            location= st.selectbox("Location",         opts["locations"])

        with c2:
            st.subheader("User & Device")
            device      = st.selectbox("Device",     opts["devices"])
            is_new_user = st.radio("New User?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            prev_txns   = st.number_input("Previous Transactions", min_value=0, max_value=500, value=30)
            avg_amount  = st.number_input("User Avg Amount (₹)", min_value=1.0, max_value=200000.0,
                                          value=5000.0, step=500.0)

        with c3:
            st.subheader("Timing")
            hour     = st.slider("Hour of Day", 0, 23, 14)
            is_night = st.checkbox("Night Transaction (auto from hour)", value=(hour >= 22 or hour < 6))
            ratio    = amount / max(avg_amount, 1)
            st.metric("Amount / Avg Ratio", f"{ratio:.2f}",
                      delta="⚠️ High" if ratio > 5 else "Normal")

        submitted = st.form_submit_button("🚀 Score Now", use_container_width=True)

    if submitted:
        payload = {
            "transaction_id"   : txn_id,
            "amount"           : float(amount),
            "time"             : int(hour),
            "location"         : location,
            "device"           : device,
            "is_new_user"      : int(is_new_user),
            "transaction_type" : txn_type,
            "prev_transactions": int(prev_txns),
            "avg_amount"       : float(avg_amount),
            "is_night"         : int(is_night),
        }
        with st.spinner("Scoring…"):
            result = api_post("/score", payload)

        if not result:
            st.error("API error — is the backend running?")
        else:
            risk = result["risk_level"]
            prob = result["fraud_probability"]
            rec  = result["recommendation"]

            st.markdown(f"""
            <div class="badge-{risk}">
                <h2>{RISK_EMOJI[risk]} {risk} RISK — {prob:.1%} Fraud Probability</h2>
                <p><b>Recommendation:</b> {rec.upper().replace('_',' ')} &nbsp;|&nbsp;
                   <b>Latency:</b> {result.get('latency_ms',0):.1f} ms &nbsp;|&nbsp;
                   <b>ID:</b> {result['transaction_id']}</p>
            </div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix":"%","font":{"size":38,"color":"#f0f6fc"}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#c9d1d9"},
                        "bar" :{"color": RISK_COLOR[risk]},
                        "bgcolor":"#161b22",
                        "steps":[
                            {"range":[0,30], "color":"#00200f"},
                            {"range":[30,50],"color":"#00193d"},
                            {"range":[50,80],"color":"#2d1a00"},
                            {"range":[80,100],"color":"#3d0000"},
                        ],
                    },
                    title={"text":"Fraud Probability","font":{"color":"#f0f6fc"}},
                ))
                fig.update_layout(paper_bgcolor="#161b22", height=270)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Model breakdown
                ms  = result.get("model_scores",{})
                names = ["Random Forest","Gradient Boost","Logistic Reg","Ensemble"]
                vals  = [ms.get("random_forest",0), ms.get("gradient_boosting",0),
                         ms.get("logistic_regression",0), ms.get("ensemble",0)]
                fig = go.Figure(go.Bar(
                    x=names, y=vals,
                    marker_color=["#8957e5","#1f6feb","#0ea5e9","#ef4444"],
                    text=[f"{v:.1%}" for v in vals], textposition="outside",
                ))
                fig.update_layout(title="Model Scores", yaxis=dict(range=[0,1.15]))
                st.plotly_chart(dark_chart(fig, 270), use_container_width=True)

            # Signals
            st.subheader("🔎 Risk Signal Breakdown")
            sig = result.get("signals",{})
            sig_df = pd.DataFrame([
                {"Signal": k.replace("_"," ").title(), "Score": v}
                for k, v in sig.items()
            ])
            fig = px.bar(sig_df, x="Signal", y="Score",
                         color="Score", color_continuous_scale="RdYlGn_r",
                         range_y=[0,1.1])
            st.plotly_chart(dark_chart(fig, 260), use_container_width=True)

            if result.get("case_id"):
                st.warning(f"📋 Case auto-created: **{result['case_id']}**")

# ════════════════════════════════════════════════════════════
# PAGE 3 ── CASE MANAGEMENT
# ════════════════════════════════════════════════════════════
elif page == "📋 Case Management":
    st.title("📋 Fraud Case Management")

    col1, col2, col3 = st.columns(3)
    with col1: status_f   = st.selectbox("Status",   ["all","open","resolved"])
    with col2: priority_f = st.selectbox("Priority", ["all","urgent","high"])
    with col3: st.write("")

    params = {}
    if status_f   != "all": params["status"]   = status_f
    if priority_f != "all": params["priority"] = priority_f

    data = api_get("/cases", params)
    if not data or not data.get("cases"):
        st.info("No cases yet. Load the dataset or score a high-risk transaction.")
        st.stop()

    cases = data["cases"]
    st.metric("Cases shown", data["total"])
    st.markdown("---")

    for case in cases[:40]:
        risk = case.get("risk_level","HIGH")
        prob = case.get("fraud_probability",0)
        amt  = case.get("amount",0)
        loc  = case.get("location","—")

        with st.expander(
            f"{RISK_EMOJI[risk]} {case['case_id']}  |  ₹{amt:,.0f}  |  {loc}  "
            f"|  {risk}  ({prob:.1%})  |  {case['status'].upper()}"
        ):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Transaction:** `{case['transaction_id']}`")
                st.markdown(f"**Amount:** ₹{amt:,.0f}")
                st.markdown(f"**Location:** {loc}")
                st.markdown(f"**Device:** {case.get('device','—')}")
                st.markdown(f"**Fraud Probability:** {prob:.1%}")
                st.progress(min(prob, 1.0))
                st.markdown(f"**Created:** {case['created_at'][:19]}")

            with right:
                if case["status"] == "open":
                    analyst  = st.text_input("Analyst ID", "analyst_01", key=f"a_{case['case_id']}")
                    decision = st.selectbox("Decision", ["approve","decline","hold","investigate"],
                                            key=f"d_{case['case_id']}")
                    notes    = st.text_area("Notes", key=f"n_{case['case_id']}", height=70)
                    if st.button("✅ Submit", key=f"s_{case['case_id']}"):
                        upd = api_patch(f"/cases/{case['case_id']}",
                                        {"analyst_id":analyst,"decision":decision,"notes":notes})
                        if upd:
                            st.success(f"Resolved: {decision.upper()}")
                            st.rerun()
                else:
                    st.success(f"✅ **{case.get('decision','—').upper()}**")
                    st.markdown(f"By: {case.get('analyst_id','—')}")
                    if case.get("notes"):
                        st.info(case["notes"])

# ════════════════════════════════════════════════════════════
# PAGE 4 ── MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    perf = api_get("/analytics/model-performance")
    if not perf:
        st.error("Cannot reach API.")
        st.stop()

    report = perf.get("classification_report", {})
    fraud_r = report.get("1", report.get("1.0", {}))

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("AUC-ROC",        f"{perf['auc_roc']:.4f}")
    with c2: st.metric("Precision",       f"{fraud_r.get('precision',0):.4f}")
    with c3: st.metric("Recall",          f"{fraud_r.get('recall',0):.4f}")
    with c4: st.metric("F1-Score",        f"{fraud_r.get('f1-score',0):.4f}")

    col1, col2 = st.columns(2)

    with col1:
        cm = perf.get("confusion_matrix",[[0,0],[0,0]])
        fig = px.imshow(cm, text_auto=True,
                        x=["Pred: Legit","Pred: Fraud"],
                        y=["Actual: Legit","Actual: Fraud"],
                        color_continuous_scale="Blues",
                        title="Confusion Matrix")
        st.plotly_chart(dark_chart(fig, 350), use_container_width=True)

    with col2:
        fi = perf.get("feature_importance", {})
        fi_df = (pd.DataFrame(list(fi.items()), columns=["Feature","Importance"])
                   .sort_values("Importance", ascending=True))
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Viridis",
                     title="Feature Importance (Random Forest)")
        st.plotly_chart(dark_chart(fig, 350), use_container_width=True)

    st.subheader("ℹ️ Training Info")
    st.markdown(f"""
| | |
|---|---|
| **Dataset** | fraud_dataset_1500.csv |
| **Total rows** | {perf['total_in_dataset']:,} |
| **Fraud cases** | {perf['fraud_in_dataset']} ({perf['fraud_in_dataset']/perf['total_in_dataset']*100:.1f}%) |
| **Training samples** | {perf['training_samples']:,} |
| **Test samples** | {perf['test_samples']:,} |
| **Decision threshold** | {perf['threshold']} |
| **Model type** | Ensemble: RF (45%) + GBM (45%) + LR (10%) |
| **AUC-ROC** | {perf['auc_roc']} |
""")

# ════════════════════════════════════════════════════════════
# PAGE 5 ── ALERT FEED
# ════════════════════════════════════════════════════════════
elif page == "🔔 Alert Feed":
    st.title("🔔 High-Risk Alert Feed")

    if st.button("🔄 Refresh"):
        st.rerun()

    txns_raw = api_get("/transactions/recent", {"limit": 1500})
    if not txns_raw or not txns_raw.get("transactions"):
        st.info("Load dataset first.")
        st.stop()

    df = pd.DataFrame(txns_raw["transactions"])
    df_alerts = df[df["risk_level"].isin(["CRITICAL","HIGH"])].sort_values(
        "fraud_probability", ascending=False
    )

    st.metric("🚨 Active Alerts", len(df_alerts))
    st.markdown("---")

    for _, row in df_alerts.iterrows():
        risk  = row["risk_level"]
        cls   = f"badge-{risk}"
        em    = RISK_EMOJI[risk]
        actual= f" | **Actual label: {'🔴 FRAUD' if row.get('actual_class',0)==1 else '🟢 LEGIT'}**" \
                if "actual_class" in row else ""
        st.markdown(f"""
        <div class="{cls}">
            <strong>{em} {risk}</strong> &nbsp;|&nbsp;
            <code>{row.get('transaction_id','—')}</code> &nbsp;|&nbsp;
            <strong>₹{row.get('amount',0):,.0f}</strong> &nbsp;|&nbsp;
            {row.get('location','—')} &nbsp;|&nbsp;
            {row.get('device','—')} &nbsp;|&nbsp;
            {row.get('transaction_type','—')} &nbsp;|&nbsp;
            Prob: <strong>{row.get('fraud_probability',0):.1%}</strong>
            {actual}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 6 ── DATA EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "📁 Data Explorer":
    st.title("📁 Transaction Data Explorer")

    txns_raw = api_get("/transactions/recent", {"limit": 1500})
    if not txns_raw or not txns_raw.get("transactions"):
        st.info("Load dataset first.")
        st.stop()

    df = pd.DataFrame(txns_raw["transactions"])

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_f = st.multiselect("Risk Level",
                                ["LOW","MEDIUM","HIGH","CRITICAL"],
                                default=["LOW","MEDIUM","HIGH","CRITICAL"])
    with c2:
        loc_f = st.multiselect("Location",
                               df["location"].unique().tolist() if "location" in df else [],
                               default=df["location"].unique().tolist() if "location" in df else [])
    with c3:
        dev_f = st.multiselect("Device",
                               df["device"].unique().tolist() if "device" in df else [],
                               default=df["device"].unique().tolist() if "device" in df else [])
    with c4:
        min_a, max_a = int(df["amount"].min()), int(df["amount"].max())
        amt_f = st.slider("Amount (₹)", min_a, max_a, (min_a, max_a))

    mask = (
        df["risk_level"].isin(risk_f) &
        (df["amount"] >= amt_f[0]) & (df["amount"] <= amt_f[1])
    )
    if loc_f and "location" in df: mask = mask & df["location"].isin(loc_f)
    if dev_f and "device"   in df: mask = mask & df["device"].isin(dev_f)
    df_f = df[mask].copy()

    st.metric("Filtered Transactions", len(df_f))

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_f.head(500), x="amount", y="fraud_probability",
                         color="risk_level",
                         color_discrete_map=RISK_COLOR,
                         hover_data=["transaction_id","location","device","transaction_type"],
                         title="Amount vs Fraud Probability")
        st.plotly_chart(dark_chart(fig, 380), use_container_width=True)

    with col2:
        if "time" in df_f.columns:
            hour_risk = df_f.groupby("time")["fraud_probability"].mean().reset_index()
            fig = px.line(hour_risk, x="time", y="fraud_probability",
                          markers=True, title="Avg Fraud Probability by Hour")
            fig.add_vline(x=22, line_dash="dash", line_color="#ef4444", annotation_text="Night starts")
            st.plotly_chart(dark_chart(fig, 380), use_container_width=True)

    st.subheader("📋 Table")
    show_cols = [c for c in ["transaction_id","amount","time","location","device",
                              "transaction_type","is_new_user","prev_transactions",
                              "avg_amount","is_night","fraud_probability",
                              "risk_level","actual_class"] if c in df_f.columns]
    df_show = df_f[show_cols].copy()
    df_show["amount"]            = df_show["amount"].apply(lambda x: f"₹{x:,.0f}")
    df_show["fraud_probability"] = df_show["fraud_probability"].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_show, use_container_width=True, height=420)

    csv = df_f.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "fraud_results.csv", "text/csv")
