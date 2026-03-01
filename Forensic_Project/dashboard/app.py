# =========================================================
# GitHub Software Project Risk Analyzer – NEXT GEN UI
# =========================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from data_collection.github_analyzer import analyze_github_repo

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI GitHub Risk Analyzer",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS (NEXT LEVEL UI)
# ---------------------------------------------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.big-title {
    font-size: 42px;
    font-weight: 800;
    background: -webkit-linear-gradient(#00f5ff, #00ff9f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 30px rgba(0,0,0,0.3);
    transition: 0.3s ease-in-out;
}
.card:hover {
    transform: translateY(-5px);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#00f5ff,#00ff9f);
    border-radius: 10px;
    font-weight: bold;
    color: black;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#00ff9f,#00f5ff);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.markdown('<div class="big-title">🚀 AI-Powered GitHub Risk Analyzer</div>', unsafe_allow_html=True)
st.caption("Forensic-level failure risk detection using Machine Learning")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", "software_projects_final.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "prediction_model", "failure_model.pkl")

if not os.path.exists(DATASET_PATH):
    st.error("Dataset not found.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Model not found.")
    st.stop()

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "",
    ["📊 Dataset", "📈 Analytics", "🤖 Analyze Repo", "📑 Reports"]
)

# =========================================================
# DATASET PAGE
# =========================================================
if page == "📊 Dataset":

    st.markdown("## 📊 Historical Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card"><h3>Total Projects</h3><h1>{len(df)}</h1></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card"><h3>Successful</h3><h1>{(df["label"]==1).sum()}</h1></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card"><h3>Failed</h3><h1>{(df["label"]==0).sum()}</h1></div>', unsafe_allow_html=True)

    st.divider()
    st.dataframe(df, use_container_width=True)

# =========================================================
# ANALYTICS PAGE
# =========================================================
elif page == "📈 Analytics":

    st.markdown("## 📈 Risk Pattern Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(
            df,
            names="label",
            title="Project Outcome Distribution",
            color="label",
            color_discrete_map={1:"#00ff9f",0:"#ff4b4b"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x="months_inactive",
            y="open_issues",
            color="label",
            size="contributors",
            title="Inactivity vs Open Issues",
            color_discrete_map={1:"#00ff9f",0:"#ff4b4b"}
        )
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# ANALYZE REPO PAGE
# =========================================================
elif page == "🤖 Analyze Repo":

    st.markdown("## 🔍 Live GitHub Repository Analysis")

    github_url = st.text_input(
        "Enter GitHub Repository URL",
        placeholder="https://github.com/owner/repository"
    )

    if st.button("🔮 Analyze Project"):

        if not github_url:
            st.warning("Please enter a valid URL")
            st.stop()

        with st.spinner("Collecting GitHub intelligence..."):
            features = analyze_github_repo(github_url)

        feature_vector = [
            features["open_issues"],
            features["total_issues"],
            features["contributors"],
            features["months_inactive"]
        ]

        X = np.array([feature_vector], dtype=float)

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        failure_risk = proba[0] * 100
        confidence = max(proba) * 100

        st.divider()

        # Risk Status
        if failure_risk >= 70:
            st.error(f"🚨 HIGH FAILURE RISK — {failure_risk:.2f}%")
        elif failure_risk >= 40:
            st.warning(f"⚠️ MODERATE RISK — {failure_risk:.2f}%")
        else:
            st.success(f"✅ LOW RISK — {failure_risk:.2f}%")

        st.info(f"Model Confidence: {confidence:.2f}%")

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_risk,
            title={"text": "Failure Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4b4b"},
                "steps": [
                    {"range": [0, 40], "color": "#00ff9f"},
                    {"range": [40, 70], "color": "#ffaa00"},
                    {"range": [70, 100], "color": "#ff4b4b"},
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # Feature Breakdown
        st.markdown("### 📊 Feature Breakdown")

        feature_df = pd.DataFrame({
            "Metric": ["Open Issues", "Total Issues", "Contributors", "Months Inactive"],
            "Value": feature_vector
        })

        fig_bar = px.bar(
            feature_df,
            x="Metric",
            y="Value",
            color="Value",
            color_continuous_scale="Tealgrn"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# REPORT PAGE
# =========================================================
elif page == "📑 Reports":

    st.markdown("## 📑 Forensic Failure Intelligence")

    rules_path = os.path.join(ROOT_DIR, "report", "association_rules.csv")

    if os.path.exists(rules_path):
        rules = pd.read_csv(rules_path)
        st.dataframe(rules, use_container_width=True)
    else:
        st.warning("Association rules not generated yet.")

    st.markdown("""
    ### 🧠 Risk Interpretation Model

    - High open issues → Maintenance overload  
    - Long inactivity → Abandonment risk  
    - Low contributors → Bus-factor vulnerability  
    - Combined risk factors → Exponential failure probability  

    **Model:** Random Forest Classifier  
    **Domain:** Software Forensics & Mining Software Repositories  
    """)
