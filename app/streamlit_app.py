"""
streamlit_app.py
----------------
AI-Based Phishing Website Detector – Streamlit Dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – allow imports from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Local modules (may raise ImportError if model not trained yet)
try:
    from feature_extraction import extract_features, get_feature_names
    FEATURE_EXTRACTION_OK = True
except ImportError as e:
    FEATURE_EXTRACTION_OK = False
    FE_ERROR = str(e)

try:
    from predict import PhishingPredictor, RISK_COLOR
    PREDICT_OK = True
except ImportError as e:
    PREDICT_OK = False
    PREDICT_ERROR = str(e)

try:
    from whois_lookup import get_domain_info
    WHOIS_OK = True
except ImportError:
    WHOIS_OK = False

try:
    from ssl_checker import check_ssl
    SSL_OK = True
except ImportError:
    SSL_OK = False

try:
    from database import log_prediction, get_stats, get_recent_predictions
    DB_OK = True
except ImportError:
    DB_OK = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark cybersecurity CSS theme
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Courier New', Courier, monospace;
    color: #e0e0e0;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 60%, #0a1628 100%);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #00ff88;
}

/* ---- Headers ---- */
h1 { color: #00ff88 !important; text-shadow: 0 0 20px #00ff8866; }
h2 { color: #00d4ff !important; }
h3 { color: #7ee787 !important; }

/* ---- Cards / containers ---- */
.cyber-card {
    background: rgba(13, 17, 23, 0.9);
    border: 1px solid #21262d;
    border-left: 3px solid #00ff88;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 10px 0;
}
.metric-card {
    background: rgba(0,212,255,0.05);
    border: 1px solid #00d4ff33;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

/* ---- Phishing / Legit labels ---- */
.phishing-label {
    background: linear-gradient(90deg, #ff000033, #ff444422);
    border: 1px solid #ff4444;
    color: #ff6666;
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 1.6rem;
    font-weight: bold;
    text-align: center;
    text-shadow: 0 0 10px #ff4444;
}
.legit-label {
    background: linear-gradient(90deg, #00ff8822, #00ff8811);
    border: 1px solid #00ff88;
    color: #00ff88;
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 1.6rem;
    font-weight: bold;
    text-align: center;
    text-shadow: 0 0 10px #00ff88;
}

/* ---- Risk tags ---- */
.risk-high   { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffa500; font-weight: bold; }
.risk-low    { color: #00ff88; font-weight: bold; }

/* ---- Input / Button overrides ---- */
.stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e0e0e0 !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00ff88, #00d4ff) !important;
    color: #0a0a1a !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 26px !important;
}
.stButton > button:hover {
    box-shadow: 0 0 15px #00ff8866 !important;
}

/* ---- Divider ---- */
hr { border-color: #21262d; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ AI Phishing Detector")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 URL Analyzer", "📊 Model Dashboard", "📋 Prediction Log"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#666;'>Built with Streamlit · Scikit-learn · XGBoost</small>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE: HOME
# ===========================================================================
if page == "🏠 Home":
    # Banner
    st.markdown(
        """
        <div style='text-align:center; padding: 30px 0 10px 0;'>
            <h1 style='font-size:2.8rem;'>🛡️ AI-Based Phishing Website Detector</h1>
            <p style='color:#8b949e; font-size:1.1rem; max-width:700px; margin:auto;'>
                Real-time machine-learning powered cybersecurity analysis.<br>
                Detects phishing URLs using advanced feature engineering and ensemble models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Stats row
    if DB_OK:
        stats = get_stats()
    else:
        stats = {"total_analyzed": 0, "threats_detected": 0, "avg_confidence": 0, "safe_urls": 0}

    # Load model accuracy from reports if available
    accuracy_display = "N/A"
    reports_path = ROOT / "reports" / "metrics.json"
    if reports_path.exists():
        import json
        with open(reports_path) as f:
            report = json.load(f)
        best = report.get("best_model", {})
        accuracy_display = f"{best.get('accuracy', 0)*100:.1f}%"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, icon in [
        (c1, "URLs Analyzed",    stats["total_analyzed"],  "🔗"),
        (c2, "Threats Detected", stats["threats_detected"], "⚠️"),
        (c3, "Model Accuracy",   accuracy_display,          "🎯"),
        (c4, "Safe URLs",        stats["safe_urls"],        "✅"),
    ]:
        col.markdown(
            f"""<div class='metric-card'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-size:1.8rem; color:#00ff88; font-weight:bold;'>{value}</div>
                <div style='color:#8b949e; font-size:0.9rem;'>{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Feature highlights
    st.markdown("## 🚀 Key Capabilities")
    cols = st.columns(3)
    features_info = [
        ("🤖 ML-Powered Analysis", "Ensemble of Random Forest & XGBoost with automatic best-model selection based on ROC-AUC score."),
        ("🔍 Deep URL Inspection", "Extracts 20+ URL-based features: length, entropy, special characters, subdomains, keyword detection."),
        ("🌐 Domain Intelligence", "Real-time WHOIS lookup, SSL certificate verification, and DNS record validation."),
        ("📊 Explainable AI", "Transparent reasoning: see exactly why a URL was flagged as phishing."),
        ("⚡ Real-Time Scoring", "Instant threat score 0–100 with Low / Medium / High risk classification."),
        ("📈 MLOps Monitoring", "SQLite prediction logging, model versioning, and retraining pipeline included."),
    ]
    for i, (title, desc) in enumerate(features_info):
        cols[i % 3].markdown(
            f"""<div class='cyber-card'>
                <strong style='color:#00d4ff;'>{title}</strong><br>
                <small style='color:#8b949e;'>{desc}</small>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.85rem;'>"
        "Navigate to <b style='color:#00ff88;'>URL Analyzer</b> to scan a website."
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE: URL ANALYZER
# ===========================================================================
elif page == "🔍 URL Analyzer":
    st.markdown("# 🔍 URL Analyzer")
    st.markdown("Enter a URL below to analyse it for phishing indicators.")

    # --- Input ---------------------------------------------------------------
    col_in, col_btn, col_clr = st.columns([5, 1, 1])
    with col_in:
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_btn = st.button("Analyze URL", use_container_width=True)
    with col_clr:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # --- Analysis ------------------------------------------------------------
    if analyze_btn and url_input:
        if not PREDICT_OK:
            st.error("⚠️ Model not loaded. Please run `python src/train.py` first to train the model.")
            st.stop()

        with st.spinner("🔬 Analysing URL …"):
            predictor = PhishingPredictor()
            try:
                result = predictor.predict(url_input)
            except FileNotFoundError:
                st.error("⚠️ No trained model found. Run `python src/train.py` to train the model first.")
                st.stop()

        # --- Prediction label ------------------------------------------------
        if result["is_phishing"]:
            st.markdown(
                f"<div class='phishing-label'>⚠️ {result['prediction']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='legit-label'>✅ {result['prediction']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 1: Confidence gauge + Risk card + Threat gauge ---------------
        col_g1, col_risk, col_g2 = st.columns(3)

        with col_g1:
            st.markdown("#### 🎯 Confidence Score")
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(result["confidence"] * 100, 1),
                number={"suffix": "%", "font": {"color": "#00ff88", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar":  {"color": "#00ff88" if not result["is_phishing"] else "#ff4444"},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 30],  "color": "#0d1f0d"},
                        {"range": [30, 70], "color": "#1f1a0d"},
                        {"range": [70, 100], "color": "#1f0d0d"},
                    ],
                },
                title={"text": "Confidence", "font": {"color": "#8b949e"}},
            ))
            fig_conf.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=230, margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        with col_risk:
            st.markdown("#### 🚨 Risk Classification")
            rc = result["risk_color"]
            risk = result["risk_level"]
            st.markdown(
                f"""<div style='background:rgba(0,0,0,0.4); border:2px solid {rc};
                    border-radius:12px; padding:28px; text-align:center; margin-top:10px;'>
                    <div style='font-size:2.5rem; color:{rc}; font-weight:bold;'>{risk}</div>
                    <hr style='border-color:{rc}44;'>
                    <div style='color:#8b949e;'>Phishing Probability</div>
                    <div style='font-size:1.6rem; color:{rc};'>{result['p_phishing']*100:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_g2:
            st.markdown("#### 🔥 Threat Score")
            ts = result["threat_score"]
            ts_color = "#ff4444" if ts > 70 else ("#ffa500" if ts > 30 else "#00ff88")
            fig_ts = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ts,
                number={"font": {"color": ts_color, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar":  {"color": ts_color},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 30],  "color": "#0d1f0d"},
                        {"range": [30, 70], "color": "#1f1a0d"},
                        {"range": [70, 100], "color": "#1f0d0d"},
                    ],
                },
                title={"text": "Threat Score / 100", "font": {"color": "#8b949e"}},
            ))
            fig_ts.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=230, margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # --- Row 2: Explainability + Feature breakdown -----------------------
        col_xai, col_feat = st.columns([1, 1])

        with col_xai:
            st.markdown("#### 🧠 Explainable AI – Detection Reasons")
            if result["is_phishing"] and result["reasons"]:
                for r in result["reasons"]:
                    st.markdown(
                        f"<div class='cyber-card' style='border-left-color:#ff4444;'>"
                        f"🚩 {r}</div>",
                        unsafe_allow_html=True,
                    )
            elif not result["is_phishing"]:
                st.markdown(
                    "<div class='cyber-card' style='border-left-color:#00ff88;'>"
                    "✅ URL passed all heuristic checks. No phishing indicators found."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No specific heuristic reasons triggered.")

        with col_feat:
            st.markdown("#### 📐 Extracted Features")
            feats = result["features"]
            feat_df = pd.DataFrame(
                [{"Feature": k.replace("_", " ").title(), "Value": v}
                 for k, v in feats.items()]
            )
            st.dataframe(
                feat_df,
                use_container_width=True,
                height=320,
                hide_index=True,
            )

        st.markdown("---")

        # --- Row 3: Charts ---------------------------------------------------
        col_bar, col_pie = st.columns(2)

        with col_bar:
            st.markdown("#### 📊 Feature Values")
            top_feats = {k: v for k, v in feats.items() if isinstance(v, (int, float)) and v > 0}
            top_feats = dict(sorted(top_feats.items(), key=lambda x: x[1], reverse=True)[:12])
            if top_feats:
                fig_bar = px.bar(
                    x=list(top_feats.keys()),
                    y=list(top_feats.values()),
                    labels={"x": "Feature", "y": "Value"},
                    color=list(top_feats.values()),
                    color_continuous_scale=["#00ff88", "#ffa500", "#ff4444"],
                )
                fig_bar.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_color="#8b949e", showlegend=False,
                    height=300, margin=dict(t=10, b=60),
                    coloraxis_showscale=False,
                    xaxis={"tickangle": -35},
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            st.markdown("#### 🥧 Risk Probability Distribution")
            p_legit   = round((1 - result["p_phishing"]) * 100, 1)
            p_phishing_pct = round(result["p_phishing"] * 100, 1)
            fig_pie = go.Figure(go.Pie(
                labels=["Legitimate", "Phishing"],
                values=[p_legit, p_phishing_pct],
                marker_colors=["#00ff88", "#ff4444"],
                hole=0.5,
                textinfo="label+percent",
                textfont={"color": "#e0e0e0"},
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0d1117",
                font_color="#8b949e",
                height=300,
                margin=dict(t=10, b=10),
                legend={"font": {"color": "#8b949e"}},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # --- Row 4: WHOIS + SSL (live lookup) --------------------------------
        st.markdown("---")
        st.markdown("#### 🌐 Real-Time Domain Intelligence")
        col_whois, col_ssl = st.columns(2)

        with col_whois:
            st.markdown("**WHOIS Lookup**")
            if WHOIS_OK:
                with st.spinner("Fetching WHOIS …"):
                    domain_info = get_domain_info(url_input)
                whois_data = {
                    "Domain":             domain_info.get("domain", "N/A"),
                    "DNS Exists":         "✅ Yes" if domain_info.get("dns_record_exists") else "❌ No",
                    "Registrar":          domain_info.get("registrar") or "N/A",
                    "Domain Age (days)":  domain_info.get("domain_age_days", "N/A"),
                    "Expiry (days left)": domain_info.get("domain_expiry_days", "N/A"),
                    "Creation Date":      str(domain_info.get("creation_date") or "N/A")[:10],
                    "Expiration Date":    str(domain_info.get("expiration_date") or "N/A")[:10],
                }
                for k, v in whois_data.items():
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;"
                        f"border-bottom:1px solid #21262d; padding:4px 0;'>"
                        f"<span style='color:#8b949e;'>{k}</span>"
                        f"<span style='color:#e0e0e0;'>{v}</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Install `python-whois` to enable WHOIS lookups.")

        with col_ssl:
            st.markdown("**SSL Certificate**")
            if SSL_OK:
                with st.spinner("Checking SSL …"):
                    ssl_info = check_ssl(url_input)
                ssl_data = {
                    "Valid Certificate": "✅ Yes" if ssl_info.get("ssl_valid") else "❌ No",
                    "Expired":           "⚠️ Yes" if ssl_info.get("ssl_expired") else "✅ No",
                    "Issuer":            ssl_info.get("ssl_issuer") or "N/A",
                    "Expiry Date":       str(ssl_info.get("ssl_expiry_date") or "N/A")[:10],
                    "Days Remaining":    ssl_info.get("ssl_days_remaining", "N/A"),
                }
                for k, v in ssl_data.items():
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;"
                        f"border-bottom:1px solid #21262d; padding:4px 0;'>"
                        f"<span style='color:#8b949e;'>{k}</span>"
                        f"<span style='color:#e0e0e0;'>{v}</span></div>",
                        unsafe_allow_html=True,
                    )
                if ssl_info.get("ssl_error"):
                    st.caption(f"ℹ️ {ssl_info['ssl_error']}")
            else:
                st.info("SSL checker module loaded but an import issue was detected.")

        # --- Log prediction --------------------------------------------------
        if DB_OK:
            log_prediction(
                url=url_input,
                prediction=result["prediction"],
                confidence=result["confidence"],
                risk_level=result["risk_level"],
                threat_score=result["threat_score"],
                is_phishing=result["is_phishing"],
            )

    elif analyze_btn and not url_input:
        st.warning("Please enter a URL to analyse.")


# ===========================================================================
# PAGE: MODEL DASHBOARD
# ===========================================================================
elif page == "📊 Model Dashboard":
    st.markdown("# 📊 Model Performance Dashboard")

    reports_path = ROOT / "reports" / "metrics.json"
    if not reports_path.exists():
        st.warning(
            "No metrics found. Train the model first:\n\n"
            "```bash\npython src/train.py\n```"
        )
        st.stop()

    import json
    with open(reports_path) as f:
        report = json.load(f)

    models = report.get("models", [])
    best   = report.get("best_model", {})

    # Best model highlight
    st.markdown(
        f"""<div class='cyber-card'>
            🏆 <strong style='color:#00ff88;'>Best Model: {best.get('model','N/A')}</strong>
            &nbsp;·&nbsp; ROC-AUC: <strong>{best.get('roc_auc','N/A')}</strong>
            &nbsp;·&nbsp; Accuracy: <strong>{best.get('accuracy','N/A')}</strong>
        </div>""",
        unsafe_allow_html=True,
    )

    # Metrics comparison table
    st.markdown("### Model Comparison")
    if models:
        rows = []
        for m in models:
            rows.append({
                "Model":     m["model"],
                "Accuracy":  f"{m['accuracy']*100:.2f}%",
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall":    f"{m['recall']*100:.2f}%",
                "F1 Score":  f"{m['f1_score']*100:.2f}%",
                "ROC-AUC":   f"{m['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart comparison
    if len(models) > 1:
        st.markdown("### Metrics Comparison")
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        fig = go.Figure()
        for m in models:
            fig.add_trace(go.Bar(
                name=m["model"],
                x=[k.replace("_", " ").title() for k in metric_keys],
                y=[m[k] for k in metric_keys],
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#8b949e", height=350,
            legend={"font": {"color": "#8b949e"}},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if Random Forest is available)
    rf_path = ROOT / "models" / "random_forest.pkl"
    meta_path = ROOT / "models" / "model_meta.json"
    if rf_path.exists() and meta_path.exists():
        import pickle
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", [])

        if hasattr(rf_model, "feature_importances_") and feature_names:
            st.markdown("### 🌲 Random Forest – Feature Importance")
            importance = rf_model.feature_importances_
            idx = np.argsort(importance)[::-1][:15]
            feat_imp_df = pd.DataFrame({
                "Feature":    [feature_names[i].replace("_", " ").title() for i in idx],
                "Importance": [importance[i] for i in idx],
            })
            fig_imp = px.bar(
                feat_imp_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#00ff88", "#ffa500", "#ff4444"],
            )
            fig_imp.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#8b949e", height=420,
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ===========================================================================
# PAGE: PREDICTION LOG
# ===========================================================================
elif page == "📋 Prediction Log":
    st.markdown("# 📋 Prediction History")

    if not DB_OK:
        st.error("Database module not available.")
        st.stop()

    stats = get_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analyzed",   stats["total_analyzed"])
    c2.metric("Threats Detected", stats["threats_detected"])
    c3.metric("Safe URLs",        stats["safe_urls"])

    st.markdown("---")
    records = get_recent_predictions(100)
    if not records:
        st.info("No predictions logged yet. Analyse some URLs first!")
    else:
        df_log = pd.DataFrame(records)
        # Style is_phishing column
        df_log["is_phishing"] = df_log["is_phishing"].map({1: "⚠️ Phishing", 0: "✅ Legitimate"})
        df_log["confidence"]  = df_log["confidence"].apply(lambda x: f"{x*100:.1f}%")
        df_log = df_log[["timestamp", "url", "is_phishing", "confidence", "risk_level", "threat_score"]]
        df_log.columns = ["Timestamp", "URL", "Prediction", "Confidence", "Risk Level", "Threat Score"]
        st.dataframe(df_log, use_container_width=True, hide_index=True, height=500)

        # Distribution chart
        st.markdown("### 📈 Prediction Distribution")
        counts = df_log["Prediction"].value_counts().reset_index()
        counts.columns = ["Type", "Count"]
        fig_dist = px.pie(
            counts, names="Type", values="Count",
            color="Type",
            color_discrete_map={
                "✅ Legitimate": "#00ff88",
                "⚠️ Phishing":  "#ff4444",
            },
            hole=0.45,
        )
        fig_dist.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#8b949e",
            height=350,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
