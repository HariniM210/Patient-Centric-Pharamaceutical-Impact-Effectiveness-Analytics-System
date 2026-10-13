# ===========================
# STREAMLIT APP (ENTERPRISE + ADMIN + DATABASE)
# ===========================

import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import datetime

st.set_page_config(
    page_title="Patient-Centric Pharmaceutical Impact & Effectiveness",
    layout="wide"
)

# ===========================
# IMPORT ENGINE
# ===========================

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from recommendation_engine import generate_recommendation

model = joblib.load("models/drug_effectiveness_model.pkl")

DATA_FILE = "analysis_history.csv"

# ===========================
# LOAD OR CREATE DATABASE
# ===========================

if os.path.exists(DATA_FILE):
    history_df = pd.read_csv(DATA_FILE)
else:
    history_df = pd.DataFrame(columns=[
        "Drug", "Condition",
        "Effectiveness %",
        "Impact %",
        "Severity",
        "Recommendation",
        "Timestamp"
    ])
    history_df.to_csv(DATA_FILE, index=False)

# ===========================
# PDF FUNCTION
# ===========================

def generate_pdf_report(result, review_text, drug_name, condition):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    style = styles["Normal"]

    elements.append(Paragraph("Patient-Centric Pharmaceutical Impact Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Drug: {drug_name}", style))
    elements.append(Paragraph(f"Condition: {condition}", style))
    elements.append(Paragraph(f"Review: {review_text}", style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Effectiveness: {result['effectiveness_probability'] * 100:.1f}%", style))
    elements.append(Paragraph(f"Impact Score: {result['drug_impact_index'] * 100:.1f}%", style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Recommendation: {result['final_recommendation']}", style))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Generated On: {datetime.datetime.now()}", style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===========================
# NAVIGATION
# ===========================

mode = st.sidebar.radio(
    "Select Mode",
    ["Clinical Prediction", "Analytics Comparison", "Admin Dashboard"]
)

# ===========================
# CLINICAL PREDICTION PAGE
# ===========================

if mode == "Clinical Prediction":

    st.title("🧬 Clinical Prediction Dashboard")
    st.divider()

    review_text = st.text_area("Patient Review", height=150)

    col1, col2 = st.columns(2)
    with col1:
        drug_name = st.text_input("Drug Name")
    with col2:
        condition = st.text_input("Condition")

    if st.button("🚀 Analyze Drug"):

        if review_text and drug_name and condition:

            with st.spinner("Running AI Engine..."):

                result = generate_recommendation(
                    review_text,
                    drug_name,
                    condition
                )

                # Save to CSV
                new_entry = {
                    "Drug": drug_name,
                    "Condition": condition,
                    "Effectiveness %": result["effectiveness_probability"] * 100,
                    "Impact %": result["drug_impact_index"] * 100,
                    "Severity": result["severity_level"],
                    "Recommendation": result["final_recommendation"],
                    "Timestamp": datetime.datetime.now()
                }

                df = pd.read_csv(DATA_FILE)
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)

            st.success("Analysis Complete & Saved to Database")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["effectiveness_probability"] * 100,
                number={'suffix': "%"},
                title={'text': "Effectiveness Probability"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig, width="stretch")

            # PDF Download
            pdf_buffer = generate_pdf_report(result, review_text, drug_name, condition)
            st.download_button(
                "📄 Download Report",
                pdf_buffer,
                "Clinical_Report.pdf",
                "application/pdf"
            )

        else:
            st.error("Please fill all fields.")

# ===========================
# ANALYTICS COMPARISON PAGE
# ===========================

if mode == "Analytics Comparison":

    st.title("📈 Analytics Comparison")
    st.divider()

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:
        st.warning("No data available.")
    else:

        st.dataframe(df, use_container_width=True)

        fig1 = px.bar(
            df,
            x="Drug",
            y="Impact %",
            color="Recommendation",
            title="Impact Comparison"
        )
        st.plotly_chart(fig1, width="stretch")

        fig2 = px.line(
            df,
            x="Timestamp",
            y="Effectiveness %",
            markers=True,
            title="Effectiveness Over Time"
        )
        st.plotly_chart(fig2, width="stretch")

        ranked = df.sort_values("Impact %", ascending=False)
        st.subheader("🏆 Ranking by Impact")
        st.dataframe(ranked, use_container_width=True)

# ===========================
# ADMIN DASHBOARD
# ===========================

if mode == "Admin Dashboard":

    st.title("🔐 Admin Dashboard")
    st.divider()

    password = st.text_input("Enter Admin Password", type="password")

    if password == "admin123":

        df = pd.read_csv(DATA_FILE)

        if len(df) == 0:
            st.warning("No data available.")
        else:

            st.success("Admin Access Granted")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Analyses", len(df))
            col2.metric("Avg Impact %", round(df["Impact %"].mean(), 2))
            col3.metric("Avg Effectiveness %", round(df["Effectiveness %"].mean(), 2))

            st.subheader("Top Performing Drug")
            best_drug = df.sort_values("Impact %", ascending=False).iloc[0]
            st.write(best_drug)

            st.subheader("Download Full Dataset")
            st.download_button(
                "⬇ Download CSV",
                df.to_csv(index=False),
                "Full_Clinical_Data.csv",
                "text/csv"
            )

            if st.button("🗑 Clear All History"):
                empty_df = df.iloc[0:0]
                empty_df.to_csv(DATA_FILE, index=False)
                st.success("Database Cleared")

    else:
        st.warning("Admin login required.")